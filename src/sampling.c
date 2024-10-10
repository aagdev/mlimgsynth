/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 */
#include "sampling.h"
#include "ids.h"
#include "ccommon/ccommon.h"
#include "ccommon/rng_philox.h"
#include "ccommon/logging.h"
#include <math.h>

#define log_vec(LVL,DESC,VEC,VAR,I0,...) \
if (log_level_check(LVL)) { \
	log_line_begin(LVL); \
	log_line_str(DESC ":"); \
	vec_for(VEC,VAR,I0) log_line_strf(" " __VA_ARGS__); \
	log_line_end(); \
}

#define log_debug_vec(...)  log_vec(LOG_LVL_DEBUG, __VA_ARGS__)

void dnsamp_free(DenoiseSampler* S)
{
	ltensor_free(&S->noise);
	ltensor_free(&S->x0);
	solver_free(&S->solver);
	vec_free(S->sigmas);
}

int dnsamp_init(DenoiseSampler* S)
{
	int R=1;

	//if (!(0 <= S->c.s_noise && S->c.s_noise <= 1))
	//	ERROR_LOG(-1, "snoise out of range");

	// Solver
	if (!S->c.method) {
		if (S->c.s_noise > 0 || S->c.s_ancestral > 0)
			S->c.method = ID_euler;
		else
			S->c.method = ID_taylor3;
	}
	TRY( solver_init(&S->solver, S->c.method) );
	
	// Scheduling
	// Compute times and sigmas for inference
	S->n_step = S->c.n_step;
	if (S->n_step < 1) S->n_step = 12;

	S->nfe_per_step = S->solver.C->n_fe;
	// Reduce number of steps to keep the number of neural function evaluations
	if (S->nfe_per_step > 1)
		S->n_step = (S->n_step + S->nfe_per_step-1) / S->nfe_per_step;
	
	S->nfe_per_step *= S->nfe_per_dxdt;
	
	vec_resize(S->sigmas, S->n_step+1);
	S->sigmas[S->n_step] = 0;

	IFNPOSSET(S->c.f_t_ini, 1);
	float t_ini = (S->unet_p->n_step_train - 1) * S->c.f_t_ini;
	float t_end = (S->unet_p->n_step_train - 1) * S->c.f_t_end;

	IFFALSESET(S->c.sched, ID_uniform);
	switch (S->c.sched) {
	case ID_uniform: {
		float b = t_ini,
		      f = S->n_step>1 ? (t_end-t_ini)/(S->n_step-1) : 0;
		for (unsigned i=0; i<S->n_step; ++i)
			S->sigmas[i] = unet_t_to_sigma(S->unet_p, b+i*f);
	} break;
	case ID_karras: {
		// Uses the model's min and max sigma instead of 0.1 and 10.
		float smin = unet_t_to_sigma(S->unet_p, t_end),
		      smax = unet_t_to_sigma(S->unet_p, t_ini),
			  p=7,
		      sminp = pow(smin, 1/p),
		      smaxp = pow(smax, 1/p),
			  b = smaxp,
			  f = S->n_step>1 ? (sminp - smaxp) / (S->n_step-1) : 0;
		for (unsigned i=0; i<S->n_step; ++i)
			S->sigmas[i] = pow(b+i*f, p);
	} break;
	default:
		ERROR_LOG(-1, "Unknown scheduler '%s'", id_str(S->c.sched));
	}

	//log_debug_vec("Times" , times , i, 0, "%.6g", times[i]);
	log_debug_vec("Sigmas", S->sigmas, i, 0, "%.6g", S->sigmas[i]);
	
	S->solver.t = S->sigmas[0];  //initial t
	S->i_step = 0;

end:
	return R;
}

void dnsamp_mask_apply(DenoiseSampler* S, LocalTensor* x)
{
	int n0 = x->s[0], n1 = x->s[1], n2 = x->s[2],
		s1 = n0, s2 = n0*n1;
	assert( ltensor_shape_check(S->c.lmask, n0, n1, 1, 1) );
	for (int i2=0; i2<n2; ++i2)
	for (int i1=0; i1<n1; ++i1)
	for (int i0=0; i0<n0; ++i0) {
		float m = S->c.lmask->d[i0 +i1*s1];
		int i = i0 +i1*s1 +i2*s2;
		x->d[i] = S->x0.d[i] * m + x->d[i] * (1-m);
	}
}

void dnsamp_noise_add(DenoiseSampler* S, LocalTensor* x, float sigma)
{
	ltensor_resize_like(&S->noise, x);
	rng_randn(ltensor_nelements(&S->noise), S->noise.d);
	ltensor_for(*x,i,0) x->d[i] += S->noise.d[i] * sigma;
}

int dnsamp_step(DenoiseSampler* S, LocalTensor* x)
{
	int R=1;

	int s = S->i_step;
	if (!(s < S->n_step)) return 0;
	
	float s_up = 0,
	      s_down = S->sigmas[s+1];

	if (s == 0) {  // Initial tasks
		if (S->c.lmask) ltensor_copy(&S->x0, x);

		// Add noise to initial latent
		dnsamp_noise_add(S, x, S->sigmas[0]);
		if (S->c.lmask) dnsamp_mask_apply(S, x);
		log_debug3_ltensor(x, "x0+noise");
	}

	if (S->c.s_noise > 0 && s > 0) {
		// Stochastic sampling: may help to add detail lost during sampling
		// Ref.: Karras2022, see Algo2 with S_churn
		// Produces softer images
		// Similar to the ancestral sampling below
		float s_curr  = S->sigmas[s],
		      s_hat   = s_curr * sqrt(2) * S->c.s_noise,
			  s_noise = sqrt(s_hat*s_hat - s_curr*s_curr);
		log_debug("s_noise:%g s_hat:%g", s_noise, s_hat);
		
		dnsamp_noise_add(S, x, s_noise);
		if (S->c.lmask) dnsamp_mask_apply(S, x);
		S->solver.t = s_hat;
	}
		
	if (S->c.s_ancestral > 0) {
		// Ancestral sampling
 		// Ref.: k_diffusion/sampling.py  get_ancestral_step
		// Produces softer images
		float s1 = S->sigmas[s],  //sigma_from
			  s2 = S->sigmas[s+1];  //sigma_to
		
		s_up = sqrt((s2*s2) * (s1*s1 - s2*s2) / (s1*s1));
		s_up *= S->c.s_ancestral;  //eta * s_noise
		MINSET(s_up, s2);
		s_down = sqrt(s2*s2 - s_up*s_up);

		log_debug("ancestral s_down:%g s_up:%g", s_down, s_up);
	}

	TRY( solver_step(&S->solver, s_down, x) );
	
	if (s_up > 0 && s+1 != S->n_step) {
		// Ancestral sampling
		dnsamp_noise_add(S, x, s_up);
		S->solver.t = S->sigmas[s+1];
	}

	// In-painting mask apply
	if (S->c.lmask)
		dnsamp_mask_apply(S, x);
	
	log_debug3_ltensor(x, "x");

	S->i_step++;
end:
	return R;
}
