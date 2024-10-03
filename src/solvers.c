/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 */
#include "solvers.h"
#include "ids.h"
#include "ccommon/ccommon.h"
#include <math.h>

void solver_free(Solver* S)
{
	for (unsigned i=0; i<COUNTOF(S->tmp); ++i)
		ltensor_free(&S->tmp[i]);
	ltensor_free(&S->dx);
}

int solver_init(Solver* S, int clsname)
{
	int R=1;
	const SolverClass *cls=NULL;
	for (unsigned i=0; g_solvers[i]; ++i) {
		if (clsname == g_solvers[i]->name) {
			cls = g_solvers[i];
		}
	}
	if (!cls) ERROR_LOG(-1, "Unknown solver '%s'", id_str(clsname));
	S->C = cls;
	S->i_step = 0;
end:
	return R;
}

int solver_step(Solver* S, float t, LocalTensor* x)
{
	S->i_tmp = 0;
	ltensor_resize_like(&S->dx, x);
	int r = S->C->step(S, t, x);
	if (r < 0) return r;
	S->t = t;
	S->i_step++;
	return r;
}

/* Euler
 * Ref.: any textbook
 * Baseline.
 */
int solver_euler_step(Solver* S, float t, LocalTensor* x)
{
	float dt = t - S->t;
	TRYR( solver_dxdt(S, S->t, x, &S->dx) );
	ltensor_for(*x,i,0) x->d[i] += S->dx.d[i] * dt;
	return 1;
}

const SolverClass g_solver_euler = {
	.step = solver_euler_step,
	.n_fe = 1,
	.name = ID_euler,
};

/* Heun (improved Euler)
 * Ref.: Karras et al. 2022 "Elucidating..." Algo1
 * Tends to distort the images with low step counts.
 */
int solver_heun_step(Solver* S, float t, LocalTensor* x)
{
	float dt = t - S->t;
	LocalTensor *x1 = solver_tmp_get_resize_like(S, x);
	LocalTensor *d1 = solver_tmp_get_resize_like(S, x);

	TRYR( solver_dxdt(S, S->t, x, &S->dx) );
	ltensor_for(*x,i,0) x1->d[i] = x->d[i] + S->dx.d[i] * dt;

	if (!(t > 0)) {  //last step: just euler
		ltensor_for(*x,i,0) x->d[i] = x1->d[i];
	}
	else {  //2nd order correction
		TRYR( solver_dxdt(S, t, x1, d1) );
		ltensor_for(*x,i,0)
			x->d[i] += (S->dx.d[i] + d1->d[i]) * 0.5 * dt;
	}
	
	return 1;
}

const SolverClass g_solver_heun = {
	.step = solver_heun_step,
	.n_fe = 2,
	.name = ID_heun,
};

/* Third-order-Taylor extension of Euler
 * Ref.: own
 * Similar to Euler with less steps.

x_{i+1} = x_i + dx_i dt_i + (1/2) dx2_i dt_i^2 + (1/6) dx3_i dt_i^3

dx2_i = (dx_i - dx_{i-1}) / dt_{i-1}
dx3_i = (dx2_i - dx2_{i-1}) / dt_{i-1}
 */
int solver_taylor3_step(Solver* S, float t, LocalTensor* x)
{
	float dt = t - S->t;
	LocalTensor *lt_dt_prev = solver_tmp_get_resize(S, 1,1,1,1);
	LocalTensor *lt_dp1 = solver_tmp_get_resize_like(S, x);
	LocalTensor *lt_dp2 = solver_tmp_get_resize_like(S, x);

	float *dt_prev = lt_dt_prev->d,
	      *dp1 = lt_dp1->d,
		  *dp2 = lt_dp2->d;

	TRYR( solver_dxdt(S, S->t, x, &S->dx) );
	ltensor_for(*x,i,0) x->d[i] += S->dx.d[i] * dt;
	
	// 2nd and 3nd order corrections
	float idtp = 1 / dt_prev[0],
	      f2 = S->i_step >= 1 ? dt*dt/2 : 0,
		  f3 = S->i_step >= 2 ? dt*dt*dt/6 : 0;
	ltensor_for(*x,i,0) {
		float d2 = (S->dx.d[i] - dp1[i]) * idtp,
			  d3 = (d2 - dp2[i]) * idtp;
		x->d[i] += d2 * f2 + d3 * f3;
		dp1[i] = S->dx.d[i];
		dp2[i] = d2;
	}
	
	dt_prev[0] = dt;
	return 1;
}

const SolverClass g_solver_taylor3 = {
	.step = solver_taylor3_step,
	.n_fe = 1,
	.name = ID_taylor3,
};

/* DPM++(2M)
 * Ref.: Lu et al. 2023 "DPM-Solver++ ..." Algo2
 * Ref.: k-diffusion/sampling.py  sample_dpmpp_2m
 * Produces sharper images.
 * Use with Karras scheduler to prevent overly sharp images.

alpha_i     = 1
sigma_{i+1} = t
sigma_i     = S->t

lambda_i = log(alpha_i / sigma_i)
         = -log(sigma_i)

a_i = sigma_{i+1} / sigma_i

h_i = lambda_{i+1} - lambda_i
    = -log(sigma_{i+1} / sigma_i)
	= -log(a_i)

b_i = exp(-h_i) - 1 = a_i - 1

c_i = 1/(2r)
    = h_{i} / (2 h_{i-1})

d_i = x_i - sigma_i dx_i

D_i = (1 + c_i) d_i - c_i d_{i-1}

x_{i+1} = a_i x_i - b_i D_i
        = a_i x_i + (1 - a_i) D_i

if c_i == 0:
	x_{i+1} = x_i + (sigma_{i+1} - sigma_i) dx_i   (Euler)
 */
int solver_dpmpp2m_step(Solver* S, float t, LocalTensor* x)
{
	LocalTensor *vars = solver_tmp_get_resize(S, 1,1,1,1);
	LocalTensor *dprev = solver_tmp_get_resize_like(S, x);

	float a = t / S->t,
		  h = -log(a),
		  h_last = vars->d[0],
		  c = h / (2*h_last);

	if (S->i_step == 0 || !(t > 0))  //first or last step
		c = 0;

	TRYR( solver_dxdt(S, S->t, x, &S->dx) );
	ltensor_for(*x,i,0) {
		float d0 = x->d[i] - S->t * S->dx.d[i],
		      d1 = dprev->d[i],
		      d  = (1+c) * d0 - c * d1;
		x->d[i] = a * x->d[i] + (1-a) * d;
		dprev->d[i] = d0;
	}

	vars->d[0] = h;
	return 1;
}

const SolverClass g_solver_dpmpp2m = {
	.step = solver_dpmpp2m_step,
	.n_fe = 1,
	.name = ID_dpmpp2m,
};

/* DPM++(2S)
 * Ref.: Lu et al. 2023 "DPM-Solver++ ..." Algo1
 * Ref.: k-diffusion/sampling.py  sample_dpmpp_2s_ancestral
 * Should be used with ancestral sampling.

Check DPM++(2M) first.

lambda_i = -log(sigma_i)

s_i = sqrt(sigma_{i+1} sigma_i)   From k-diffusion r=1/2

a'_i = s_i / sigma_i
h'_i = -log(a'_i)
d_i  = x_i - sigma_i dx_i

x'_i = a'_i x_i + (1 - a'_i) d_i
	 = x_i + (s_i - sigma_i) dx_i

d'_i = x'_i - s_i dx'_i

a_i = sigma_{i+1} / sigma_i
h_i = -log(a_i)

x_{i+1} = a_i x_i + (1 - a_i) d'_i
 */
int solver_dpmpp2s_step(Solver* S, float t, LocalTensor* x)
{
	LocalTensor *x1  = solver_tmp_get_resize_like(S, x);
	LocalTensor *dx1 = solver_tmp_get_resize_like(S, x);
	
	TRYR( solver_dxdt(S, S->t, x, &S->dx) );

	if (!(t > 0)) {  //last step: just euler
		float dt = t - S->t;
		ltensor_for(*x,i,0) x->d[i] += S->dx.d[i] * dt;
	}
	else {
		float t1 = sqrt(t * S->t),
			  dt1 = t1 - S->t,
			  a = t / S->t;

		ltensor_for(*x,i,0) x1->d[i] = x->d[i] + S->dx.d[i] * dt1;
	
		TRYR( solver_dxdt(S, t1, x1, dx1) );
		ltensor_for(*x,i,0) {
			float d = x1->d[i] - t1 * dx1->d[i];
			x->d[i] = a * x->d[i] + (1-a) * d;
		}
	}

	return 1;
}

const SolverClass g_solver_dpmpp2s = {
	.step = solver_dpmpp2s_step,
	.n_fe = 2,
	.name = ID_dpmpp2s,
};

/* */

const SolverClass *g_solvers[] = {
	&g_solver_euler,
	&g_solver_heun,
	&g_solver_taylor3,
	&g_solver_dpmpp2m,
	&g_solver_dpmpp2s,
	NULL
};
