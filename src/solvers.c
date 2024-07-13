/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 */
#include "solvers.h"
#include "ids.h"

void solver_free(Solver* S)
{
	vec_for(S->ex,i,0) ltensor_free(&S->ex[i]);
	vec_free(S->ex);
	ltensor_free(&S->dx);
}

int solver_euler_step(Solver* S, float dt, LocalTensor* x)
{
	TRYR( solver_dxdt(S, S->t, x, &S->dx) );
	ltensor_for(*x,i,0) x->d[i] += S->dx.d[i] * dt;
	return 1;
}

const SolverClass g_solver_euler = {
	.step = solver_euler_step,
	.n_fe = 1,
	.name = ID_euler,
};

int solver_heun_step(Solver* S, float dt, LocalTensor* x)
{
	if (S->n_step == 0) {
		vec_resize_zero(S->ex, 2);
		ltensor_resize_like(&S->ex[0], x);
		ltensor_resize_like(&S->ex[1], x);
	}
	LocalTensor x1 = S->ex[0],
	            d1 = S->ex[1];

	TRYR( solver_dxdt(S, S->t, x, &S->dx) );
	ltensor_for(*x,i,0) x1.d[i] = x->d[i] + S->dx.d[i] * dt;

	int r = solver_dxdt(S, S->t+dt, &x1, &d1);
	if (r < 0) return r;
	if (r > 0)
		ltensor_for(*x,i,0)
			x->d[i] += (S->dx.d[i] + d1.d[i]) * 0.5 * dt;
	
	return 1;
}

const SolverClass g_solver_heun = {
	.step = solver_heun_step,
	.n_fe = 2,
	.name = ID_heun,
};

int solver_taylor3_step(Solver* S, float dt, LocalTensor* x)
{
	if (S->n_step == 0) {
		vec_resize_zero(S->ex, 3);
		ltensor_resize(&S->ex[0], 2,1,1,1);
		ltensor_resize_like(&S->ex[1], x);
		ltensor_resize_like(&S->ex[2], x);
	}
	float *dt_prev = S->ex[0].d,
	      *dp1 = S->ex[1].d,
		  *dp2 = S->ex[2].d;

	TRYR( solver_dxdt(S, S->t, x, &S->dx) );
	ltensor_for(*x,i,0) x->d[i] += S->dx.d[i] * dt;

	if (S->n_step >= 1) {
		float dtp = dt_prev[0],
			  f2 = 0.5*dt*dt / dtp;
		ltensor_for(*x,i,0) {
			float d2 = (S->dx.d[i] - dp1[i]) * f2;
			x->d[i] += d2;
		}
		if (S->n_step >= 2) {
			float dtpp = dt_prev[1],
				  f2p = 0.5*dt*dt / dtpp;
			f2  *= dt/3/dtp;
			f2p *= dt/3/dtp;
			ltensor_for(*x,i,0) {
				float d2  = (S->dx.d[i] - dp1[i]) * f2,
					  d2p = (    dp1[i] - dp2[i]) * f2p;
				x->d[i] += d2 + d2p;
			}
		}
	}
	
	ltensor_for(*x,i,0) dp2[i] = dp1[i];
	ltensor_for(*x,i,0) dp1[i] = S->dx.d[i];
	dt_prev[1] = dt_prev[0];
	dt_prev[0] = dt;
	return 1;
}

const SolverClass g_solver_taylor3 = {
	.step = solver_taylor3_step,
	.n_fe = 1,
	.name = ID_taylor3,
};

const SolverClass *g_solvers[] = {
	&g_solver_euler,
	&g_solver_heun,
	&g_solver_taylor3,
	NULL
};
