/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * Initial value problem (IVP) solvers.
 * Used as sampling methods for generative diffusion models.
 *
 * Example:

Solver sol={ .C=&solver_euler };
sol.t = 0; //initial
ltensor_resize_zero(&sol.x, 100,1,1,1);
sol.x.d[50] = 1;

int dxdt(Solver* S, float t, const LocalTensor* x, LocalTensor* dx) {
	unsigned n = x->s[0];
	for (unsigned i=1; i+1<n; ++i)
		dx->d[i] = (x->d[i-1] -2*x->d[i] + x->d[i+1]) / 4;
	dx->d[0] = dx->d[n-1] = 0;
	return 1;
}

sol.dxdt = dxdt;

for (float dt=0.1, t_end=10, t=dt; t<=t_end; t+=dt)
	TRY( solver_step(&sol, t) );

TRY( ltensor_save_path(&sol.x, "result.tensor") );
 */
#pragma once
#include "localtensor.h"

struct Solver;

typedef struct {
	int (*step)(struct Solver*, float dt, LocalTensor* x);
	int n_fe;  //number of calls to dxdt per step
	int name;  //string id
} SolverClass;

extern const SolverClass g_solver_euler;
extern const SolverClass g_solver_heun;
extern const SolverClass g_solver_taylor3;
extern const SolverClass g_solver_dpmpp2m;
extern const SolverClass g_solver_dpmpp2s;

extern const SolverClass *g_solvers[];  //list of all available solvers

typedef struct Solver {
	const SolverClass *C;

	// State
	LocalTensor dx,
	            tmp[8];  //vector, temporal tensors
	float t;
	unsigned i_step, i_tmp;

	// Config (fill before use)
	int (*dxdt)(struct Solver*, float t, const LocalTensor* x, LocalTensor* dx);
	void *user;
} Solver;

void solver_free(Solver* S);

int solver_init(Solver* S, int clsname);

int solver_step(Solver* S, float t, LocalTensor* x);

static inline
int solver_dxdt(Solver* S, float t, const LocalTensor* x, LocalTensor* dx)
{
	assert( ltensor_shape_equal(x, dx) );
	int r = S->dxdt(S, t, x, dx);
	if (r < 0) return r;
	return r;
}

// Internal use

static inline
LocalTensor* solver_tmp_get(Solver* S)
{
	assert( S->i_tmp < COUNTOF(S->tmp) );
	S->i_tmp++;
	return &S->tmp[S->i_tmp-1];
}

static inline
LocalTensor* solver_tmp_get_resize(Solver* S, int n0, int n1, int n2, int n3)
{
	LocalTensor* lt = solver_tmp_get(S);
	ltensor_resize(lt, n0, n1, n2, n3);
	return lt;
}

static inline
LocalTensor* solver_tmp_get_resize_like(Solver* S, const LocalTensor* x)
{
	LocalTensor* lt = solver_tmp_get(S);
	ltensor_resize_like(lt, x);
	return lt;
}
