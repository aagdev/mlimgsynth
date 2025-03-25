/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * Initial value problem (IVP) solvers.
 * Used as sampling methods for generative diffusion models.
 *
 * Example:

int dxdt(Solver* S, float t, const LocalTensor* x, LocalTensor* dx) {
	unsigned n = x->n[0];
	for (unsigned i=1; i+1<n; ++i)
		dx->d[i] = (x->d[i-1] -2*x->d[i] + x->d[i+1]) / 4;
	dx->d[0] = dx->d[n-1] = 0;
	return 1;
}

void solve() {
	// Set solver
	Solver sol={ .C=&solver_euler };
	// Set initial time
	sol.t = 0; 
	// Set initial state
	ltensor_resize_zero(&sol.x, 100,1,1,1);
	sol.x.d[50] = 1;
	// Set differential equation
	sol.dxdt = dxdt;
	// Solve until t_end=10
	for (float dt=0.1, t_end=10, t=dt; t<=t_end; t+=dt)
		TRY( solver_step(&sol, t) );
	// Do something here with the result in sol.x .
	// You may reuse the solver by setting i_step to zero.
	// Free memory
	solver_free(&sol);
}
 */
#pragma once
#include "localtensor.h"

struct Solver;

typedef struct {
	int (*step)(struct Solver*, float dt, LocalTensor* x);
	int n_fe;  //number of calls to dxdt per step
	const char *name;
} SolverClass;

// Default methods
extern const SolverClass g_solver_euler;
extern const SolverClass g_solver_heun;
extern const SolverClass g_solver_taylor3;
extern const SolverClass g_solver_dpmpp2m;
extern const SolverClass g_solver_dpmpp2s;

enum {
	SOLVER_METHOD_EULER		= 1,
	SOLVER_METHOD_HEUN		= 2,
	SOLVER_METHOD_TAYLOR3	= 3,
	SOLVER_METHOD_DPMPP2M	= 4,
	SOLVER_METHOD_DPMPP2S	= 5,
};

const SolverClass* solver_class_get(int idx);  //idx >= 1
const SolverClass* solver_class_find(const char* name);

typedef struct Solver {
	const SolverClass *C;  // Fill before using

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

int solver_step(Solver* S, float t, LocalTensor* x);

static inline
int solver_dxdt(Solver* S, float t, const LocalTensor* x, LocalTensor* dx)
{
	assert( ltensor_shape_equal(x, dx) );
	int r = S->dxdt(S, t, x, dx);
	if (r < 0) return r;
	return r;
}
