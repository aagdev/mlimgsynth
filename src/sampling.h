/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "unet.h"
#include "solvers.h"
#include "localtensor.h"

typedef struct {
	Solver solver;
	float *sigmas;  //vector
	int i_step, n_step, nfe_per_step;
	
	const UnetParams *unet_p;  //fill before use
	int nfe_per_dxdt;  //fill before use

	LocalTensor noise, x0;

	struct {
		int n_step, method, sched;
		float f_t_ini, f_t_end, s_noise, s_ancestral;
		LocalTensor *lmask;
	} c;
} DenoiseSampler;

void dnsamp_free(DenoiseSampler* S);

int dnsamp_init(DenoiseSampler* S);

int dnsamp_step(DenoiseSampler* S, LocalTensor* x);

static inline
int dnsamp_sample(DenoiseSampler* S, LocalTensor* x)
{
	int r;
	while ((r = dnsamp_step(S, x)) > 0) ;
	return r;
}
