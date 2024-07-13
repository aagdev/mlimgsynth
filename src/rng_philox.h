/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * Pseudo-random number generator imitating torch cuda randn.
 * Based on: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/rng_philox.py
 */
#pragma once
#include <stdint.h>

typedef struct {
    uint64_t seed;
    uint32_t offset;	
} RngPhilox;

void rng_philox_randn(RngPhilox* S, unsigned n, float* out);

extern RngPhilox g_rng;

static inline
void rng_randn(unsigned n, float* out) {
	rng_philox_randn(&g_rng, n, out);
}
