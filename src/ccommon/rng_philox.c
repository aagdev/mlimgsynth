/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#include "rng_philox.h"
#include <math.h>

RngPhilox g_rng;

const uint32_t philox_m[2] = {0xD2511F53, 0xCD9E8D57};
const uint32_t philox_w[2] = {0x9E3779B9, 0xBB67AE85};

const double two_pow32_inv     = 2.3283064365386963e-10; //   1/2^32
const double two_pow32_inv_2pi = 1.4629180792671596e-09; // 2pi/2^32

static inline
double box_muller(double x, double y)
{
	double u = (x + 0.5) * two_pow32_inv;  
	double v = (y + 0.5) * two_pow32_inv_2pi;
	return sqrt(-2.0 * log(u)) * sin(v);
}

void rng_philox_randn(RngPhilox* S, unsigned n, float* out)
{
	uint32_t cnt[4], key[2];
	for (unsigned i=0; i<n; ++i) {
		cnt[0] = S->offset;
		cnt[1] = 0;
		cnt[2] = i;
		cnt[3] = 0;

		key[0] = S->seed;
		key[1] = S->seed>>32;

		for (unsigned r=0; r<10; ++r) {
			// Round
			uint64_t v1 = (uint64_t)cnt[0] * philox_m[0];
			uint64_t v2 = (uint64_t)cnt[2] * philox_m[1];
			cnt[0] = (uint32_t)(v2>>32) ^ cnt[1] ^ key[0];
			cnt[1] = v2;
			cnt[2] = (uint32_t)(v1>>32) ^ cnt[3] ^ key[1];
			cnt[3] = v1;

			key[0] += philox_w[0];
			key[1] += philox_w[1];
		}

		out[i] = box_muller(cnt[0], cnt[1]);
	}
	S->offset++;
}
