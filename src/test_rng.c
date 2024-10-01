/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * Utility to test the Philox RNG.
 */
#include "ccommon/timing.h"
#include "ccommon/rng_philox.h"
#include <stdlib.h>
#include <stdio.h>

/* Seed: 0, Offset: 0, n: 12
 -0.92466259
 -0.42534414
 -2.64384580
  0.14518388
 -0.12086648
 -0.57972562
 -0.62285119
 -0.32838708
 -1.07454228
 -0.36314407
 -1.67105067
  2.26550508
*/

int main(int argc, char* argv[])
{
	RngPhilox rng={0};
	unsigned n=12;
	
	if (argc > 1) rng.seed = strtoull(argv[1], NULL, 10);
	if (argc > 2) rng.offset = strtoul(argv[2], NULL, 10);
	if (argc > 3) n = strtoul(argv[3], NULL, 10);

	float *out = malloc(sizeof(float)*n);
	if (!out) { printf("out of memory\n"); return 1; }
	
	double t = timing_time();
	rng_philox_randn(&rng, n, out);
	t = timing_time() - t;
	fprintf(stderr, "%d numbers in %.3fms (%.3fns/num)\n", n, t*1e3, t*1e9/n);
	for (unsigned i=0; i<n; ++i) printf("%12.8f\n", out[i]);
	
	return 0;
}
