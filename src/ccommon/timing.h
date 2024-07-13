/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#pragma once

// Get a monotonic time measured in seconds
double timing_time();

void timing_sleep(double dt);

static inline
double timing_tic(double* t_last) {
	double t=timing_time(), dt=t-*t_last;
	*t_last = t;
	return dt;
}

// Get the current number of seconds since 1970-01-01 00:00:00 (UTC).
double timing_timeofday();
