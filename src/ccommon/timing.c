/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#include "timing.h"

// -----------------------------------------------------------------------------
#if defined(__unix__)
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <time.h>
#include <errno.h>

double timing_time() {
	struct timespec tp;
	clock_gettime(CLOCK_MONOTONIC, &tp);
	return (double)tp.tv_sec + (double)tp.tv_nsec * 1e-9;
}

void timing_sleep(double dt) {
	struct timespec tp;
	tp.tv_sec = (int)dt;
	tp.tv_nsec = (dt - tp.tv_sec) * 1e9;
	while (clock_nanosleep(CLOCK_MONOTONIC, 0, &tp, &tp) == EINTR) ;
}

double timing_timeofday() {
	//struct timeval tv={0};
	//gettimeofday(&tv, NULL);
	//return tv.tv_sec + tv.tv_usec * 1e-6;
	struct timespec tp;
	clock_gettime(CLOCK_REALTIME, &tp);
	return (double)tp.tv_sec + (double)tp.tv_nsec * 1e-9;
}

// -----------------------------------------------------------------------------
#elif defined(__WIN32__)
#define WIN32_LEAN_AND_MEAN
#define WIN32_EXTRA_LEAN
#include <windows.h>
#include <stdint.h>

static struct {
	double d_freq;
	LARGE_INTEGER li_freq;
} timing_win_data;

void timing_win_init() {
	QueryPerformanceFrequency(&timing_win_data.li_freq);
	timing_win_data.d_freq = timing_win_data.li_freq.QuadPart;
}

double timing_time() {
	if (!timing_win_data.d_freq) timing_win_init();
	LARGE_INTEGER value;
	QueryPerformanceCounter(&value);
	return (double)value.QuadPart / timing_win_data.d_freq;
}

void timing_sleep(double dt) {
	Sleep(dt*1000);
}

double timing_timeofday() {
	int64_t t;
	GetSystemTimeAsFileTime((FILETIME*)&t);
	return (t - 116444736000000000LL) * 1e-7;
}

// -----------------------------------------------------------------------------
#elif defined(SDL_VERSION)
#include <SDL2/SDL.h>

static struct {
	double d_freq;
	Uint64 u64_freq;
} timing_sdl_data;

void timing_sdl_init() {
	timing_sdl_data.u64_freq = SDL_GetPerformanceFrequency();
	timing_sdl_data.d_freq = timing_sdl_data.u64_freq;
}

double timing_time() {
	return (double)SDL_GetPerformanceCounter() / timing_sdl_data.d_freq;
}

void timing_sleep(double dt) {
	SDL_Delay(dt*1000);
}

#include <time.h>
double timing_timeofday() {
	return time(NULL);  //TODO: not portable
}

// -----------------------------------------------------------------------------
#else
#include <time.h>

double timing_time() {
	return (double)time(0);  //TODO: use clock?
}

void timing_sleep(double dt) {
	//TODO: implement with polling?
}

double timing_timeofday() {
	return (double)time(NULL);  //TODO: not portable
}

#endif
