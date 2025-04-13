/* Copyright 2025, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define log(...) do { \
	printf(__VA_ARGS__); \
	printf("\n"); \
} while (0)

#define error(...) do { \
	printf("ERROR "); \
	printf(__VA_ARGS__); \
	printf("\n"); \
	printf("TEST FAIL " __FILE__ "\n"); \
	exit(1); \
} while (0)

#ifdef NDEBUG
#define debug(...)
#else
#define debug(...) do { \
	printf("DEBUG "); \
	printf(__VA_ARGS__); \
	printf("\n"); \
} while (0)
#endif

#define assert_int(A, B, ...) do { \
	int a = (A), b = (B); \
	if (a != b) error(__VA_ARGS__); \
} while(0)
