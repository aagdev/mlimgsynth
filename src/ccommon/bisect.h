/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#pragma once
#include <stddef.h>
#include <stdbool.h>

/* Bisection for binary search and sorting
 * Example with integer:
	bool found;
	size_t idx;
	BISECT_RIGHT(found, idx, 0, vec_count(index), index[i_] - key) );
 * Example with string key:
	BISECT_RIGHT(found, idx, 0, vec_count(index), strcmp(index[i_], key) );
 */
#define BISECT_RIGHT(FOUND, IDX, INI, LEN, CMPE) \
	BISECT_RIGHT_S(FOUND, IDX, INI, LEN, r_ = (CMPE); )

#define BISECT_RIGHT_DECL(FOUND, IDX, INI, LEN, CMPE) \
	bool FOUND=0; FOUND=FOUND; \
	size_t IDX=0; IDX=IDX; \
	BISECT_RIGHT_S(FOUND, IDX, INI, LEN, r_ = (CMPE); );

/* Alternative version where CMPM can be function-like macro.
 */
#define BISECT_RIGHT_M(FOUND, IDX, INI, LEN, CMPM) \
	BISECT_RIGHT_S(FOUND, IDX, INI, LEN, CMPM(r_,i_); )

/* Alternative version where CMPS is an statement setting i_.
 */
#define BISECT_RIGHT_S(FOUND, IDX, INI, LEN, CMPS) do { \
	size_t i_, b_=(INI), e_=(LEN); \
	int r_=-1; \
	while (b_ < e_) { \
		i_ = (b_+e_)/2; \
		CMPS \
		if (r_ < 0)			b_ = i_+1; \
		else if (r_ > 0)	e_ = i_; \
		else { b_=i_; break; } \
	} \
	(FOUND) = (r_ == 0); \
	(IDX) = b_; \
} while(0)
