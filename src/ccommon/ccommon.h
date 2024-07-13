/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 *
 * Common C code header
 */
#pragma once

/* General use macros
 */
#define COUNTOF(X)  (sizeof(X)/sizeof(*(X)))

// Stringify
#define STR(X) STR_(X)
#define STR_(X) #X

// Concatenate and evaluate
#define CAT2(_1,_2)			_1 ## _2
#define CAT3(_1,_2,_3)		_1 ## _2 ## _3
#define CAT4(_1,_2,_3,_4)	_1 ## _2 ## _3 ## _4
// Evaluate and concatenate
#define PASTE2(_1,_2)		CAT2(_1,_2)
#define PASTE3(_1,_2,_3)	CAT3(_1,_2,_3)
#define PASTE4(_1,_2,_3,_4)	CAT4(_1,_2,_3,_4)

#define MINSET(VAR,VAL)  ((VAR) > (VAL) ? ((VAR) = (VAL)) : (VAR))
#define MAXSET(VAR,VAL)  ((VAR) < (VAL) ? ((VAR) = (VAL)) : (VAR))

#define IFFALSE(VAR,DEF)	((VAR) ? (VAR) : (DEF))
#define IFNPOS(VAR,DEF)		((VAR) > 0 ? (VAR) : (DEF))

#define IFFALSESET(VAR,DEF)		((VAR) ? (VAR) : ((VAR) = (DEF)))
#define IFNPOSSET(VAR,DEF)		((VAR) > 0 ? (VAR) : ((VAR) = (DEF)))

#define SIGNg(X)		((X) < 0 ? -1 : (X) > 0 ? 1 : 0)
#define ABSg(X)			((X)<0 ? -(X) : (X))
#define MINg(X,Y)		((X)<(Y) ? (X) : (Y))
#define MAXg(X,Y)		((X)>(Y) ? (X) : (Y))
#define MIN3(A,B,C)		MINg(MINg(A,B),C)
#define MAX3(A,B,C)		MAXg(MAXg(A,B),C)
#define LIMITg(V,L,H)	((V) < (L) ? (V) = (L) : (V) > (H) ? (V) = (H) : (V))
#define TRUNCATEg		LIMITg
#define ccCLAMPED(V,L,H)	((V)<(L) ? (L) : (V)>(H) ? (H) : (V))
#define ccCLAMP(V,L,H)		((V)<(L) ? ((V)=(L)) : (V)>(H) ? ((V)=(H)) : (V))

#define SWAPVg(V,A,B)  ((V)=(A), (A)=(B), (B)=(V))
#define SWAPTg(T,A,B)  do { T tmp_=(A); (A)=(B); (B)=tmp_; } while(0)

#define SWAPg(A,B)  do { \
	char tmp_[sizeof(A)]; \
	void *a=&(A), *b=&(B); \
	memcpy(tmp_, a, sizeof(A)); \
	memcpy(a, b, sizeof(A)); \
	memcpy(b, tmp_, sizeof(B)); \
} while(0)
	
#define MEM_ZERO(D)				memset(&(D), 0, sizeof(D))
#define MEM_COPY(D, S)			memcpy(&(D), &(S), sizeof(D))
#define MEM_CMP(D,S)			memcmp(&(D), &(S), sizeof(D))
#define ARRAY_ZERO(D, C)		memset((D), 0, sizeof(*(D))*(C))
#define ARRAY_COPY(D, S, C)		memcpy((D), (S), sizeof(*(D))*(C))
#define ARRAY_CMP(D, S, C)		memcmp((D), (S), sizeof(*(D))*(C))
	
#ifndef M_PI
#define M_PI  3.14159265358979323846
#endif

/* Error handling
 *
 * Example:
 *   result_t f(...) {
 *     result_t R=1;
 *     if (...) RETURN(code);
 *     TRY( f2(...) );
 *   end:
 *     //clean-up
 *     return R;
 *   }
 */
typedef int result_t;

/* Return going through the end label */
#define RETURN(CODE) do { \
	R = (CODE); \
	goto end; \
} while (0)

#define ERROR_LOG(CODE, ...) do { \
	log_error(__VA_ARGS__); \
	RETURN(CODE); \
} while (0)

// needs stdlib.h
#define FATAL_LOG(...) do { \
	log_error(__VA_ARGS__); \
	exit(1); \
} while (0)

/* Propagate errors */
#define TRY(EXPR) do { \
	result_t _R_ = (EXPR); \
	if (_R_ < 0) RETURN(_R_); \
} while (0)

#define TRYR(EXPR) do { \
	result_t _R_ = (EXPR); \
	if (_R_ < 0) return _R_; \
} while (0)

#define TRYB(CODE, EXPR) \
	TRY( (EXPR) ? 1 : (CODE) )

#define TRYRB(CODE, EXPR) \
	TRYR( (EXPR) ? 1 : (CODE) )

#define TRY_LOG(EXPR, ...) do { \
	result_t _R_ = (EXPR); \
	if (_R_ < 0) ERROR_LOG(_R_, __VA_ARGS__); \
} while (0)

/*#ifndef ERROR_E_MEM
#define ERROR_E_MEM -2
#endif

#define TRYMEM(EXPR) \
	TRY( (EXPR) ? 1 : ERROR_E_MEM )

#define TRYRMEM(EXPR) \
	TRYR( (EXPR) ? 1 : ERROR_E_MEM ) */
