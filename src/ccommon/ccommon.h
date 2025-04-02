/* Copyright 2024-2025, Alejandro A. García <aag@zorzal.net>
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

#define ccSIGN(X)			((X) < 0 ? -1 : (X) > 0 ? 1 : 0)
#define ccABS(X)			((X)<0 ? -(X) : (X))
#define ccMIN(X,Y)			((X)<(Y) ? (X) : (Y))
#define ccMAX(X,Y)			((X)>(Y) ? (X) : (Y))
#define ccMIN3(A,B,C)		ccMIN(ccMIN(A,B),C)
#define ccMAX3(A,B,C)		ccMAX(ccMAX(A,B),C)
#define ccCLAMPED(V,L,H)	((V)<(L) ? (L) : (V)>(H) ? (H) : (V))
#define ccCLAMP(V,L,H)		((V)<(L) ? ((V)=(L)) : (V)>(H) ? ((V)=(H)) : (V))

#define ccSWAPV(V,A,B)		((V)=(A), (A)=(B), (B)=(V))
#define ccSWAPT(T,A,B)		do { T tmp_=(A); (A)=(B); (B)=tmp_; } while(0)

#define ccSWAP(A,B)  do { \
	char tmp_[sizeof(A)]; \
	void *a=&(A), *b=&(B); \
	memcpy(tmp_, a, sizeof(A)); \
	memcpy(a, b, sizeof(A)); \
	memcpy(b, tmp_, sizeof(B)); \
} while(0)

#define ccFLAG_SET(VAR, FLAG, CTRL) \
	((VAR) = (CTRL) ? (VAR) | (FLAG) : (VAR) & ~(FLAG))
	
#define MEM_ZERO(D)				memset(&(D), 0, sizeof(D))
#define MEM_CMP(D,S)			memcmp(&(D), &(S), sizeof(D))
#define MEM_COPY(D, S)			memcpy(&(D), &(S), sizeof(D))
#define ARRAY_ZERO(D, C)		memset((D), 0, sizeof(*(D))*(C))
#define ARRAY_CMP(D, S, C)		memcmp((D), (S), sizeof(*(D))*(C))
#define ARRAY_COPY(D, S, C)		memcpy((D), (S), sizeof(*(D))*(C))
#define ARRAY_MOVE(D, S, C)		memmove((D), (S), sizeof(*(D))*(C))

#define ccUNUSED(x) (void)(x)
	
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

#define TRYB(CODE, EXPR) do { \
	if (!(EXPR)) RETURN(CODE); \
} while (0)

#define TRYRB(CODE, EXPR) do { \
	if (!(EXPR)) return (CODE); \
} while (0)

#define TRY_LOG(EXPR, ...) do { \
	result_t _R_ = (EXPR); \
	if (_R_ < 0) ERROR_LOG(_R_, __VA_ARGS__); \
} while (0)

#define TRY_ASSERT(EXPR) do { \
	result_t _R_ = (EXPR); \
	if (_R_ < 0) ERROR_LOG(_R_, "Error 0x%x in %s:%d:\n%s", \
		-_R_, __FILE__, __LINE__, #EXPR); \
} while (0)

#define TRYB_ASSERT(CODE, EXPR) do { \
	if (!(EXPR)) ERROR_LOG((CODE), "Assertion Error %s:%d:\n%s", \
		__FILE__, __LINE__, #EXPR); \
} while (0)
