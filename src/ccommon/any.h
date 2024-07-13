/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 *
 * Any type: runtime dynamic typing.
 */
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include "alloc.h"

/*
	Any base/scalar type (no data)
*/
typedef enum AnyBaseType {
	ANY_T_NULL,
	ANY_T_BOOL,
	ANY_T_CHAR,
	ANY_T_UINT8, ANY_T_UINT16, ANY_T_UINT32, ANY_T_UINT64,
	ANY_T_INT8,  ANY_T_INT16,  ANY_T_INT32,  ANY_T_INT64,
	ANY_T_FLOAT32, ANY_T_FLOAT64,
	// Pointer / array
	ANY_T_VOIDP,
	ANY_T_BOOLP,
	ANY_T_CHARP,
	ANY_T_UINT8P, ANY_T_UINT16P, ANY_T_UINT32P, ANY_T_UINT64P,
	ANY_T_INT8P,  ANY_T_INT16P,  ANY_T_INT32P,  ANY_T_INT64P,
	ANY_T_FLOAT32P, ANY_T_FLOAT64P,

	ANY_T_ARRAY,		//Any*
	ANY_T_MAP,		//AnyPair*

	ANY_T_STRING = ANY_T_CHARP,
	ANY_T_SCALAR_LAST = ANY_T_FLOAT64,
	ANY_T_POINTER_LAST = ANY_T_FLOAT64P,
	ANY_T_OBJECT_LAST = ANY_T_MAP,
	ANY_T_LAST = ANY_T_MAP
} AnyBaseType;

typedef enum AnyTypeGroup {
	ANY_T_GROUP_NULL,
	ANY_T_GROUP_BOOL,
	ANY_T_GROUP_CHAR,
	ANY_T_GROUP_UINT,
	ANY_T_GROUP_INT,
	ANY_T_GROUP_FLOAT,
	ANY_T_GROUP_POINTER,
	ANY_T_GROUP_STRING,
} AnyTypeGroup;

extern AnyTypeGroup any_type_group[ANY_T_LAST+1];
extern size_t any_type_size[ANY_T_LAST+1];
extern const char * any_type_name[ANY_T_LAST+1];

static inline
AnyTypeGroup anyb_group(AnyBaseType t) {
	return (t <= ANY_T_LAST) ? any_type_group[t] : ANY_T_GROUP_NULL;
}

static inline
size_t anyb_size(AnyBaseType t) {
	return (t <= ANY_T_LAST) ? any_type_size[t] : 0;
}

static inline
const char* anyb_name(AnyBaseType t) {
	return (t <= ANY_T_LAST) ? any_type_name[t] : "";
}

#define ANY_T_UINT_(S)  (ANY_T_UINT8 + ((S)>4 ? 3 : (S)>2 ? 2 : (S)>1 ? 1 : 0))
#define ANY_T_INT_(S)  (ANY_T_INT8 + ((S)>4 ? 3 : (S)>2 ? 2 : (S)>1 ? 1 : 0))
#define ANY_T_FLOAT_(S)  (ANY_T_FLOAT32 + ((S)>4 ? 1 : 0))

static inline
AnyBaseType anyb_from_group_size(AnyTypeGroup g, size_t s) {
	switch (g) {
	case ANY_T_GROUP_BOOL:		return ANY_T_BOOL;
	case ANY_T_GROUP_CHAR:		return ANY_T_CHAR;
	case ANY_T_GROUP_UINT:		return ANY_T_UINT_(s);
	case ANY_T_GROUP_INT:		return ANY_T_INT_(s);
	case ANY_T_GROUP_FLOAT:		return ANY_T_FLOAT_(s);
	case ANY_T_GROUP_POINTER:	return ANY_T_VOIDP;
	case ANY_T_GROUP_STRING:	return ANY_T_STRING;
	default:					return ANY_T_NULL;
	}
}

static inline
bool anyb_scalar_is(AnyBaseType t) {
	return t <= ANY_T_SCALAR_LAST;
}

static inline
bool anyb_pointer_is(AnyBaseType t) {
	return ANY_T_SCALAR_LAST < t && t <= ANY_T_POINTER_LAST;
}

static inline
AnyBaseType anyb_pointer_get(AnyBaseType t) {
	return anyb_scalar_is(t) ? t + ANY_T_SCALAR_LAST + 1 : ANY_T_VOIDP;
}

static inline
AnyBaseType anyb_pointer_deref(AnyBaseType t) {
	return anyb_pointer_is(t) ? t - ANY_T_SCALAR_LAST - 1 : ANY_T_NULL;
}

//TODO: check scalar or pointer classification of Array and Map

static inline
bool anyb_cast_check(AnyBaseType td, AnyBaseType ts) {
	return !(anyb_pointer_is(td) ^ anyb_pointer_is(ts));
}

/*
	Any type (no data)
*/
#define ANYT_LENGTH_INDEF 0xffffffff

typedef struct AnyFullType {
	uint16_t t;		//AnyBaseType
	uint16_t cls;	//registered class/allocator index, 0 is none
	uint32_t len;	//Array length
} AnyFullType;

/*
	Any payload (data only)
*/
//typedef __fp16 AnyFloat16;
typedef float AnyFloat32;
typedef double AnyFloat64;

typedef struct Any Any;
typedef struct AnyPair AnyPair;

typedef union AnyPayload {
	bool b;
	char c;
	uint8_t u8;  uint16_t u16;  uint32_t u32;  uint64_t u64;
	int8_t i8;   int16_t i16;   int32_t i32;   int64_t i64;
	AnyFloat32 f32;  AnyFloat64 f64;

	void* p;
	bool* bp;
	char* cp;
	uint8_t* u8p;  uint16_t* u16p;  uint32_t* u32p;  uint64_t* u64p;
	int8_t* i8p;   int16_t* i16p;   int32_t* i32p;   int64_t* i64p;
	AnyFloat32* f32p;  AnyFloat64* f64p;

	Any* ap;  //array
	AnyPair* app;  //map
} AnyPayload;

bool anyp_cast(AnyBaseType td, void* pd, AnyBaseType ts, const void* ps);

void anyp_ncast(unsigned long n,
	AnyBaseType td, void*restrict pd, AnyBaseType ts, const void*restrict ps);

/*
	Any scalar (type and data, no arrays)
*/
struct Any {
	uint16_t t;		//AnyBaseType
	uint16_t cls;	//registered class/allocator index, 0 is none
	uint32_t len;	//Array length
	AnyPayload p;
};

struct AnyPair { Any k, v; };

typedef Any AnyScalar;

static inline
bool anys_cast(AnyScalar* a, AnyBaseType t) {
	if (a->t == t) return true;
	bool r = anyp_cast(t, &a->p, a->t, &a->p);
	a->t = t;
	return r;
}

bool anys_equal(const Any l, const Any r);

long anys_tostr(const AnyScalar*restrict a, size_t n, char*restrict buffer);

/**
@def any_TYPE(V)
@return Any with type TYPE and value V.
*/
#define any_null()		((AnyScalar){ .t=ANY_T_NULL, })
#define any_bool(V)		((AnyScalar){ .t=ANY_T_BOOL, .p={ .b=(V) } })
#define any_char(V)		((AnyScalar){ .t=ANY_T_CHAR, .p={ .c=(V) } })
#define any_uint8(V)	((AnyScalar){ .t=ANY_T_UINT8, .p={ .u8=(V) } })
#define any_uint16(V)	((AnyScalar){ .t=ANY_T_UINT16, .p={ .u16=(V) } })
#define any_uint32(V)	((AnyScalar){ .t=ANY_T_UINT32, .p={ .u32=(V) } })
#define any_uint64(V)	((AnyScalar){ .t=ANY_T_UINT64, .p={ .u64=(V) } })
#define any_int8(V)		((AnyScalar){ .t=ANY_T_INT8, .p={ .i8=(V) } })
#define any_int16(V)	((AnyScalar){ .t=ANY_T_INT16, .p={ .i16=(V) } })
#define any_int32(V)	((AnyScalar){ .t=ANY_T_INT32, .p={ .i32=(V) } })
#define any_int64(V)	((AnyScalar){ .t=ANY_T_INT64, .p={ .i64=(V) } })
#define any_float32(V)	((AnyScalar){ .t=ANY_T_FLOAT32, .p={ .f32=(V) } })
#define any_float64(V)	((AnyScalar){ .t=ANY_T_FLOAT64, .p={ .f64=(V) } })

/**
@fn CTYPE anys_TYPE_get(AnyScalar* a)
Returns a C scalar value of type TYPE from a.
*/

/**
@fn CTYPE any_TYPE_cast_get(AnyScalar* a)
Convert a to TYPE and returns its value.
*/

#define DEF_ANYS_FUNCTIONS(R,N,T,V) \
	static inline R anys_##N##_get(const AnyScalar* a) { \
		if (a->t == ANY_T_##T) return a->p.V; \
		R v; \
		anyp_cast(ANY_T_##T, &v, a->t, &a->p); \
		return v; \
	} \
	static inline R anys_##N##_cast_get(AnyScalar* a) { \
		anys_cast(a, ANY_T_##T); \
		return a->p.V; \
	}

DEF_ANYS_FUNCTIONS(bool, bool, BOOL, b)
DEF_ANYS_FUNCTIONS(char, char, CHAR, c)
DEF_ANYS_FUNCTIONS(uint8_t, uint8, UINT8, u8)
DEF_ANYS_FUNCTIONS(uint16_t, uint16, UINT16, u16)
DEF_ANYS_FUNCTIONS(uint32_t, uint32, UINT32, u32)
DEF_ANYS_FUNCTIONS(uint64_t, uint64, UINT64, u64)
DEF_ANYS_FUNCTIONS(int8_t, int8, INT8, i8)
DEF_ANYS_FUNCTIONS(int16_t, int16, INT16, i16)
DEF_ANYS_FUNCTIONS(int32_t, int32, INT32, i32)
DEF_ANYS_FUNCTIONS(int64_t, int64, INT64, i64)
DEF_ANYS_FUNCTIONS(AnyFloat32, float32, FLOAT32, f32)
DEF_ANYS_FUNCTIONS(AnyFloat64, float64, FLOAT64, f64)

/*
	Any arrays (type, data, handles arrays, no memory allocations)
*/

// These must be macros so that the address can be taken.

#define any_voidp(N,P) \
	((Any){ .t=ANY_T_VOIDP, .len=(N), .p={ .cp=(char*)(P) } })

#define any_charp(N,P) \
	((Any){ .t=ANY_T_CHARP, .len=(N), .p={ .cp=(char*)(P) } })

#define any_string(N,P) \
	((Any){ .t=ANY_T_STRING, .len=(N), .p={ .cp=(char*)(P) } })

#define any_stringz(S) \
	((Any){ .t=ANY_T_STRING, .len=strlen(S), .p={ .cp=(char*)(S) } })

// vector.h
#define any_stringd(S) \
	((Any){ .t=ANY_T_STRING, .len=dstr_count(S), .p={ .cp=(char*)(S) } })

// strslice.h
#define any_strings(S) \
	((Any){ .t=ANY_T_STRING, .len=(S).s, .p={ .cp=(char*)(S).b } })

#define any_vector(T,N,P) \
	((Any){ .t=anyb_pointer_get(T), .len=(N), .p={ .p=(void*)(P) } })

#define any_vector_indef(T) \
	((Any){ .t=anyb_pointer_get(T), .len=ANYT_LENGTH_INDEF })

#define any_vector_vec(T,V) \
	((Any){ .t=anyb_pointer_get(T), .len=vec_count(V), .p={ .p=(void*)(V) } })

#define any_vector_dyn(T,C) \
	((Any){ .t=anyb_pointer_get(T), .cls=(C) })

#define any_array(N,P) \
	((Any){ .t=ANY_T_ARRAY, .len=(N), .p={ .ap=(Any*)(P) } })

#define any_array_indef() \
	((Any){ .t=ANY_T_ARRAY, .len=ANYT_LENGTH_INDEF })

#define any_array_dyn(C) \
	((Any){ .t=ANY_T_ARRAY, .cls=(C) })

#define any_map(N,P) \
	((Any){ .t=ANY_T_MAP, .len=(N), .p={ .app=(AnyPair*)(P) } })

#define any_map_indef() \
	((Any){ .t=ANY_T_MAP, .len=ANYT_LENGTH_INDEF })

#define any_map_dyn(C) \
	((Any){ .t=ANY_T_MAP, .cls=(C) })

static inline
size_t any_size(const Any* a) {
	if (anyb_scalar_is(a->t)) return anyb_size(a->t);
	if (a->len == ANYT_LENGTH_INDEF) return 0;
	return anyb_size(anyb_pointer_deref(a->t)) * a->len;
}

static inline
bool any_identical_is(const Any* l, const Any* r) {
	if (l->t != r->t) return false;
	if (l->len != r->len) return false;
	if (anyb_scalar_is(l->t)) {
		return !memcmp(&l->p, &r->p, anyb_size(l->t));
	} else if (!l->len) {
		return true;
	} else if (l->p.p) {
		return !memcmp(l->p.p, r->p.p, any_size(l));
	} else {
		return !(r->p.p);
	}
}

bool any_equal(const Any* l, const Any* r);  //TODO: pointers?

static inline
Any any_pointer_get(const Any* a) {
	return (Any){ .t=anyb_pointer_get(a->t), .len=1, .p={ .p=(void*)&a->p } };
}

static inline
bool any_pointer_index(Any* dst, const Any* a, unsigned long i) {
	AnyBaseType t = anyb_pointer_deref(a->t);
	if (!t) return false;
	const size_t sz = anyb_size(t);
	*dst = (Any){ .t=t };
	memcpy(&dst->p, a->p.u8p+sz*i, sz);
	return true;
}

//TODO: pointer_set

long any_tostr(const AnyScalar *restrict a, size_t n, char *restrict buffer);

/*
 * Dynamic allocation
 *
 * Allows changing the length of vector, array or map types,
 * allocating and freeing memory as needed.
 * You need to globally register the allocators first.
 * Suggestion: use an allocator with ALLOC_F_DOPTIMAL.
 */
 
// Globally register an allocator.
// Returns an id to use in Any.cls.
// If <at> != 0, then tries to use this id.
unsigned any_allocator_register(Allocator* al, unsigned at);

// Returns a registered allocator
Allocator* any_allocator_reg_get(unsigned at);

static inline
bool any_allocator_set(Any* a, unsigned cls) {
	if (a->cls) return false;
	a->cls = cls;
}

void any_free(Any* a);

bool any_realloc(Any* a, uint32_t len);

bool any_obj_copy(Any* dst, const Any* src);

//TODO: make all operations cls aware or use a prefix anyd_

/*
	Any object (type, data, handles arrays and memory allocations)
*/
//TODO

