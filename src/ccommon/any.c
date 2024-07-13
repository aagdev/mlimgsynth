/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#include "any.h"
#include <string.h>
#include <assert.h>
#include <inttypes.h>

AnyTypeGroup any_type_group[ANY_T_LAST+1] = {
	ANY_T_GROUP_NULL,
	ANY_T_GROUP_BOOL,
	ANY_T_GROUP_CHAR,
	ANY_T_GROUP_UINT, ANY_T_GROUP_UINT, ANY_T_GROUP_UINT, ANY_T_GROUP_UINT,
	ANY_T_GROUP_INT, ANY_T_GROUP_INT, ANY_T_GROUP_INT, ANY_T_GROUP_INT,
	ANY_T_GROUP_FLOAT, ANY_T_GROUP_FLOAT,

	ANY_T_GROUP_POINTER,
	ANY_T_GROUP_POINTER,
	ANY_T_GROUP_STRING,
	ANY_T_GROUP_POINTER, ANY_T_GROUP_POINTER, ANY_T_GROUP_POINTER, ANY_T_GROUP_POINTER,
	ANY_T_GROUP_POINTER, ANY_T_GROUP_POINTER, ANY_T_GROUP_POINTER, ANY_T_GROUP_POINTER,
	ANY_T_GROUP_POINTER, ANY_T_GROUP_POINTER,

	ANY_T_GROUP_POINTER, ANY_T_GROUP_POINTER
};

size_t any_type_size[ANY_T_LAST+1] = {
	0,
	sizeof(bool),
	sizeof(char),
	sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t), sizeof(uint64_t),
	sizeof(int8_t),  sizeof(int16_t),  sizeof(int32_t),  sizeof(int64_t),
	sizeof(AnyFloat32), sizeof(AnyFloat64),

	sizeof(void*),
	sizeof(bool*),
	sizeof(char*),
	sizeof(uint8_t*), sizeof(uint16_t*), sizeof(uint32_t*), sizeof(uint64_t*),
	sizeof(int8_t*),  sizeof(int16_t*),  sizeof(int32_t*),  sizeof(int64_t*),
	sizeof(AnyFloat32*), sizeof(AnyFloat64*),

	sizeof(Any*), sizeof(AnyPair*)
};

const char * any_type_name[ANY_T_LAST+1] = {
	"null",
	"bool",
	"char",
	"uint8", "uint16", "uint32", "uint64",
	"int8", "int16", "int32", "int64",
	"float32", "float64",

	"void*",
	"bool*",
	"char*",
	"uint8*", "uint16*", "uint32*", "uint64*",
	"int8*", "int16*", "int32*", "int64*",
	"float32*", "float64*",

	"Any*", "AnyPair*"
};

#define ANY_SCALAR_GET(T,D,S) do { \
	switch (T) { \
	case ANY_T_NULL:	(D) = 0;  break; \
	case ANY_T_BOOL:	(D) = *(bool*)(S);  break; \
	case ANY_T_CHAR:	(D) = *(char*)(S);  break; \
	case ANY_T_UINT8:	(D) = *(uint8_t*)(S);  break; \
	case ANY_T_UINT16:	(D) = *(uint16_t*)(S);  break; \
	case ANY_T_UINT32:	(D) = *(uint32_t*)(S);  break; \
	case ANY_T_UINT64:	(D) = *(uint64_t*)(S);  break; \
	case ANY_T_INT8:	(D) = *(int8_t*)(S);  break; \
	case ANY_T_INT16:	(D) = *(int16_t*)(S);  break; \
	case ANY_T_INT32:	(D) = *(int32_t*)(S);  break; \
	case ANY_T_INT64:	(D) = *(int64_t*)(S);  break; \
	case ANY_T_FLOAT32:	(D) = *(AnyFloat32*)(S);  break; \
	case ANY_T_FLOAT64:	(D) = *(AnyFloat64*)(S);  break; \
	default:			(D) = 0; return false; \
	} \
} while (0)

bool anyp_cast(AnyBaseType td, void* pd, AnyBaseType ts, const void* ps)
{
	switch (td) {
	case ANY_T_NULL:	break;
	case ANY_T_BOOL:	ANY_SCALAR_GET(ts, *(bool*)pd, ps); break;
	case ANY_T_CHAR:	ANY_SCALAR_GET(ts, *(char*)pd, ps); break;
	case ANY_T_UINT8:	ANY_SCALAR_GET(ts, *(uint8_t*)pd, ps); break;
	case ANY_T_UINT16:	ANY_SCALAR_GET(ts, *(uint16_t*)pd, ps); break;
	case ANY_T_UINT32:	ANY_SCALAR_GET(ts, *(uint32_t*)pd, ps); break;
	case ANY_T_UINT64:	ANY_SCALAR_GET(ts, *(uint64_t*)pd, ps); break;
	case ANY_T_INT8:	ANY_SCALAR_GET(ts, *(int8_t*)pd, ps); break;
	case ANY_T_INT16:	ANY_SCALAR_GET(ts, *(int16_t*)pd, ps); break;
	case ANY_T_INT32:	ANY_SCALAR_GET(ts, *(int32_t*)pd, ps); break;
	case ANY_T_INT64:	ANY_SCALAR_GET(ts, *(int64_t*)pd, ps); break;
	case ANY_T_FLOAT32:	ANY_SCALAR_GET(ts, *(AnyFloat32*)pd, ps); break;
	case ANY_T_FLOAT64:	ANY_SCALAR_GET(ts, *(AnyFloat64*)pd, ps); break;
	default:
		assert(!anyb_scalar_is(td));
		*(void**)pd = 0;
		return false;
		//if (anyb_pointer_is(ts)) *(void**)pd = *(void**)ps;
		//else { *(void**)pd = 0; return false; }
	}
	return true;
}

void anyp_ncast(unsigned long n,
	AnyBaseType td, void*restrict pd, AnyBaseType ts, const void*restrict ps)
{
	if (td == ts) memcpy(pd, ps, (size_t)n*anyb_size(td));
	else {
		const unsigned sd = anyb_size(td);
		const unsigned ss = anyb_size(ts);
		while (n--) {
			anyp_cast(td, pd, ts, ps);
			pd = ((unsigned char*)pd) + sd;
			ps = ((const unsigned char*)ps) + ss;
		}
	}
}

#define ANYS_OP2(T,L,R,O) do { \
	switch (T) { \
	case ANY_T_NULL:	return ((L) O 0); \
	case ANY_T_BOOL:	return ((L) O (R).b); \
	case ANY_T_CHAR:	return ((L) O (R).c); \
	case ANY_T_UINT8:	return ((L) O (R).u8); \
	case ANY_T_UINT16:	return ((L) O (R).u16); \
	case ANY_T_UINT32:	return ((L) O (R).u32); \
	case ANY_T_UINT64:	return ((L) O (R).u64); \
	case ANY_T_INT8:	return ((L) O (R).i8); \
	case ANY_T_INT16:	return ((L) O (R).i16); \
	case ANY_T_INT32:	return ((L) O (R).i32); \
	case ANY_T_INT64:	return ((L) O (R).i64); \
	case ANY_T_FLOAT32:	return ((L) O (R).f32); \
	case ANY_T_FLOAT64:	return ((L) O (R).f64); \
	default:			break; \
	} \
} while (0)

bool anys_equal(const Any l, const Any r)
{
	switch (l.t) {
	case ANY_T_NULL:	ANYS_OP2(r.t, 0, r.p, ==);
	case ANY_T_BOOL:	ANYS_OP2(r.t, l.p.b, r.p, ==);
	case ANY_T_CHAR:	ANYS_OP2(r.t, l.p.c, r.p, ==);
	case ANY_T_UINT8:	ANYS_OP2(r.t, l.p.u8, r.p, ==);
	case ANY_T_UINT16:	ANYS_OP2(r.t, l.p.u16, r.p, ==);
	case ANY_T_UINT32:	ANYS_OP2(r.t, l.p.u32, r.p, ==);
	case ANY_T_UINT64:	ANYS_OP2(r.t, l.p.u64, r.p, ==);
	case ANY_T_INT8:	ANYS_OP2(r.t, l.p.i8, r.p, ==);
	case ANY_T_INT16:	ANYS_OP2(r.t, l.p.i16, r.p, ==);
	case ANY_T_INT32:	ANYS_OP2(r.t, l.p.i32, r.p, ==);
	case ANY_T_INT64:	ANYS_OP2(r.t, l.p.i64, r.p, ==);
	case ANY_T_FLOAT32:	ANYS_OP2(r.t, l.p.f32, r.p, ==);
	case ANY_T_FLOAT64:	ANYS_OP2(r.t, l.p.f64, r.p, ==);
	default:			break;
	}
	assert(ANY_T_FLOAT64 < l.t && l.t <= ANY_T_LAST);
	return (l.t == r.t && l.p.p == r.p.p);
}

bool any_equal(const Any* l, const Any* r)
{
	switch (l->t) {
	case ANY_T_NULL:	ANYS_OP2(r->t, 0, r->p, ==);
	case ANY_T_BOOL:	ANYS_OP2(r->t, l->p.b, r->p, ==);
	case ANY_T_CHAR:	ANYS_OP2(r->t, l->p.c, r->p, ==);
	case ANY_T_UINT8:	ANYS_OP2(r->t, l->p.u8, r->p, ==);
	case ANY_T_UINT16:	ANYS_OP2(r->t, l->p.u16, r->p, ==);
	case ANY_T_UINT32:	ANYS_OP2(r->t, l->p.u32, r->p, ==);
	case ANY_T_UINT64:	ANYS_OP2(r->t, l->p.u64, r->p, ==);
	case ANY_T_INT8:	ANYS_OP2(r->t, l->p.i8, r->p, ==);
	case ANY_T_INT16:	ANYS_OP2(r->t, l->p.i16, r->p, ==);
	case ANY_T_INT32:	ANYS_OP2(r->t, l->p.i32, r->p, ==);
	case ANY_T_INT64:	ANYS_OP2(r->t, l->p.i64, r->p, ==);
	case ANY_T_FLOAT32:	ANYS_OP2(r->t, l->p.f32, r->p, ==);
	case ANY_T_FLOAT64:	ANYS_OP2(r->t, l->p.f64, r->p, ==);
	default:			break;
	}
	assert(ANY_T_FLOAT64 < l->t && l->t <= ANY_T_LAST);
	if (!(l->t == r->t && l->len == r->len)) return false;
	if (l->p.p == NULL) return r->p.p == NULL;
	return !memcmp(l->p.p, r->p.p, any_size(l));
}

//TODO: use local functions
#if __STDC_HOSTED__
#include <stdio.h>
long anys_tostr(const AnyScalar *restrict a, size_t n, char *restrict buf)
{
	switch (a->t) {
	case ANY_T_NULL:	return snprintf(buf, n, "null");
	case ANY_T_BOOL:	return snprintf(buf, n, a->p.b ? "true" : "false");
	case ANY_T_CHAR:	return snprintf(buf, n, "%c", a->p.c);
	case ANY_T_UINT8:	return snprintf(buf, n, "%" PRIu8, a->p.u8);
	case ANY_T_UINT16:	return snprintf(buf, n, "%" PRIu16, a->p.u16);
	case ANY_T_UINT32:	return snprintf(buf, n, "%" PRIu32, a->p.u32);
	case ANY_T_UINT64:	return snprintf(buf, n, "%" PRIu64, a->p.u64);
	case ANY_T_INT8:	return snprintf(buf, n, "%" PRId8, a->p.i8);
	case ANY_T_INT16:	return snprintf(buf, n, "%" PRId16, a->p.i16);
	case ANY_T_INT32:	return snprintf(buf, n, "%" PRId32, a->p.i32);
	case ANY_T_INT64:	return snprintf(buf, n, "%" PRId64, a->p.i64);
	case ANY_T_FLOAT32:	return snprintf(buf, n, "%g", a->p.f32);
	case ANY_T_FLOAT64:	return snprintf(buf, n, "%g", a->p.f64);
	default:
		if (a->p.p)		return snprintf(buf, n, "(%p)", a->p.p);
		else			return snprintf(buf, n, "(nil)");
	}
}
#endif

long any_tostr(const AnyScalar*restrict a, size_t n, char*restrict buf)
{
	if (n < 1) return 0;
	char *cur=buf, *end=cur+n-1;
	switch (a->t) {
	case ANY_T_STRING:
		if (a->p.cp) {
			if (cur < end) *cur++ = '"';
			char *c=a->p.cp, *e=c+a->len;
			for (; c<e && cur<end; ++c) {
				if (32 <= *c && *c <= 126) *cur++ = *c;
				else *cur++ = '.';  //TODO: \xNN
			}
			if (cur < end) *cur++ = '"';
			*cur++ = 0;
			return cur - buf;
		}
		break;
	default:
		break;
	}
	return anys_tostr(a, n, buf);
}

/* Dynamic */
#ifndef ANY_ALLOCATORS_MAX
#define ANY_ALLOCATORS_MAX 64
#endif
Allocator * g_allocators[ANY_ALLOCATORS_MAX];

//TODO: default allocator?

Allocator* any_allocator_reg_get(unsigned at) {
	if (at < ANY_ALLOCATORS_MAX) return g_allocators[at];
	return NULL;
}

unsigned any_allocator_register(Allocator* al, unsigned at)
{
	if (!at) {
		at++;
		while (at < ANY_ALLOCATORS_MAX && g_allocators[at]) at++;
	}
	if (at >= ANY_ALLOCATORS_MAX) return 0;
	if (g_allocators[at]) return 0;
	g_allocators[at] = al;
	return at;
}

static inline
void any_map_elements_free(AnyPair* app, uint32_t i, uint32_t iE) {
	for (; i<iE; ++i) {
		any_free(&app[i].k);
		any_free(&app[i].v);
	}
}

static inline
void any_array_elements_free(Any* ap, uint32_t i, uint32_t iE) {
	for (; i<iE; ++i) {
		any_free(&ap[i]);
	}
}

void any_free(Any* a)
{
	Allocator * al = g_allocators[a->cls];
	if (a->t == ANY_T_MAP) {
		any_map_elements_free(a->p.app, 0, a->len);
		alloc_free(al, a->p.p);
	}
	else if (a->t == ANY_T_ARRAY) {
		any_array_elements_free(a->p.ap, 0, a->len);
		alloc_free(al, a->p.p);
	}
	else if (anyb_pointer_is(a->t)) {
		alloc_free(al, a->p.p);
	}
}

bool any_realloc(Any* a, uint32_t len)
{
	Allocator * al = g_allocators[a->cls];
	if (a->t == ANY_T_MAP) {
		if (len < a->len)
			any_map_elements_free(a->p.app, len, a->len);
		else if (len > a->len) {
			AnyPair * p = alloc_realloc(al, a->p.app, sizeof(AnyPair) * len);
			if (!p) return false;
			a->p.app = p;
			// zero init
			memset(a->p.app + a->len, 0, sizeof(AnyPair) * (len - a->len));
		}
	}
	else if (a->t == ANY_T_ARRAY) {
		if (len < a->len)
			any_array_elements_free(a->p.ap, len, a->len);
		else if (len > a->len) {
			Any * p = alloc_realloc(al, a->p.ap, sizeof(Any) * len);
			if (!p) return false;
			a->p.ap = p;
			// zero init
			memset(a->p.ap + a->len, 0, sizeof(Any) * (len - a->len));
		}
	}
	else {
		AnyBaseType t = anyb_pointer_deref(a->t);
		if (!t) return false;
		if (len > a->len) {
			void * p = alloc_realloc(al, a->p.p, anyb_size(t) * len);
			if (!p) return false;
			a->p.p = p;
		}
	}
	a->len = len;
	return true;
}

#define TRYB(...) do { \
	if (!(__VA_ARGS__)) return false; \
} while (0)

bool any_obj_copy(Any* dst, const Any* src)
{
	if (anyb_scalar_is(src->t)) {
		any_free(dst);
		*dst = *src;
	}
	else {
		if (dst->t != src->t || dst->cls == 0) {
			any_free(dst);
			*dst = (Any){ .t=src->t, .cls=src->cls };
		}
		TRYB( any_realloc(dst, src->len) );
		if (src->t == ANY_T_MAP) {
			for (unsigned i=0; i<src->len; ++i) {
				TRYB( any_obj_copy(&dst->p.app[i].k, &src->p.app[i].k) );
				TRYB( any_obj_copy(&dst->p.app[i].v, &src->p.app[i].v) );
			}
		}
		else if (src->t == ANY_T_ARRAY) {
			for (unsigned i=0; i<src->len; ++i)
				TRYB( any_obj_copy(&dst->p.ap[i], &src->p.ap[i]) );
		}
		else {
			AnyBaseType t = anyb_pointer_deref(src->t);
			if (!t) return false;
			memcpy(dst->p.p, src->p.p, anyb_size(t) * src->len);
		}
	}
	return true;
}
