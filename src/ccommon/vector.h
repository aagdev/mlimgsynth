/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 *
 * Dynamic arrays of arbitrary type (vectors).
 *
 * Example:
 * 	VECTOR(int) v=NULL;
 * 	vec_push(v, 8)
 * 	assert(v[0] == 8);
 * 	vec_free(v);
 * 
 * Dynamic string interface.
 * It is a vector with a guaranteed zero-element at the end.
 * 
 * Example:
 * 	DynStr s=NULL;
 * 	dstr_copyz(s, "Hello world!");
 * 	assert(s[0] == 'H');
 * 	assert(s[22] == 0);
 * 	dstr_free(s);
 *
 * Notes:
 *   Requires an allocator that stores the size. By default, uses Allocator_stdlib.
*/
#pragma once
#include "ccommon.h"
#include "alloc.h"
#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

//TODO: use alloc_size for capacity ?
//TODO: vec_push(P, ...) to allow commas in the value ?

#ifndef VECTOR_DEF_ALLOC
#define VECTOR_DEF_ALLOC  g_allocator_dopt
#endif

/* Vector Interface */

#define vec_uint		dynbuf_uint

// Helps document that a type is a vector
//#define VECTOR(T)  T*

// info
#define vec_count(P)		dynbuf_count(P)
#define vec_bytesize(P)		(dynbuf_count(P) * sizeof(*(P)))
#define vec_capacity(P)		dynbuf_capacity((P))

// static allocation
#define vec_frombuffer(T,S,B) \
	((T*)dynbuf_static(((S), (B), sizeof(T), 0))

#define vec_fromarray(A) \
	dynbuf_static(sizeof(A), (A), sizeof(*A), 0)

#define vec_stack(T,C) \
	((T*)dynbuf_static((C)*sizeof(T)+sizeof(DynBuf), \
					   (uint8_t[(C)*sizeof(T)+sizeof(DynBuf)]){0}, \
					   sizeof(T), 0))


// modify with default allocator
#define vec_free(P)				veca_free((P),VECTOR_DEF_ALLOC)
#define vec_realloc(P,C)		veca_realloc((P),(C),VECTOR_DEF_ALLOC)
#define vec_resize(P,C)			veca_resize((P),(C),VECTOR_DEF_ALLOC)
#define vec_resize_zero(P,C)	veca_resize_zero((P),(C),VECTOR_DEF_ALLOC)
#define vec_copy(P,C,D)			veca_copy((P),(C),(D),VECTOR_DEF_ALLOC)
#define vec_copyv(P,V)			veca_copyv((P),(V),VECTOR_DEF_ALLOC)
#define vec_insert(P,I,C,D)		veca_insert((P),(I),(C),(D),VECTOR_DEF_ALLOC)
#define vec_insertv(P,I,V)		veca_insertv((P),(I),(V),VECTOR_DEF_ALLOC)
#define vec_append(P,C,D)		veca_append((P),(C),(D),VECTOR_DEF_ALLOC)
#define vec_append_zero(P,C)	veca_append_zero((P),(C),VECTOR_DEF_ALLOC)
#define vec_appendv(P,V)		veca_appendv((P),(V),VECTOR_DEF_ALLOC)
#define vec_push(P,V)			veca_push((P),(V),VECTOR_DEF_ALLOC)

// modify with custom allocator
#define veca_free(P,A) \
	dynbuf_free((void**)&(P), (A))

#define veca_realloc(P,C,A) \
	dynbuf_realloc((void**)&(P), (C), sizeof(*(P)), 0, (A))

#define veca_resize(P,C,A) \
	dynbuf_resize((void**)&(P), (C), sizeof(*(P)), 0, (A))

#define veca_resize_zero(P,C,A) \
	dynbuf_resize((void**)&(P), (C), sizeof(*(P)), 4, (A))

#define veca_copy(P,C,D,A) \
	dynbuf_copy((void**)&(P), (C), (D), sizeof(*(P)), 0, (A))

#define veca_copyv(P,V,A) \
	veca_copy((P), vec_count(V), (V), (A))

#define veca_insert(P,I,C,D,A) \
	dynbuf_insert((void**)&(P), (I), (C), (D), sizeof(*(P)), 0, (A))

#define veca_insertv(P,I,V,A) \
	vec_insert((P), (I), vec_count(V), (V), (A))

#define veca_append(P,C,D,A) \
	dynbuf_append((void**)&(P), (C), (D), sizeof(*(P)), 0, (A))

#define veca_append_zero(P,C,A) \
	dynbuf_append((void**)&(P), (C), NULL, sizeof(*(P)), 2, (A))

#define veca_appendv(P,V,A) \
	veca_append((P), vec_count(V), (V), (A))

#define veca_push(P,V,A) \
	(veca_append((P),1,NULL,(A)) ? ((P)[vec_count(P)-1] = (V), (P)) : NULL)

// modify (no allocator needed)
#define vec_remove(P,I,C) \
	dynbuf_remove((void**)&(P), (I), (C), sizeof(*(P)), 0)

#define vec_reduce(P,C) \
	dynbuf_reduce((void**)&(P), (C), sizeof(*(P)), 0)

#define vec_pop(P) \
	(vec_reduce((P), 1), (P)[vec_count(P)])

#define vec_popd(P,D) \
	(vec_count(P) ? vec_pop(P) : (D))

#define vec_zero(P) \
	memset((P), 0, vec_count(P)*sizeof(*P))

// access
#define vec_getd(P,I,D) \
	(vec_count(P)>(I) ? (P)[I] : (D))

#define vec_getpd(P,I,D) \
	(vec_count(P)>(I) ? &(P)[I] : (D))

#define vec_last(P,I) \
	((P)[vec_count(P)-1-(I)])

#define vec_lastd(P,I,D) \
	(vec_count(P)>(I) ? (P)[vec_count(P)-1-(I)] : (D))

#define vec_lastpd(P,I,D) \
	(vec_count(P)>(I) ? &(P)[vec_count(P)-1-(I)] : (D))

#define vec_lastp(P,I) \
	vec_lastpd((P), (I), NULL)

#define vec_end(P) \
	((P) + vec_count(P))

#define vec_for(P,V,I) \
	for (dynbuf_uint V=(I), V##e_=vec_count(P); V<V##e_; ++V)

#define vec_forr(P,V) \
	for (dynbuf_uint V##e_=vec_count(P), V=V##e_-1; V<V##e_; --V)

#define vec_forrn(P,V,N) \
	for (dynbuf_uint V##e_=vec_count(P), V##s_=e-N, V=V##e_-1; \
		V<V##e_ && V>=V##s_; --V)

#define vec_forp(T,P,V,I) \
	for (T *V=(P)+(I), *V##e_=V+vec_count(P); V<V##e_; ++V)

#define vec_forrp(T,P,V) \
	for (T *V=(P)+vec_count(P)-1; V>=(P); --V)

/* String Interface */

// Use this type to make clear that a variable is a dynamic string.
typedef char* DynStr;

// info
#define dstr_capacity(P)	dynbuf_capacity(P)
#define dstr_count(P)		dynbuf_count(P)
#define dstr_empty(P)		(!((P) && (P)[0]))

// static allocation
#define dstr_frombuffer(S,B) \
	((char*)dynbuf_static((S), (B), sizeof(char), 1))

#define dstr_stack(C) \
	((char*)dynbuf_static(((C)+1)*sizeof(char)+sizeof(DynBuf), \
					   (uint8_t[((C)+1)*sizeof(char)+sizeof(DynBuf)]){0}, \
					   sizeof(char), 1))

// modify with default allocator
#define dstr_free(P) \
	dynbuf_free((void**)&(P), VECTOR_DEF_ALLOC)

#define dstr_realloc(P,C) \
	dynbuf_realloc((void**)&(P), (C), sizeof(*(P)), 1, VECTOR_DEF_ALLOC)

#define dstr_resize(P,C) \
	dynbuf_resize((void**)&(P), (C), sizeof(*(P)), 1, VECTOR_DEF_ALLOC)

#define dstr_copy(P,C,D) \
	dynbuf_copy((void**)&(P), (C), (D), sizeof(*(P)), 1, VECTOR_DEF_ALLOC)

#define dstr_copyd(P,V) \
	dstr_copy((P), dstr_count(V), (V))

#define dstr_copyz(P,D) \
	dynstr_copyz(&(P), (D), VECTOR_DEF_ALLOC)

#define dstr_insert(P,I,C,D) \
	dynbuf_insert((void**)&(P), (I), (C), (D), sizeof(*(P)), 1, VECTOR_DEF_ALLOC)

#define dstr_insertd(P,I,V) \
	dstr_insert((P), (I), dstr_count(V), (V))

#define dstr_insertz(P,I,D) \
	dynstr_insertz(&(P), (I), (D), VECTOR_DEF_ALLOC)

#define dstr_append(P,C,D) \
	dynbuf_append((void**)&(P), (C), (D), sizeof(*(P)), 1, VECTOR_DEF_ALLOC)

#define dstr_appendd(P,V) \
	dstr_append((P), dstr_count(V), (V))

#define dstr_appendz(P,D) \
	dynstr_appendz(&(P), (D), VECTOR_DEF_ALLOC)

#define dstr_push(P,V) \
	(dstr_append((P),1,NULL) ? ((P)[dstr_count(P)-1] = (V), (P)) : NULL)

#define dstr_printf(P,...) \
	dynstr_printf(&(P), 0, VECTOR_DEF_ALLOC, __VA_ARGS__)

#define dstr_printfa(P,...) \
	dynstr_printf(&(P), 1, VECTOR_DEF_ALLOC, __VA_ARGS__)

#define dstr_vprintf(P,F,A) \
	dynstr_vprintf(&(P), 0, VECTOR_DEF_ALLOC, (F), (A))

#define dstr_vprintfa(P,F,A) \
	dynstr_vprintf(&(P), 1, VECTOR_DEF_ALLOC, (F), (A))

// modify (no allocator needed)
#define dstr_remove(P,I,C) \
	dynbuf_remove((void**)&(P), (I), (C), sizeof(*(P)), 1)

#define dstr_reduce(P,C) \
	dynbuf_reduce((void**)&(P), (C), sizeof(*(P)), 1)
	
// access
#define dstr_end(P) \
	((P) + dstr_count(P))

#define dstr_for(P,V,I) \
	for (dynbuf_uint V=(I), V##e_=dstr_count(P); V<V##e_; ++V)

#define dstr_forr(P,V) \
	for (dynbuf_uint V##e_=dstr_count(P), V=V##e_-1; V<V##e_; --V)

// utilize
#define dstr_ifempty_copyz(P,D) \
	(dstr_empty(P) ? dstr_copyz((P),(D)) : 0)

#define dstr_ifempty_printf(P,...) \
	(dstr_empty(P) ? dstr_printf((P), __VA_ARGS__) : 0)

/*
	Inline implementation
*/
#define DYNBUF_FLAG_ZERO_END 1
#define DYNBUF_FLAG_ZERO_NEW 2
#define DYNBUF_FLAG_ZERO_ALL 4

typedef uint32_t dynbuf_uint;

#define DYNBUF_CAP_MASK		0x7FFFFFFF
#define DYNBUF_CAP_F_STATIC	0x80000000

typedef struct DynBuf {
	dynbuf_uint	count,
				capacity;
	uint8_t		d[];
} DynBuf;

#define dynbuf__fatal()  abort()

static inline
DynBuf* dynbuf_cast(void* p) {
	return p ? ((DynBuf*)p)-1 : NULL;
}
static inline
const DynBuf* dynbuf_ccast(const void* p) {
	return p ? ((const DynBuf*)p)-1 : NULL;
}
static inline
void* dynbuf_decast(DynBuf* p) {
	return p ? (void*)(p+1) : NULL;
}

static inline
dynbuf_uint dynbuf_capacity(const void* p) {
	const DynBuf* obj = dynbuf_ccast(p);
	return obj ? (obj->capacity & DYNBUF_CAP_MASK) : 0;
}
static inline
dynbuf_uint dynbuf_count(const void* p) {
	const DynBuf* obj = dynbuf_ccast(p);
	return obj ? obj->count : 0;
}

static inline
void dynbuf_free(void** p, Allocator* a)
{
	DynBuf* obj = dynbuf_cast(*p);
	if (obj && !(obj->capacity & DYNBUF_CAP_F_STATIC))
		alloc_free(a, obj);
	*p = NULL;
}

#define dynbuf__zero1(D)  memset((D), 0, tsize)

static inline
void* dynbuf_static(dynbuf_uint bsize, void* buf, size_t tsize, unsigned flags)
{
	if (bsize < sizeof(DynBuf)+tsize) return NULL;
	DynBuf* obj = buf;
	obj->count = 0;
	obj->capacity = ((bsize - sizeof(DynBuf)) / tsize) | DYNBUF_CAP_F_STATIC;
	if (flags & DYNBUF_FLAG_ZERO_END) {
		dynbuf__zero1(obj->d);
		obj->capacity--;
	}
	return dynbuf_decast(obj);
}

static inline
DynBuf* dynbuf_realloc(void** pp, dynbuf_uint count, size_t tsize, unsigned flags,
	Allocator* a)
{
	DynBuf* obj = dynbuf_cast(*pp);
	if (flags & DYNBUF_FLAG_ZERO_END) count++;

	if (obj && obj->capacity & DYNBUF_CAP_F_STATIC) {
		if (count <= (obj->capacity & DYNBUF_CAP_MASK)) return obj;
		else dynbuf__fatal();
	}

	size_t nsz = size_round2( sizeof(DynBuf) + count * tsize );
	obj = alloc_realloc(a, obj, nsz);
	if (!obj) return NULL;

	obj->capacity = (nsz - sizeof(DynBuf)) / tsize;
	if (flags & DYNBUF_FLAG_ZERO_END) obj->capacity--;

	if (!*pp) {
		obj->count = 0;
		if (flags & DYNBUF_FLAG_ZERO_END) dynbuf__zero1(obj->d);
	}
	else if (obj->count > obj->capacity) {
		obj->count = obj->capacity;
		if (flags & DYNBUF_FLAG_ZERO_END) dynbuf__zero1(obj->d+obj->count);
	}

	*pp = dynbuf_decast(obj);
	return obj;
}

static inline
DynBuf* dynbuf_resize(void** pp, dynbuf_uint count, size_t tsize, unsigned flags,
	Allocator* a)
{
	DynBuf* obj = dynbuf_cast(*pp);

	if (!obj && !count) return NULL;
	if (!obj || (obj->capacity & DYNBUF_CAP_MASK) < count) {
		obj = dynbuf_realloc(pp, count, tsize, flags, a);
		if (!obj) return NULL;
	}

	if (flags & DYNBUF_FLAG_ZERO_ALL)
		memset(obj->d, 0, tsize*count);
	else if (flags & DYNBUF_FLAG_ZERO_NEW) {
		if (count > obj->count)
			memset(obj->d + tsize*obj->count, 0, tsize*(count - obj->count));
	}

	if (flags & DYNBUF_FLAG_ZERO_END)
		dynbuf__zero1(obj->d + tsize*count);

	obj->count = count;
	return obj;
}


static inline
DynBuf* dynbuf_copy(void** pp, dynbuf_uint count, const void* data,
					size_t tsize, unsigned flags, Allocator* a)
{
	DynBuf* obj = dynbuf_resize(pp, count, tsize, flags, a);
	if (data && count) memcpy(obj->d, data, count*tsize);
	return obj;
}

static inline
DynBuf* dynbuf_insert(void** pp, dynbuf_uint i, dynbuf_uint count,
					  const void* data, size_t tsize, unsigned flags, Allocator* a)
{
	DynBuf* obj = dynbuf_cast(*pp);
	dynbuf_uint c0 = obj ? obj->count : 0;
	if (c0 < i) c0 = i;
	obj = dynbuf_resize(pp, c0+count, tsize, flags, a);

	if (i < c0) memmove(obj->d+(i+count)*tsize, obj->d+i*tsize, (c0-i)*tsize);
	if (data && count) memcpy(obj->d+i*tsize, data, count*tsize);
	return obj;
}

static inline
DynBuf* dynbuf_append(void** pp, dynbuf_uint count, const void* data,
					  size_t tsize, unsigned flags, Allocator* a)
{
	DynBuf* obj = dynbuf_cast(*pp);
	dynbuf_uint c0 = obj ? obj->count : 0;
	obj = dynbuf_resize(pp, c0+count, tsize, flags, a);

	if (data && count) memcpy(obj->d+c0*tsize, data, count*tsize);
	return obj;
}

static inline
DynBuf* dynbuf_reduce(void** pp, dynbuf_uint count,
					  size_t tsize, unsigned flags)
{
	DynBuf* obj = dynbuf_cast(*pp);
	dynbuf_uint c0 = obj ? obj->count : 0;
	if (count > c0) count = c0;
	return dynbuf_resize(pp, c0 - count, tsize, flags, NULL);
}

static inline
DynBuf* dynbuf_remove(void** pp, dynbuf_uint i, dynbuf_uint count,
					  size_t tsize, unsigned flags)
{
	DynBuf* obj = dynbuf_cast(*pp);
	dynbuf_uint c0 = obj ? obj->count : 0;
	if (i < c0) {
		if (i+count > c0) count = c0 - i;
		else if (i+count < c0)
			memmove(obj->d+i*tsize, obj->d+(i+count)*tsize, (c0-i-count)*tsize);
		obj = dynbuf_resize(pp, c0 - count, tsize, flags, NULL);
	}
	return obj;
}

/* DynStr only */

static inline
int dynstr_copyz(DynStr* pp, const char* data, Allocator* a)
{
	size_t l = data ? strlen(data) : 0;
	if (!dynbuf_copy((void**)pp, l, data, sizeof(**pp), 1, a))
		return -1;
	return l;
}

static inline
int dynstr_insertz(DynStr* pp, dynbuf_uint i, const char* data, Allocator* a)
{
	if (!data) return 0;
	size_t l = strlen(data);
	if (!dynbuf_insert((void**)pp, i, l, data, sizeof(**pp), 1, a))
		return -1;
	return l;
}

static inline
int dynstr_appendz(DynStr* pp, const char* data, Allocator* a)
{
	if (!data) return 0;
	size_t l = strlen(data);
	if (!dynbuf_append((void**)pp, l, data, sizeof(**pp), 1, a))
		return -1;
	return l;
}

static inline
int dynstr_vprintf(DynStr* pp, unsigned flags, Allocator* a, const char* fmt, va_list ap)
{
	va_list ap2;
	va_copy(ap2, ap);

	int sz = vsnprintf(0, 0, fmt, ap);
	if (sz <= 0) goto end;

	unsigned c0 = (flags & 1) ? dynbuf_count(*pp) : 0;
	DynBuf* obj = dynbuf_resize((void**)pp, c0+sz, sizeof(**pp), 1, a);
	if (!obj) { sz=-1; goto end; }

	sz = vsnprintf((char*)obj->d+c0, sz+1, fmt, ap2);

end:
	va_end(ap2);
	return sz;
}

static inline
int dynstr_printf(DynStr* pp, unsigned flags, Allocator* a, const char* fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	int sz = dynstr_vprintf(pp, flags, a, fmt, ap);
	va_end(ap);
	return sz;
}
