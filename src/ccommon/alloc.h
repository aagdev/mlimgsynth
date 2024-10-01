/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 *
 * Common allocator interface.
 * Handles failure calling an special function instead of returning NULL,
 * so, there is no need to check for errors.
 */
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <stdlib.h>

//TODO: modify vector.h to take advantage of size info ?
//TODO: compile option to use stdlib instead of the custom allocator

#ifndef ALLOC_SIZE_ALIGNMENT
#define ALLOC_SIZE_ALIGNMENT 16
#endif

#ifndef ALLOC_RESIZE_MARGIN
#define ALLOC_RESIZE_MARGIN 4  //+6%
#endif

/* Allocator interface */

typedef struct Allocator Allocator;
struct Allocator {
	// Allocation, reallocation and freeing of memory.
	void * (*alloc)(Allocator* a, void* ptr, size_t sz, int flags);
	// Free all associated memory (if possible).
	void (*ctx_free)(Allocator* a);
	// Handles fatal errors (out of memory). Can be NULL or user supplied.
	void (*fatal)(const Allocator* a);
	// Allocator context
	void * ctx;
	// Options
	int flags;
};

// Allocator flags
enum {
	// Store size of each allocation in the previous 4 bytes
	ALLOC_F_HAS_SIZE4	= 1,
	ALLOC_F_HAS_SIZE	= ALLOC_F_HAS_SIZE4,
	// Set of flags for efficient dynamic arrays
	ALLOC_F_DOPTIMAL	= ALLOC_F_HAS_SIZE4,
};

// alloc() flags
enum {
	ALLOC_AF_ZERO = 1,  //Zero memory (new allocation only)
};

// Checks if an allocator is ready to use
static inline bool allocator_good(const Allocator* a) {
	return !!a->alloc;
}

// Free all the memory associated with the allocator (if it corresponds).
// May be a no-op.
static inline void allocator_free(Allocator* a) {
	if (a->ctx_free) a->ctx_free(a);
}

// This called to handle fatal errors (out of memory)
static inline void alloc_fatal(const Allocator* a) {
	if (a->fatal) a->fatal(a);
	abort();
}

#define ALLOC_SIZE_MASK  0x0ffffffc

// Allocates a new block
#ifdef __GNUC__
__attribute((malloc, alloc_size(2)))
#endif
static inline
void * alloc_alloc(Allocator* a, size_t sz) {
	void * p = a->alloc(a, NULL, sz, ALLOC_AF_ZERO);
	if (!p && sz) alloc_fatal(a);
	return p;
}

// Allocates a new blocks of C elements of type T
#define alloc_new(A, T, C) \
	((T*)alloc_alloc((A), sizeof(T)*(C)))

/* Get the size of a block.
 * May be larger than the requested size. The additional space can be used normally.
 * Returns zero if not supported.
 */
static inline
size_t alloc_size(const Allocator* a, const void* p) {
	if (!(a && a->flags & ALLOC_F_HAS_SIZE4)) return 0; 
	return p ? ((uint32_t*)p)[-1] & ALLOC_SIZE_MASK : 0;
}

/* Get the size of a block.
 * Returns <def> if not known.
 */
static inline
size_t alloc_size_opt(const Allocator* a, const void* p, size_t def) {
	if (!(a && a->flags & ALLOC_F_HAS_SIZE4)) return def;
	return alloc_size(a, p);
}

// Changes the size of a block
#ifdef __GNUC__
__attribute((malloc, alloc_size(3)))
#endif
static inline
void * alloc_realloc(Allocator* a, void* p, size_t sz) {
	if (a->flags & ALLOC_F_HAS_SIZE4 && sz <= alloc_size(a, p)) return p;
	p = a->alloc(a, p, sz, 0);
	if (!p && sz) alloc_fatal(a);
	return p;
}

#define alloc_resize(A, P, T, C) \
	((T*)alloc_realloc((A), (P), sizeof(T)*(C)))

// Frees a block
static inline
void alloc_free(Allocator* a, void* p) {
	if (!p) return;
	a->alloc(a, p, 0, 0);
}

/* Global allocators for modules that can not take it as a parameter.
 * May be modified by the user.
 */
extern Allocator *g_allocator, *g_allocator_dopt;

/* Standard library wrapper
 */
#if __STDC_HOSTED__
void* alloc_stdlib_alloc(Allocator* a, void* ptr, size_t sz, int flags);

/* Returns a wrapper allocator for stdlib.
 */
static inline
Allocator allocator_stdlib() {
	return (Allocator){ alloc_stdlib_alloc, NULL, NULL, NULL, 0 };
}

/* Returns a wrapper allocator for stdlib.
 * Optimized for efficient dynamic arrays (frequent reallocations).
 */
static inline
Allocator allocator_stdlib_dopt() {
	return (Allocator){ alloc_stdlib_alloc, NULL, NULL, NULL, ALLOC_F_DOPTIMAL };
}
#endif

/* Utility */

// Round-up to the nearest power of two (up to 32 bits)
static inline
size_t size_round2(size_t v)
{
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}
