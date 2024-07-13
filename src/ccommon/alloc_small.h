/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 *
 * Simple and fast arena allocator that frees everything at once.
 * Optimized for small allocations that do not resize.
 * Can extend itself by allocating new arenas.
 *
 * Example:
 *   AllocatorSmall al={0};
 *   char * d = alloc_small_alloc(&al, 16);
 *   alloc_small_free(&al);
 */
#pragma once
#include "alloc.h"

typedef struct AllocatorSmall {	
	size_t rem;
	Allocator * al;  //Set this if a non-default allocator is desired
	struct AllocSmallPage {
		struct AllocSmallPage *prev;
		size_t size;
		uint8_t data[];
	} *page;
} AllocatorSmall;

// Return an allocator using only the space provided.
int alloc_small_frombuffer(AllocatorSmall*, size_t sz, void* buf);

#define alloc_small_fromarray(S, A) \
	alloc_small_frombuffer((S), sizeof(A), (A))

// Allocate memory from it
void * alloc_small_alloc(AllocatorSmall* S, size_t sz);

// Reserve space at least <sz> bytes
int alloc_small_reserve(AllocatorSmall* S, size_t sz);

// Free all allocations
void alloc_small_free(AllocatorSmall* S);

//TODO: rollback: get mark and free up to it only

/*void * allocator_small_alloc(Allocator* a, void* ptr, size_t oldsz,
	size_t newsz, size_t align, int flags);

// Returns a generic allocator interface
static inline
Allocator allocator_small(AllocatorSmall* S) {
	return (Allocator){ .alloc=&allocator_small_alloc, .ctx=S };
}*/
