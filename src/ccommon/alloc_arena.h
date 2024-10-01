/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 *
 * Simple and fast arena allocator that frees everything at once.
 * Optimized for small allocations that do not resize.
 * Can extend itself by allocating new arenas.
 *
 * Example:
 *   AllocatorArena al={0};
 *   char * d = alloc_arena_alloc(&al, 16);
 *   alloc_arena_free(&al);
 */
#pragma once
#include "alloc.h"

typedef struct AllocatorArena {	
	size_t rem;
	Allocator * al;  //Set this if a non-default allocator is desired
	struct AllocArenaPage {
		struct AllocArenaPage *prev;
		size_t size;
		uint8_t data[];
	} *page;
} AllocatorArena;

// Return an allocator using only the space provided.
int alloc_arena_frombuffer(AllocatorArena*, size_t sz, void* buf);

#define alloc_arena_fromarray(S, A) \
	alloc_arena_frombuffer((S), sizeof(A), (A))

// Allocate memory from it
void * alloc_arena_alloc(AllocatorArena* S, size_t sz);

// Reserve space at least <sz> bytes
int alloc_arena_reserve(AllocatorArena* S, size_t sz);

// Free all allocations
void alloc_arena_free(AllocatorArena* S);

//TODO: rollback: get mark and free up to it only

void * allocator_arena_alloc(Allocator* a, void* ptr, size_t sz, int flags);

void allocator_arena_ctx_free(Allocator* a);

// Returns a generic allocator interface
static inline
Allocator allocator_arena(AllocatorArena* S) {
	return (Allocator){
		.alloc = allocator_arena_alloc,
		.ctx_free = allocator_arena_ctx_free,
		.ctx = S };
}