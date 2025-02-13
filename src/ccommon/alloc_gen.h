/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 *
 * General purpose allocator.
 * Loosely-based in dlmalloc.
 */
#pragma once
#include "alloc.h"

void* alloc_gen_alloc(Allocator* a, void* ptr, size_t sz, int flags);

void alloc_gen_ctx_free(Allocator* a);

/* Returns a new allocator.
 */
static inline
Allocator allocator_gen() {
	return (Allocator){ &alloc_gen_alloc, &alloc_gen_ctx_free,
		NULL, NULL, ALLOC_F_HAS_SIZE4 };
}

// Reduces the memory used to a minimum
void allocator_gen_trim(Allocator* a);

// Free all the allocated memory, but not the allocator itself
//void allocator_gen_free_all(Allocator* a);

// Return nonzero if the allocator has no allocations besides the space
// used internally. Useful to detect memory leaks.
int allocator_gen_empty_is(const Allocator* a);

// Return various summary statistics
// The values are calculated on the spot, so it could be slow.
typedef struct AllocGenInfo {
	size_t		mtot,		// Total memory allocated from the system
				mfree,		// Free memory
				mfchunk;
	unsigned	nseg,		// Number of segments
				nchunk,		// Number of chunks
				nchunkf,	// Number of free chunks
				nfchunk;
} AllocGenInfo;
AllocGenInfo allocator_gen_info(const Allocator* a);
