/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#include "alloc.h"

#define size_align(S,A) \
	(((S) + ((A) - 1)) & ~((A) - 1))

// libc wrapper
#if __STDC_HOSTED__
#include <stdlib.h>
void* alloc_stdlib_alloc(Allocator* a, void* p, size_t sz, int flags)
{
	if (a->flags & ALLOC_F_HAS_SIZE4) {
		if (p) p = (uint8_t*)p - ALLOC_SIZE_ALIGNMENT;
		if (sz) sz = size_align(sz + ALLOC_SIZE_ALIGNMENT, ALLOC_SIZE_ALIGNMENT);
	}
	//if (a->flags & ALLOC_F_ROUND2 && sz) {
	//	sz = size_round2(sz);
	//}
	if (p) {
		if (sz == 0) {
			free(p);
			p = NULL;
		} else {
			sz += sz >> ALLOC_RESIZE_MARGIN;
			p = realloc(p, sz);
			if (!p) alloc_fatal(a);
		}
	} else {
		if (flags & ALLOC_AF_ZERO) {
			p = calloc(1, sz);
		} else {
			p = malloc(sz);
		}
		if (!p) alloc_fatal(a);
	}
	if (a->flags & ALLOC_F_HAS_SIZE4 && p) {
		p = (uint8_t*)p + ALLOC_SIZE_ALIGNMENT;
		((uint32_t*)p)[-1] = sz - ALLOC_SIZE_ALIGNMENT;
	}
	return p;
}
#endif

// Global allocators
#ifdef CC_ALLOC_GLOBAL_USE_STDLIB
Allocator global_allocator =
			{ alloc_stdlib_alloc, NULL, NULL, NULL, 0 },
          *g_allocator = &global_allocator;

Allocator global_allocator_dopt =
			{ alloc_stdlib_alloc, NULL, NULL, NULL, ALLOC_F_DOPTIMAL },
          *g_allocator_dopt = &global_allocator_dopt;
#endif
