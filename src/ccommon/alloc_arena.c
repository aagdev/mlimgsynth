/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#include "alloc_arena.h"
#include "ccommon.h"
#include <string.h>

#ifndef ALLOC_arena_PAGE_SIZE
#define ALLOC_arena_PAGE_SIZE  (4096-16)
#endif

#ifndef ALLOC_arena_ALLOCATOR
#define ALLOC_arena_ALLOCATOR  g_allocator
#endif

int alloc_arena_frombuffer(AllocatorArena * S, size_t sz, void* buf)
{
	if (S->al) alloc_arena_free(S);
	if (sz < sizeof(*S->page)) return -1;
	S->al = NULL;
	S->page = buf;
	S->page->prev = NULL;
	S->rem = S->page->size = sz - sizeof(*S->page);
	return 1;
}

int alloc_arena_reserve(AllocatorArena* S, size_t size)
{
	if (S->rem >= size) return 0;
	if (!S->al) {
		if (!S->page) S->al = ALLOC_arena_ALLOCATOR;
		else return -1;
	}

	size += sizeof(struct AllocArenaPage);  //header size
	MAXSET(size, ALLOC_arena_PAGE_SIZE);  //minimum page size
	
	// Allocate a new page, previous page remaining space is lost
	struct AllocArenaPage * p = alloc_alloc(S->al, size);
	size = alloc_size_opt(S->al, p, size);
	p->prev = S->page;
	S->page = p;
	S->rem = p->size = size - sizeof(*p);
	return 1;
}

void * alloc_arena_alloc(AllocatorArena* S, size_t sz)
{
	if (sz > S->rem && alloc_arena_reserve(S, sz) < 0) return NULL;
	void * p = S->page->data + S->page->size - S->rem;
	S->rem -= sz;
	return p;
}

void alloc_arena_free_last(AllocatorArena* S, void* p_)
{
	if (!S->page) return;
	uint8_t *ini = S->page->data,
	        *end = S->page->data + S->page->size,
			*p = p_;
	if (ini <= p && p < end) {
		S->rem = end - p;
	}
}

void alloc_arena_free(AllocatorArena* S)
{
	if (S->al) {  //dynamic storage
		// Iterate over the pages and free them
		struct AllocArenaPage *cur, *prev=S->page;
		while ((cur = prev)) {
			prev = cur->prev;
			alloc_free(S->al, cur);
		}
		S->page = NULL;
		S->rem = 0;
	}
	else if (S->page) {  //static storage
		S->rem = S->page->size;
	}
}

void * allocator_arena_alloc(Allocator* a, void* ptr, size_t sz, int flags)
{
	AllocatorArena * S = a->ctx;
	//TODO: implement size storage?
	//TODO: alignment?
	if (a->flags & ALLOC_F_HAS_SIZE4) { alloc_fatal(a); return NULL; }
	if (ptr) {
		if (sz == 0) {
			// Free: no op
			return NULL;
		} else {
			// The old size is not known
			alloc_fatal(a);
			return NULL;
		}
	}
	void * p = alloc_arena_alloc(S, sz);
	if (flags & ALLOC_AF_ZERO)
		memset(p, 0, sz);
	//if (ptr && p && oldsz)
	//	memcpy(p, ptr, oldsz);
	return p;
}

void allocator_arena_ctx_free(Allocator* a)
{
	AllocatorArena * S = a->ctx;
	alloc_arena_free(S);
}
