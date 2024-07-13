/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#include "alloc_small.h"
#include "ccommon.h"
#include <string.h>

#ifndef ALLOC_SMALL_PAGE_SIZE
#define ALLOC_SMALL_PAGE_SIZE  (4096-16)
#endif

#ifndef ALLOC_SMALL_ALLOCATOR
#define ALLOC_SMALL_ALLOCATOR  g_allocator
#endif

int alloc_small_frombuffer(AllocatorSmall * S, size_t sz, void* buf)
{
	if (S->al) alloc_small_free(S);
	if (sz < sizeof(*S->page)) return -1;
	S->al = NULL;
	S->page = buf;
	S->page->prev = NULL;
	S->rem = S->page->size = sz - sizeof(*S->page);
	return 1;
}

void * alloc_small_alloc(AllocatorSmall* S, size_t sz)
{
	if (sz > S->rem && alloc_small_reserve(S, sz) < 0) return NULL;
	void * p = S->page->data + S->page->size - S->rem;
	S->rem -= sz;
	return p;
}

int alloc_small_reserve(AllocatorSmall* S, size_t size)
{
	if (S->rem >= size) return 0;
	if (!S->al) {
		if (!S->page) S->al = ALLOC_SMALL_ALLOCATOR;
		else return -1;
	}

	size += sizeof(struct AllocSmallPage);  //header size
	MAXSET(size, ALLOC_SMALL_PAGE_SIZE);  //minimum page size
	
	// Allocate a new page, previous page remaining space is lost
	struct AllocSmallPage * p = alloc_alloc(S->al, size);
	size = alloc_size_opt(S->al, p, size);
	p->prev = S->page;
	S->page = p;
	S->rem = p->size = size - sizeof(*p);
	return 1;
}

void alloc_small_free(AllocatorSmall* S)
{
	if (S->al) {  //dynamic storage
		// Iterate over the page and free them
		struct AllocSmallPage *cur, *prev=S->page;
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

/*void * allocator_small_alloc(Allocator* a, void* ptr, size_t oldsz,
	size_t newsz, size_t align, int flags)
{
	AllocatorSmall * S = a->ctx;
	if (ptr) {
		if (newsz == 0) {
			//free: no op
			//TODO: this can be improved if ptr is the last allocation
			//TODO: same for realloc
			return NULL;
		} else {
			//realloc: only if the old size is given
			if (!oldsz) alloc_fatal(a);
		}
	}
	void * p = alloc_small_alloc(S, newsz);
	if (ptr && p && oldsz)
		memcpy(p, ptr, oldsz);
	return p;
}*/
