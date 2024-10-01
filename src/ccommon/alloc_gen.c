/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#include "alloc_gen.h"
#include "ccommon.h"
#include <string.h>
#include <assert.h>

//TODO: ajustable segment size?
//TODO: automatic trim of unused segments?
//TODO: reserve() to have contiguous memory ready ? 
//TODO: runtime config? alignment, segment-size
//TODO: extension for large allocations (>ALLOC_SIZE_MAX) ?

//#ifndef ALLOC_SIZE_MAX
//#define ALLOC_SIZE_MAX ALLOC_SIZE_MASK
//#endif

// Custom general allocator
#define ALLOC_SIZE_MAX		0x0ffffffc
#define SIZE_MASK			0x0ffffffc
#define SIZE_F_VALID		0x80000000
#define SIZE_F_PFREE		0x40000000
#define SIZE_F_FREE			0x20000000

#define SEGMENT_SIZE		(65536*4)
#define CHUNK_ALIGN			ALLOC_SIZE_ALIGNMENT
#define CHUNK_SIZE_MIN		(CHUNK_ALIGN - sizeof(agsize_t))

typedef uint32_t agsize_t;

#define size_align(S,A) \
	(((S) + ((A) - 1)) & ~((A) - 1))

#define size_align_min(S,A) \
	((S) ? (((S) + ((A) - 1)) & ~((A) - 1)) : (A))

typedef agsize_t segment_t;
#define seg_next(S)			(*(segment_t**)(S))
#define seg_size(S)			(*((chunk_t*)(S) + sizeof(void*)/sizeof(chunk_t)))
#define seg_firstc(S)		((chunk_t*)(S) + CHUNK_ALIGN/sizeof(chunk_t))
#define seg_endc(S)			((chunk_t*)(S) + seg_size(S)/sizeof(chunk_t))

typedef agsize_t chunk_t;
#define chunk_sizeflags(C)	((C)[-1])
#define chunk_size(C)		((C)[-1] & SIZE_MASK)
#define chunk_sizeprev(C)	((C)[-2])
#define chunk_next(C)		((C) + chunk_size(C) / sizeof(chunk_t) + 1)
#define chunk_prev(C)		((C) - chunk_sizeprev(C) / sizeof(chunk_t) - 1)

#define chunk_valid_is(C) \
	(((C)[-1] & SIZE_F_VALID) && \
	 (((C)[-1] & SIZE_MASK) >= CHUNK_SIZE_MIN) && \
	 (chunk_next(C)[-1] & SIZE_F_VALID) )

#define chunk_used_valid_is(C) \
	(chunk_valid_is(C) && \
	 !((C)[-1] & SIZE_F_FREE) && \
	 !(chunk_next(C)[-1] & SIZE_F_PFREE) )

#define chunk_free_valid_is(C) \
	(chunk_valid_is(C) && \
	 ((C)[-1] & SIZE_F_FREE) && \
	 (chunk_next(C)[-1] & SIZE_F_PFREE) )

#define FCHUNK_SIZE_MIN		(sizeof(void*)*2 + sizeof(agsize_t))

#define fchunk_next(C)		(((chunk_t**)(C))[0])
#define fchunk_prev(C)		(((chunk_t**)(C))[1])

static inline int fchunk_valid_is(const chunk_t* C)
{
	if ((chunk_sizeflags(C) & (SIZE_F_VALID | SIZE_F_FREE)) !=
		(SIZE_F_VALID | SIZE_F_FREE)) return 0;
	if (!(chunk_size(C) >= FCHUNK_SIZE_MIN)) return 0;
	const chunk_t *Cp=fchunk_prev(C), *Cn=fchunk_next(C);
	if (!(Cp && fchunk_next(Cp) == C)) return 0;
	if (Cn && !(fchunk_prev(Cn) == C)) return 0;
	return 1;
}

#define FCHUNK_LIST_COUNT 10
#define fchunk_idx(S)  (\
	 ((S) >=  128) +((S) >=  256) +((S) >=  512) +((S) >=  1024) \
	+((S) >= 2048) +((S) >= 4096) +((S) >= 8192) +((S) >= 16384) \
	+((S) >= SEGMENT_SIZE))

/*#define FCHUNK_LIST_COUNT 4
#define fchunk_idx(S)  (((S) >= 256) + ((S) >= 1024) + ((S) >= 4096))*/

struct AllocGen {
	chunk_t *fchunk[FCHUNK_LIST_COUNT];
	segment_t * seg;
};

// System memory allocation
#if defined(__unix__)
#include <sys/mman.h>

#define ALLOC_GEN_SYS_FAIL  MAP_FAILED
#define alloc_gen_sys_free(P,S) \
	munmap((P), (S))

// Some systems do not have MAP_ANONYMOUS
#if !defined(MAP_ANONYMOUS) && defined(MAP_ANON)
#define MAP_ANONYMOUS  MAP_ANON
#endif
#ifdef MAP_ANONYMOUS
#define alloc_gen_sys_alloc(S) \
	mmap(0, (S), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)
#else
#include <fcntl.h>
static int dev_zero_fd = -1;
#define alloc_gen_sys_alloc(S) \
	((dev_zero_fd < 0) ? \
	 (dev_zero_fd = open("/dev/zero", O_RDWR), \
	  mmap(0, (S), PROT_READ|PROT_WRITE, MAP_PRIVATE, dev_zero_fd, 0)) : \
	  mmap(0, (S), PROT_READ|PROT_WRITE, MAP_PRIVATE, dev_zero_fd, 0))
#endif

#elif defined(__WIN32__)
#define WIN32_LEAN_AND_MEAN
#define WIN32_EXTRA_LEAN
#include <windows.h>

#define ALLOC_GEN_SYS_FAIL  NULL
#define alloc_gen_sys_alloc(S) \
	VirtualAlloc(0, (S), MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE)
#define alloc_gen_sys_free(P,S) \
	(VirtualFree((P), 0, MEM_RELEASE) ? 0 : -1)

#elif defined(__STDC_HOSTED__)
#include <stdlib.h>
#define ALLOC_GEN_SYS_FAIL  NULL
#define alloc_gen_sys_alloc(S)		malloc(S)
#define alloc_gen_sys_free(P,S)		free(P)

#else
#error No memory management backend found
#endif

//#include "stdio.h"
//#define alloc_log(...)  { fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); fflush(stderr); }
#define alloc_log(...)

// Update states to make a new free chunk reacheable
static inline void chunk_free_add(struct AllocGen* A, chunk_t* C)
{
	if (!C) return;

	agsize_t csz = chunk_size(C);
	chunk_sizeflags(C) |= SIZE_F_FREE;
	
	chunk_t * Cn = chunk_next(C);
	chunk_sizeflags(Cn) |= SIZE_F_PFREE;
	chunk_sizeprev(Cn) = csz;
	
	assert( chunk_free_valid_is(C) );
	
	if (csz >= FCHUNK_SIZE_MIN) {
		unsigned idx = fchunk_idx(csz);
		assert( idx < FCHUNK_LIST_COUNT );
		chunk_t * Cp = (chunk_t*) &A->fchunk[idx];
		Cn = A->fchunk[idx];
		unsigned nmax = (idx+1 == FCHUNK_LIST_COUNT) ? 0xffff : 0;
		for (; nmax-- && Cn && chunk_size(Cn) < csz; Cn=fchunk_next(Cn))
			Cp = Cn;

		fchunk_next(C) = Cn;
		fchunk_prev(C) = Cp;
		fchunk_next(Cp) = C;
		if (Cn) fchunk_prev(Cn) = C;
		
		assert( fchunk_valid_is(C) );
	}
}

static inline void chunk_free_remove(chunk_t* C)
{
	if (chunk_size(C) >= FCHUNK_SIZE_MIN) {
		assert( fchunk_valid_is(C) );
		chunk_t * Cn = fchunk_next(C);
		chunk_t * Cp = fchunk_prev(C);
		if (Cn) fchunk_prev(Cn) = Cp;
		if (Cp) fchunk_next(Cp) = Cn;
	}
}

// Merge contiguous free chunks
static inline int chunk_free_merge_fw(chunk_t* C)
{
	chunk_t * Cn = chunk_next(C);
	if (!(chunk_sizeflags(Cn) & SIZE_F_FREE)) return 0;
	chunk_free_remove(Cn);
	Cn = chunk_next(Cn);
	assert( !(chunk_sizeflags(Cn) & SIZE_F_FREE) );
	agsize_t csz = (Cn - C)*sizeof(*C) - sizeof(agsize_t);
	chunk_sizeflags(C) = csz | (chunk_sizeflags(C) & ~SIZE_MASK);
	assert( chunk_next(C) == Cn );
	assert( chunk_valid_is(C) );
	return 1;
}

static inline chunk_t* chunk_free_merge_bw(chunk_t* C)
{
	if (!(chunk_sizeflags(C) & SIZE_F_PFREE)) return C;
	chunk_t * Cp = chunk_prev(C);
	assert( chunk_sizeflags(Cp) & SIZE_F_FREE );
	assert( !(chunk_sizeflags(Cp) & SIZE_F_PFREE) );
	chunk_free_remove(Cp);
	agsize_t csz = (C - Cp)*sizeof(*C) + chunk_size(C);
	chunk_sizeflags(Cp) = csz | SIZE_F_FREE | SIZE_F_VALID;
	assert( chunk_valid_is(Cp) );
	return Cp;
}

static void* alloc_gen_seg_new(segment_t* next, agsize_t sz)
{
	sz = size_align(sz + CHUNK_ALIGN + sizeof(agsize_t), SEGMENT_SIZE);
	alloc_log("new segment %u", sz);
	agsize_t * S = alloc_gen_sys_alloc(sz);
	if (S == ALLOC_GEN_SYS_FAIL) return NULL;

	// Next segment pointer
	seg_next(S) = next;
	seg_size(S) = sz;
	// First chunk size and flags
	agsize_t * C = seg_firstc(S);
	agsize_t csz = sz - CHUNK_ALIGN - sizeof(agsize_t);
	chunk_sizeflags(C) = csz | SIZE_F_FREE | SIZE_F_VALID;
	// Zero-size chunk at the end
	assert( chunk_next(C) == seg_endc(S) );
	chunk_sizeflags( chunk_next(C) ) = SIZE_F_PFREE | SIZE_F_VALID;
	
	return S;
}

static inline void alloc_gen_seg_free(segment_t* S)
{
	agsize_t sz = seg_size(S);  sz=sz;
	alloc_gen_sys_free(S, sz);
}

static chunk_t* alloc_gen_chunk_use(chunk_t* C, agsize_t sz)
{
	sz = size_align(sz + sizeof(agsize_t), CHUNK_ALIGN);
	agsize_t csz = chunk_size(C);
	if (csz >= sz + CHUNK_ALIGN)
	{	// Divide chunk
		//alloc_log("chunk divide %u %u", csz, sz);
		chunk_t* C2 = C + sz/sizeof(chunk_t);
		chunk_sizeflags(C2) = (csz - sz) | SIZE_F_FREE | SIZE_F_VALID;
		chunk_sizeprev(C2) = csz - sz;
		chunk_sizeflags(C) = (sz - sizeof(agsize_t)) | SIZE_F_VALID
			| (chunk_sizeflags(C) & SIZE_F_PFREE);
		assert( chunk_next(C) == C2 );
		return C2;
	} else {
		chunk_sizeflags(C) &= ~SIZE_F_FREE;
		chunk_sizeflags( chunk_next(C) ) &= ~SIZE_F_PFREE;
		return NULL;
	}
}

static void* alloc_gen_alloc_new(Allocator* a, agsize_t sz)
{
	chunk_t *C=NULL, *Cn=NULL;
	segment_t *S=NULL;

	if (sz > ALLOC_SIZE_MAX) {
		alloc_fatal(a);
		return NULL;
	}

	// Initialization
	struct AllocGen * A = a->ctx;
	if (!A) {
		S = alloc_gen_seg_new(NULL, sz + sizeof(*A) + CHUNK_ALIGN);
		if (S == NULL) { alloc_fatal(a); return NULL; }
		A = a->ctx = C = seg_firstc(S);
		A->seg = S;
		C = alloc_gen_chunk_use(C, sizeof(*A));
		goto end;
	}

	// Check first free chunk
	for (unsigned idx=fchunk_idx(sz); idx<COUNTOF(A->fchunk); ++idx) {
		unsigned nmax = (idx+1 == FCHUNK_LIST_COUNT) ? 0xffff : 32;
		for (C=A->fchunk[idx]; nmax-- && C; C=fchunk_next(C)) {
			if (chunk_size(C) >= sz) {
				chunk_free_remove(C);
				goto end;
			}
		}
	}

	// Search over all the segments and its chunks
	/*for (S=A->seg; S; S=seg_next(S)) {
		agsize_t csz;
		for (C=seg_firstc(S); (csz = chunk_size(C)); C=chunk_next(C)) {
			if (chunk_sizeflags(C) & SIZE_F_FREE && csz >= sz) {
				chunk_free_remove(C);
				goto end;
			}
		}
	}*/

	// New segment
	S = alloc_gen_seg_new(A->seg, sz);
	if (S == NULL) { alloc_fatal(a); return NULL; }
	A->seg = S;
	C = seg_firstc(S);

end:
	assert( chunk_free_valid_is(C) );
	Cn = alloc_gen_chunk_use(C, sz);
	chunk_free_add(A, Cn);
	return C;
}

void* alloc_gen_alloc(Allocator* a, void* ptr, size_t sz, int flags)
{
	if (ptr) {
		struct AllocGen * A = a->ctx;
		chunk_t *C=ptr;
		if (!chunk_used_valid_is(C)) { alloc_fatal(a); return NULL; }
		if (sz == 0)
		{	// Free
			alloc_log("chunk free %u", chunk_size(C));
			chunk_free_merge_fw(C);
			C = chunk_free_merge_bw(C);
			chunk_free_add(A, C);
			return NULL;
		}
		else
		{	// Resize
			agsize_t csz = chunk_size(C);
			if (sz <= csz) return C;
			alloc_log("chunk resize %u -> %u", csz, (unsigned)sz);

			sz += sz >> ALLOC_RESIZE_MARGIN;

			chunk_t * Cf = C;
			int r = chunk_free_merge_fw(C);
			if (r && sz <= chunk_size(C)) {
				Cf = alloc_gen_chunk_use(C, sz);
			} else {
				chunk_t * Cp = C;
				C = chunk_free_merge_bw(C);
				if (C != Cp && sz <= chunk_size(C)) {
					memmove(C, Cp, csz);
					Cf = alloc_gen_chunk_use(C, sz);
				} else {
					Cf = C;
					C = alloc_gen_alloc_new(a, sz);
					assert( chunk_size(C) > csz );
					memcpy(C, Cp, csz);
				}
			}
			chunk_free_add(A, Cf);
			if (flags & ALLOC_AF_ZERO)
				memset(C+csz/sizeof(*C), 0, chunk_size(C) - csz);
			return C;
		}
	}
	else
	{	// New
		alloc_log("chunk alloc %u", (unsigned)sz);
		chunk_t* C = alloc_gen_alloc_new(a, sz);
		if (flags & ALLOC_AF_ZERO)
			memset(C, 0, chunk_size(C));
		return C;
	}
}

void alloc_gen_ctx_free(Allocator* a)
{
	if (!(a->alloc == &alloc_gen_alloc && a->ctx)) return;
	struct AllocGen * A = a->ctx;
	//TODO: reverse iteration of segments ?
	for (segment_t *Sp=A->seg, *S=seg_next(Sp); S; Sp=S, S=seg_next(Sp))
	{
		alloc_gen_seg_free(S);
		S = Sp;
	}
	assert( (void*)A == (void*)A->seg );
	alloc_gen_seg_free(A->seg);
	a->ctx = NULL;
}

void allocator_gen_trim(Allocator* a)
{
	if (!(a->alloc == &alloc_gen_alloc && a->ctx)) return;
	struct AllocGen * A = a->ctx;
	//TODO: reverse iteration of segments ?
	for (segment_t *Sp=A->seg, *S=seg_next(Sp); S; Sp=S, S=seg_next(Sp))
	{
		chunk_t *C=seg_firstc(S), *Ce=seg_endc(S);
		if (chunk_sizeflags(C) & SIZE_F_FREE && chunk_next(C) == Ce)
		{	// Free unused segment
			chunk_free_remove(C);
			seg_next(Sp) = seg_next(S);
			alloc_gen_seg_free(S);
			S = Sp;
		}
	}
}

AllocGenInfo allocator_gen_info(Allocator* a)
{
	AllocGenInfo info={0};
	if (a->alloc == &alloc_gen_alloc && a->ctx) {
		struct AllocGen * A = a->ctx;
		// Loop over the segments
		for (segment_t *S=A->seg; S; S=seg_next(S))
		{
			info.nseg++;
			info.mtot += CHUNK_ALIGN;
			// Loop over the chunks of the current segment
			agsize_t csz;
			for (chunk_t *C=seg_firstc(S); (csz = chunk_size(C)); C=chunk_next(C))
			{
				info.nchunk++;
				info.mtot += csz + sizeof(agsize_t);
				if (chunk_sizeflags(C) & SIZE_F_FREE) {
					assert( chunk_free_valid_is(C) );
					info.nchunkf++;
					info.mfree += csz;
				}
				else
					assert( chunk_used_valid_is(C) );
			}
		}
		// Loop over free chunks
		for (unsigned idx=0; idx<COUNTOF(A->fchunk); ++idx) {
			for (chunk_t *C=A->fchunk[idx]; C; C=fchunk_next(C))
			{
				assert( chunk_free_valid_is(C) );
				info.nfchunk++;
				info.mfchunk += chunk_size(C);
			}
		}
	}
	return info;
}

// Global allocators
#ifndef CC_ALLOC_GLOBAL_USE_STDLIB
Allocator global_allocator =
			{ alloc_gen_alloc, alloc_gen_ctx_free, NULL, NULL, ALLOC_F_HAS_SIZE4 },
          *g_allocator = &global_allocator,
          *g_allocator_dopt = &global_allocator;
#endif
