/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 *
 * Storage of unique string slices.
 *
 * Example:
 *   StringStore ss={0};
 *   StringInt si = strsto_add(&ss, strsl_static("apple"));
 *   assert( !strsl_cmp(strsto_get(&ss, si), strsl_static("apple")) );
 *   strsto_free(&ss);
 */
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "strslice.h"
#include "alloc_arena.h"
#include "vector.h"

typedef int32_t StringInt;

typedef struct StringStore {
	StrSlice * s;  //vector
	unsigned * idx;  //index, vector
	AllocatorArena al;
} StringStore;

void strsto_free(StringStore* S);

static inline
unsigned strsto_count(const StringStore* S)
	{ return vec_count(S->idx); }

static inline
unsigned strsto_next_idx(const StringStore* S)
	{ return vec_count(S->s); }

static inline
StrSlice strsto_get(const StringStore* S, StringInt idx) {
	assert(0 <= idx && idx < vec_count(S->s));
	if (!(0 <= idx && idx < vec_count(S->s))) return (StrSlice){0};
	return S->s[idx];
}

// Return -1 if not found
StringInt strsto_find(const StringStore* S, const StrSlice ss);

StringInt strsto_add2(StringStore* S, const StrSlice ss, StringInt idx,
	bool static_);

// Add an string.
static inline
StringInt strsto_add(StringStore* S, const StrSlice ss) {
	return strsto_add2(S, ss, -1, false);
}

// Find longest string in the store that matches the beginning of key.
StringInt strsto_find_prefix(const StringStore* S, const StrSlice key);

// Find the position in the index <idx> for <key>.
// Returns true if <key> is present in the store.
// Then, S->idx[*idx] is the StringInt.
bool strsto_iidx_find(const StringStore* S, const StrSlice key, size_t* idx);

/* Utility */
static inline
char* strsl_getd(DynStr* buf, const StrSlice ss) {
	dstr_copy(*buf, strsl_len(ss), ss.b);
	return *buf;
}

//static inline
//StrSlice strsl_fromd(const DynStr buf) {
//	return (StrSlice){ .b=buf, .s=dstr_count(buf) };
//}
