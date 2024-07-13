/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#include "stringstore.h"
#include "bisect.h"
#include "alloc_small.h"

/* */
void strsto_free(StringStore* S)
{
	alloc_small_free(&S->al);
	vec_free(S->idx);
	vec_free(S->s);
}

bool strsto_iidx_find(const StringStore* S, const StrSlice key, size_t* idx)
{
	bool found;
	BISECT_RIGHT(found, *idx, 0, vec_count(S->idx), strsl_cmp(S->s[S->idx[i_]], key) );
	return found;
}

StringInt strsto_find(const StringStore* S, const StrSlice ss)
{
	size_t iidx;
	return strsto_iidx_find(S, ss, &iidx) ? S->idx[iidx] : -1;
}

StringInt strsto_find_prefix(const StringStore* S, const StrSlice key)
{
	if (!vec_count(S->idx)) return -1;  //empty store

	size_t iidx;
	bool found = strsto_iidx_find(S, key, &iidx);
	if (found) return S->idx[iidx];  //exact match

	bool last=false;  //last attempt
	while (1) {
		StringInt si = S->idx[iidx];
		const StrSlice str = S->s[si];
		
		// Count the matching characters
		size_t i=0;
		while (i<str.s && i<key.s && str.b[i] == key.b[i]) i++;
		
		if (i != str.s) {  //key does not starts with str, try a shorter str
			if (!iidx) return -1;
			if (!i && last) return -1;  //no matching prefix found
			last = !i;  //length-1 cases
			iidx--;
		} else {
			assert(i != key.s);  //would be exact
			return si;
		}
	}
}

StringInt strsto_add2(StringStore* S, const StrSlice ss, StringInt idx,
	bool static_)
{
	size_t iidx;
	if (strsto_iidx_find(S, ss, &iidx))
	{
		if (idx >= 0 && idx != S->idx[iidx]) return -1;
		return S->idx[iidx];
	}
	else
	{
		unsigned n = vec_count(S->s);
		if (idx < 0) idx = n;
		
		if (idx < n) {
			// Index already used
			if (S->s[idx].b) return -1;
		} else {
			vec_append_zero(S->s, idx-n+1);
		}

		if (static_)
			S->s[idx] = ss;
		else {
			// Copy string
			char * p = alloc_small_alloc(&S->al, ss.s+1);
			memcpy(p, ss.b, ss.s);
			p[ss.s] = 0;
			S->s[idx] = (StrSlice){ .b=p, .s=ss.s };
		}
			
		vec_insert(S->idx, iidx, 1, &idx);
		return idx;
	}
}
