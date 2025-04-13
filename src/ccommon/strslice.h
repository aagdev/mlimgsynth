/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#pragma once
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

typedef struct StrSlice {
	const char	*b;
	size_t		s;
} StrSlice;

// Initialization

#define strsl_make(B,S) \
	((StrSlice){ .b=(B), .s=(S) })

#define strsl_static(S) \
	((StrSlice){ .b=(S), .s=sizeof(S)-1 })

#define strsl_fromd(D) \
	((StrSlice){ .b=(D), .s=dstr_count(D) })

#define strsl_froma(A) \
	((StrSlice){ .b=(A).p.cp, .s=(A).len })

#define strsl_fromr(B,E) \
	((StrSlice){ .b=(B), .s=(E)-(B) })

static inline
StrSlice strsl_fromz(const char* strz)
	{ return (StrSlice){ .b=strz, .s=strlen(strz) }; }

// Access

static inline
intptr_t strsl_len(const StrSlice ss)
	{ return ss.s; }

static inline
const char * strsl_begin(const StrSlice ss)
	{ return ss.b; }

static inline
const char * strsl_end(const StrSlice ss)
	{ return ss.b + ss.s; }

#define strsl_for(S, VC, VE, I) \
	for (const char *VC=strsl_begin(S)+(I), *VE=strsl_end(S); VC<VE; ++VC)

// Unsafe slice
static inline
StrSlice strsl_slice_u(const StrSlice ss, size_t b, size_t e)
	{ return (StrSlice){ ss.b+b, ss.s-e-b }; }

// Operations

static inline
int strsl_cmp(const StrSlice s1, const StrSlice s2)
{
	const char *c1=s1.b, *e1=c1+s1.s,
			   *c2=s2.b, *e2=c2+s2.s;
	do {
		int v1 = (uint8_t)*c1;
		int v2 = (uint8_t)*c2;
		if (!(c1 < e1)) {
			if (!(c2 < e2))
				return 0;
			else
				return -v2;
		} else if (!(c2 < e2))
			return v1;

		int d = v1 - v2;
		if (d) return d;
		
		c1++;
		c2++;
	} while (1);
}

static inline
int strsl_cmpz(const StrSlice ss, const char* strz)
{
	for (const char *c=ss.b, *e=c+ss.s; c<e; ++c, ++strz) {
		if (!*strz) return *c ? *c : 1;
		int d = *c - *strz;
		if (d) return d;
	}
	return *strz;
}

static inline
size_t strsl_copyz(size_t bufsz, char* buf, const StrSlice ss)
{
	if (bufsz < 1) return 0;
	bufsz--;
	size_t len = strsl_len(ss);
	if (len > bufsz) len = bufsz;
	memcpy(buf, ss.b, len);
	buf[len] = 0;
	return len;
}

static inline
char* strsl_getz(size_t bufsz, char* buf, const StrSlice ss) {
	strsl_copyz(bufsz, buf, ss);
	return buf;
}

// Utility

static inline
int strsl_startswith(const StrSlice ss, const StrSlice prefix) {
	if (!(ss.s >= prefix.s)) return 0;
	return !memcmp(ss.b, prefix.b, prefix.s);
}

static inline
int strsl_endswith(const StrSlice ss, const StrSlice suffix) {
	if (!(ss.s >= suffix.s)) return 0;
	return !memcmp(ss.b+ss.s-suffix.s, suffix.b, suffix.s);
}

static inline
int strsl_prefix_trim(StrSlice* pss, const StrSlice prefix)
{
	if (!strsl_startswith(*pss, prefix)) return 0;
	pss->b += prefix.s;
	pss->s -= prefix.s;
	return 1;
}

static inline
int strsl_prefixz_trim(StrSlice* pss, const char* prefix) {
	return strsl_prefix_trim(pss, strsl_fromz(prefix));
}

static inline
int strsl_suffix_trim(StrSlice* pss, const StrSlice suffix)
{
	if (!strsl_endswith(*pss, suffix)) return 0;
	pss->s -= suffix.s;
	return 1;
}

static inline
int strsl_suffixz_trim(StrSlice* pss, const char* suffix) {
	return strsl_suffix_trim(pss, strsl_fromz(suffix));
}
