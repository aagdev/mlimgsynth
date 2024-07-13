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

#define strsl_static(S) \
	((StrSlice){ .b=(S), .s=sizeof(S)-1 })

static inline
StrSlice strsl_fromz(const char* strz)
	{ return (StrSlice){ .b=strz, .s=strlen(strz) }; }

#define strsl_fromd(D) \
	((StrSlice){ .b=(D), .s=dstr_count(D) })

#define strsl_froma(A) \
	((StrSlice){ .b=(A).p.cp, .s=(A).len })

#define strsl_fromr(S,E) \
	((StrSlice){ .b=(S), .s=(E)-(S) })

static inline
intptr_t strsl_len(const StrSlice ss)
	{ return ss.s; }

static inline
int strsl_cmp(const StrSlice s1, const StrSlice s2)
{
	const char *c1=s1.b, *e1=c1+s1.s,
			   *c2=s2.b, *e2=c2+s2.s;
	do {
		if (!(c1 < e1)) {
			if (!(c2 < e2))
				return 0;
			else
				return - *c2;
		} else if (!(c2 < e2))
			return *c1;

		int d = *c1 - *c2;
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
