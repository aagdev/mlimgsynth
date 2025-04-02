/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#pragma once
#include <stdint.h>
#include "ccommon/unicode.h"
#include "ccommon/unicode_data.h"

static inline
int chr_ascii_space_is(int c) {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v'
		|| c == '\f';
}

static inline
int chr_ascii_lower(int c) {
	return ('A' <= c && c <= 'Z') ? (c+'a'-'A') : c;
}

static inline
void str_ascii_lower(char* cur, const char* end) {
	for (; cur<end; ++cur)
		if ('A' <= *cur && *cur <= 'Z') *cur += 'a' - 'A';
}

static inline
int str_match_advance_multiple(const char** pcur, const char* end, int b_lower,
	const char** str_list)
{
	for (int idx=0; str_list[idx]; ++idx) {
		const char *str = str_list[idx];
		const char *cur = *pcur;
		for (; cur < end && *str; ++cur, ++str) {
			int c = b_lower ? chr_ascii_lower(*cur) : *cur;
			if (c != *str) break;
		}
		if (*str == 0) {  // Match
			*pcur = cur;
			return idx;
		}
	}
	return -1;
}
#define str_match_advance_multiple(PCUR, END, LOWER, ...) \
	str_match_advance_multiple((PCUR), (END), (LOWER), \
		(const char*[]){__VA_ARGS__, NULL})

static inline
const char* str_unicode_space_skip(const char* cur, const char* end)
{
	while (cur < end) {
		if (chr_ascii_space_is(*cur)) {
			cur++;
			continue;
		}
		const char *prev = cur;
		uint32_t cp = utf8_decode_next(&cur, end);
		int cat = unicode_category_major(cp);
		if (cat != 'Z') {
			cur = prev;
			break;
		}
	}
	return cur;
}
