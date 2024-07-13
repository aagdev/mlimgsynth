/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#pragma once
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

//! Checks if a character is an string
static inline
bool char_in_str(int ch, const char* str) {
	for (; *str; ++str) if (ch == *str) return true;
	return false;
}

//! Finds the first occurrence of a character in a list
static inline
char* str_chars_find(const char* str, const char* chars) {
	while (*str && !char_in_str(*str, chars)) str++;
	return (char*)str;
}

//! Finds the first occurrence of a character in a list
static inline
char* stre_chars_find(char* str, char* end, const char* chars) {
	while (str<end && !char_in_str(*str, chars)) str++;
	return str;
}

//! Skips all the occurrences of a list of characters at the beginning of a string
static inline
char* str_ltrim(char* str, const char* spaces) {
	while (*str && char_in_str(*str, spaces)) str++;
	return str;
}

//! Skips all the occurrences of a list of characters at the beginning of a string
static inline
char* stre_ltrim(char* str, char* end, const char* spaces) {
	while (str<end && char_in_str(*str, spaces)) str++;
	return str;
}

//! Skips all the occurrences of a list of characters at the end of a string
static inline
char* stre_rtrim(char* str, char* end, const char* spaces) {
	end--;
	while (str<end && char_in_str(*end, spaces)) end--;
	return end+1;
}

//! Skips all the occurrences of a list of characters at the beginning and the end of a string
static inline
void stre_trim(char** str, char** end, char const* spaces) {
	*str = stre_ltrim(*str, *end, spaces);
	*end = stre_rtrim(*str, *end, spaces);
}

//! Copy an string
static inline
char* stre_copy(unsigned dsize, char* dst, const char* str, const char* end)
{
	if (dsize) {
		dsize--;
		if (dsize > end-str) dsize = end-str;
		memcpy(dst, str, dsize);
		dst[dsize] = 0;
	}
	return dst;
}

//! Compare two string in case insensitive way
static inline
int str_cmp_i(const char* a, const char* b) {
	for (;; ++a, ++b) {
		int d = tolower((unsigned char)*a) - tolower((unsigned char)*b);
		if (d != 0 || !*a)
			return d;
	}
}

static inline
const char* str_startswith(const char* str, const char* sub)
{
	unsigned ls = strlen(str),
			 l2 = strlen(sub);
	if (ls >= l2 && !memcmp(str, sub, l2)) return str+l2;
	return NULL;
}

static inline
const char* str_endswith(const char* str, const char* sub)
{
	unsigned ls = strlen(str),
			 l2 = strlen(sub);
	if (ls >= l2 && !memcmp(str+ls-l2, sub, l2)) return str+ls-l2;
	return NULL;
}

//! Convert an string to lower case
static inline
size_t str_tolower(char* dst, size_t max, const char* src) {
	char *cur=dst, *end = dst+max;
	for(; *src && cur<end; ++src, ++cur)
		*cur = tolower((unsigned char)*src);
	if (cur == end) cur--;
	*cur = 0;
	return (cur - dst);
}

//! Parse an string to a boolean value
static inline
bool str_to_bool(const char* text) {
	if (!strcmp(text, "1")) return true;
	if (!str_cmp_i(text, "y")) return true;
	if (!str_cmp_i(text, "yes")) return true;
	if (!str_cmp_i(text, "true")) return true;
	return false;
}

//! Parse a character to single digit value in a base
static inline
int digit_decode(int c, int base) {
	if (!( 2 <= base && base <= 36)) base = 10;
	if ('0' <= c && c <= '9') return c - '0';
	if ('A' <= c && c < 'A'+base-10) return c - 'A' + 10;
	if ('a' <= c && c < 'a'+base-10) return c - 'a' + 10;
	return -1;
}

//! Checks if a string is a member of a list of strings
static inline
bool strlist_in(unsigned count, char** list, const char* str)
{
	while (count--)
		if (!strcmp(list[count], str))
			return true;
	return false;
}

//! Copy an string dynamically allocating memory as needed
static inline
char* strcpy_alloc(char** dst, const char* src) {
	size_t size = strlen(src)+1;
	char* p = realloc(*dst, size);
	if (!p) return 0;
	*dst = p;
	memcpy(p, src, size);
	return p;
}

#ifdef __GNUC__
__attribute__((format(printf, 2, 3)))
#endif
int sprintf_alloc(char** buffer, const char* fmt, ...);

size_t string_escape_encode(char* out, size_t out_size,
	const char* in, size_t in_size, size_t* in_done);

size_t string_escape_decode(char* out, size_t out_size,
	const char* in, size_t in_size, size_t* in_done);
