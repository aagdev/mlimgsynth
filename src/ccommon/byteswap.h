/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 *
 * Byte order (endianness) convertion.
 */
#pragma once
#include <assert.h>

/*
*/
static inline bool little_endian_is() { int n=1; return *(char*)&n; }

static inline bool big_endian_is() { int n=1; return !*(char*)&n; }

/*
*/
static inline
void byteswap16(void* p) {
	unsigned char tmp, *b=p;
	tmp = b[0]; b[0] = b[1]; b[1] = tmp;
}
static inline
void byteswap16le(void* p) { if (!little_endian_is()) byteswap16(p); }
static inline
void byteswap16be(void* p) { if (!big_endian_is()) byteswap16(p); }

static inline
void byteswap32(void* p) {
	unsigned char tmp, *b=p;
	tmp = b[0]; b[0] = b[3]; b[3] = tmp;
	tmp = b[1]; b[1] = b[2]; b[2] = tmp;
}
static inline
void byteswap32le(void* p) { if (!little_endian_is()) byteswap32(p); }
static inline
void byteswap32be(void* p) { if (!big_endian_is()) byteswap32(p); }

static inline
void byteswap64(void* p) {
	unsigned char tmp, *b=p;
	tmp = b[0]; b[0] = b[7]; b[7] = tmp;
	tmp = b[1]; b[1] = b[6]; b[6] = tmp;
	tmp = b[2]; b[2] = b[5]; b[5] = tmp;
	tmp = b[3]; b[3] = b[4]; b[4] = tmp;
}
static inline
void byteswap64le(void* p) { if (!little_endian_is()) byteswap64(p); }
static inline
void byteswap64be(void* p) { if (!big_endian_is()) byteswap64(p); }

static inline
void byteswap(unsigned n, void* p) {
	switch (n) {
	case 2: byteswap16(p); break;
	case 4: byteswap32(p); break;
	case 8: byteswap64(p); break;
	}
}
static inline
void byteswaple(unsigned n, void* p) {
	if (!little_endian_is()) byteswap(n, p);
}
static inline
void byteswapbe(unsigned n, void* p) {
	if (!big_endian_is()) byteswap(n, p);
}

/*
*/
static inline
void byteswap_copy16(void*restrict dst, const void*restrict src) {
	unsigned char *d=dst;
	unsigned char const *s=src;
	d[0] = s[0];
	d[1] = s[1];
}
static inline
void byteswap_swap16(void*restrict dst, const void*restrict src) {
	unsigned char *d=dst;
	unsigned char const *s=src;
	d[0] = s[1];
	d[1] = s[0];
}
static inline
void byteswap_copy16le(void*restrict dst, const void*restrict src) {
	if (!little_endian_is()) byteswap_swap16(dst, src);
	else byteswap_copy16(dst, src);
}
static inline
void byteswap_copy16be(void*restrict dst, const void*restrict src) {
	if (!big_endian_is()) byteswap_swap16(dst, src);
	else byteswap_copy16(dst, src);
}

static inline
void byteswap_copy32(void*restrict dst, const void*restrict src) {
	unsigned char *d=dst;
	unsigned char const *s=src;
	d[0] = s[0];
	d[1] = s[1];
	d[2] = s[2];
	d[3] = s[3];
}
static inline
void byteswap_swap32(void*restrict dst, const void*restrict src) {
	unsigned char *d=dst;
	unsigned char const *s=src;
	d[0] = s[3];
	d[1] = s[2];
	d[2] = s[1];
	d[3] = s[0];
}
static inline
void byteswap_copy32le(void*restrict dst, const void*restrict src) {
	if (!little_endian_is()) byteswap_swap32(dst, src);
	else byteswap_copy32(dst, src);
}
static inline
void byteswap_copy32be(void*restrict dst, const void*restrict src) {
	if (!big_endian_is()) byteswap_swap32(dst, src);
	else byteswap_copy32(dst, src);
}

static inline
void byteswap_copy64(void*restrict dst, const void*restrict src) {
	unsigned char *d=dst;
	unsigned char const *s=src;
	d[0] = s[0];
	d[1] = s[1];
	d[2] = s[2];
	d[3] = s[3];
	d[4] = s[4];
	d[5] = s[5];
	d[6] = s[6];
	d[7] = s[7];
}
static inline
void byteswap_swap64(void*restrict dst, const void*restrict src) {
	unsigned char *d=dst;
	unsigned char const *s=src;
	d[0] = s[7];
	d[1] = s[6];
	d[2] = s[5];
	d[3] = s[4];
	d[4] = s[3];
	d[5] = s[2];
	d[6] = s[1];
	d[7] = s[0];
}
static inline
void byteswap_copy64le(void*restrict dst, const void*restrict src) {
	if (!little_endian_is()) byteswap_swap64(dst, src);
	else byteswap_copy64(dst, src);
}
static inline
void byteswap_copy64be(void*restrict dst, const void*restrict src) {
	if (!big_endian_is()) byteswap_swap64(dst, src);
	else byteswap_copy64(dst, src);
}

static inline
void byteswap_copy(unsigned n, void*restrict dst, const void*restrict src) {
	switch (n) {
	case 1: *(unsigned char*)dst = *(unsigned char*)src; break;
	case 2: byteswap_copy16(dst, src); break;
	case 4: byteswap_copy32(dst, src); break;
	case 8: byteswap_copy64(dst, src); break;
	default: assert(false);
	}
}
static inline
void byteswap_swap(unsigned n, void*restrict dst, const void*restrict src) {
	switch (n) {
	case 1: *(unsigned char*)dst = *(unsigned char*)src; break;
	case 2: byteswap_swap16(dst, src); break;
	case 4: byteswap_swap32(dst, src); break;
	case 8: byteswap_swap64(dst, src); break;
	default: assert(false);
	}
}
static inline
void byteswap_copyle(unsigned n, void*restrict dst, const void*restrict src) {
	if (!little_endian_is()) byteswap_swap(n, dst, src);
	else byteswap_copy(n, dst, src);
}
static inline
void byteswap_copybe(unsigned n, void*restrict dst, const void*restrict src) {
	if (!big_endian_is()) byteswap_swap(n, dst, src);
	else byteswap_copy(n, dst, src);
}
