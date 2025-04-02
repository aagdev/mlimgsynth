/* Copyright 2024-2025, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 *
 * Unicode, UTF-8, encoding and decoding.
 */
#pragma once
#include <stdint.h>

/* Return the next code point and advance the string pointer.
 * Return zero for an empty string.
 * In case of error, returns 0xFFFD and skips the bytes.
 */
uint32_t utf8_decode_next(const char** pstr, const char* end);

/* Skip one codepoint without fully decoding it.
 * Returns a pointer to the next codepoint.
 * Returns <cur> if cur == end.
 */
const char* utf8_decode_skip(const char* cur, const char* end);

/* Encode one code point into cursor.
 * Writes up to 4 bytes. 
 * Return the new cursor position.
 */
char* utf8_encode_next(char* dst, uint32_t cp);
