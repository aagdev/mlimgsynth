/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 *
 * Unicode database.
 * Query the properties of codepoints.
 * This is a separate module from "unicode" because the data occupies several
 * kilobytes.
 */
#pragma once
#include <stdint.h>

/* Get the major general category of a unicode codepoint.
 * Returns one of the following characters or zero the codepoint is out unicode
 * range.
 * L: Letter, M: Mark, N: Number, P: Punctuation, S: Symbol, Z: Separator, C: Other
 */
int unicode_category_major(uint32_t cp);

/* Returns the upper case variant of codepoint.
 * If there is none, it returns the same codepoint.
 */
uint32_t unicode_upper(uint32_t cp);

/* Returns the lower case variant of codepoint.
 * If there is none, it returns the same codepoint.
 */
uint32_t unicode_lower(uint32_t cp);
