/* Copyright 2024-2025, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#include "unicode.h"

uint32_t utf8_decode_next(const char** pstr, const char* end)
{
	const uint8_t *c = (const uint8_t*)*pstr,
	              *e = (const uint8_t*)end;
	if (!(c < e)) return 0;
	
	uint32_t cp = *c++;

	if ((cp & 0x80) == 0x80) {  //multibyte
		const uint8_t *b=c;
		while (c < e && (*c & 0xC0) == 0x80) ++c;  //count continuation bytes

		if ((cp & 0xE0) == 0xC0) {  //2 bytes: 110xxxxx 10xxxxxx
			if (c != b+1) goto error_end;
			uint32_t b2 = b[0];
			cp = ((cp & 0x1F) << 6) | (b2 & 0x3F);
		}
		else if ((cp & 0xF0) == 0xE0) {  //3 bytes: 1110xxxx ...
			if (c != b+2) goto error_end;
			uint32_t b2 = b[0], b3 = b[1];
			cp = ((cp & 0x0F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F);
		}
		else if ((cp & 0xF8) == 0xF0) {  //4 bytes: 11110xxx ...  
			if (c != b+3) goto error_end;
			uint32_t b2 = b[0], b3 = b[1], b4 = b[2];
			cp = ((cp & 0x07) << 18) | ((b2 & 0x3F) << 12) | ((b3 & 0x3F) << 6)
				| (b4 & 0x3F);
		}
		else goto error_end;
	}

	if ((const char*)c > end) { c=end; cp=0; }  //TODO: check before!
	*pstr = (const char*)c;
	return cp;
	
error_end:
	*pstr = (const char*)c;
	return 0xFFFD;
}

const char* utf8_decode_skip(const char* cur, const char* end)
{
	if (cur < end) cur++;  //first byte
	while (cur < end && (*cur & 0xC0) == 0x80) cur++;
	return cur;
}

char* utf8_encode_next(char* dst, uint32_t cp)
{
	if (cp <= 0x7F) {
		*dst++ = cp;
	}
	else if (cp <= 0x7FF) {
		*dst++ = 0xC0 | (cp >> 6);
		*dst++ = 0x80 | (cp & 0x3F);
	}
	else if (cp <= 0xFFFF) {
		*dst++ = 0xE0 | (cp >> 12);
		*dst++ = 0x80 | ((cp >> 6) & 0x3F);
		*dst++ = 0x80 | (cp & 0x3F);
	}
	else if (cp <= 0x10FFFF) {
		*dst++ = 0xF0 | (cp >> 18);
		*dst++ = 0x80 | ((cp >> 12) & 0x3F);
		*dst++ = 0x80 | ((cp >> 6) & 0x3F);
		*dst++ = 0x80 | (cp & 0x3F);
	}
	//else error, do nothing

	return dst;
}
