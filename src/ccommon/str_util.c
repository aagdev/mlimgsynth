/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#include "str_util.h"
#include <stdarg.h>

int sprintf_alloc(char** buffer, const char* fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	int sz = vsnprintf(0, 0, fmt, ap);
	va_end(ap);

	if (sz < 0) return sz;
	sz += 1;
	char* p = realloc(*buffer, sz);
	if (!p) return -1;
	*buffer = p;

	va_start(ap, fmt);
	sz = vsnprintf(p, sz, fmt, ap);
	va_end(ap);

	return sz;
}

size_t string_escape_encode(char* out, size_t out_size,
	const char* in, size_t in_size, size_t* in_done)
{
	if (!in || !out) return 0;
	if (out_size < 5) return 0;	// 4=\xNN + 1=zero-end
	char *o=out, *oend=out+out_size-5;
	const char *i=in, *iend=in+in_size;
	for (; i<iend && o<oend; ++i) {
		     if (*i == '"' ) { *o++ = '\\'; *o++ = '"'; }
		else if (*i == '\n') { *o++ = '\\'; *o++ = 'n'; }
		else if (*i == '\r') { *o++ = '\\'; *o++ = 'r'; }
		else if (*i == '\t') { *o++ = '\\'; *o++ = 't'; }
		else if (32 <= *i && *i < 127) *o++ = *i;
		else
			o += sprintf(o, "\\x%02x", (unsigned)(unsigned char)*i);
	}
	*o = 0;
	if (in_done) *in_done = (i - in);
	return (o - out);
}

size_t string_escape_decode(char* out, size_t out_size,
	const char* in, size_t in_size, size_t* in_done)
{
	if (!in || !out) return 0;
	if (out_size < 1) return 0;
	char *o=out, *oend=out+out_size-1;
	const char *i=in, *iend=in+in_size;
	for (; i<iend && o<oend; ++i) {
		if (*i == '\\') {
			++i;
			if (i >= iend) {
				*o++ = '\\';
				break;
			}
			switch (*i) {
			case '"': *o++ = '"'; break;
			case 'n': *o++ = '\n'; break;
			case 'r': *o++ = '\r'; break;
			case 't': *o++ = '\t'; break;
			case 'x':
				if (i+2 < iend) {
					*o++ = digit_decode(*(i+1), 16) * 16
						+ digit_decode(*(i+2), 16);
					i += 2;
				}
				else {
					*o++ = '\\';
					--i;
				}
				break;
			//TODO: more...
			default:
				*o++ = '\\';
				--i;
				break;
			}
		}
		else *o++ = *i;
	}
	*o = 0;
	if (in_done) *in_done = (i - in);
	return (o - out);
}

