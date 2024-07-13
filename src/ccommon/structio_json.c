/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#define STRUCTIO_IMPLEMENTATION
#include "structio_json.h"
#include <stdlib.h>
#include <math.h>

static inline
int stio_stream_pretty_char_put(StioStream* sio, int c)
{
	return (sio->cflags & STIO_SF_JSON_PRETTY) ? stream_char_put(sio->s, c) : 0;
}

static inline
int stio_json_comma_write(StioStream* sio)
{
	TRYR( stream_char_put(sio->s, ',') ); \
	TRYR( stio_stream_pretty_char_put(sio, ' ') ); \
	return 0;
}

static inline
int stio_json_colon_write(StioStream* sio)
{
	TRYR( stream_char_put(sio->s, ':') ); \
	TRYR( stio_stream_pretty_char_put(sio, ' ') ); \
	return 0;
}

static inline
int stream_space_skip_(Stream* S) {
	int c;
	do {
		TRYR( c = stream_char_get(S) );
	} while (c == ' ' || c == '\t' || c == '\r' || c == '\n');
	stream_unget(S, 1);
	return 0;
}

/*
	JSON encoder
*/
static inline
int stio_json_write_value(StioStream* sio, StioItem* itm);

static inline
int stio_json_write_close(StioStream* sio, StioCtx* ctx)
{
	char c=0;
	if (ctx->vtype == ANY_T_STRING) c = '"';
	else if (ctx->vtype == ANY_T_MAP) c = '}';
	else if (ctx->vtype == ANY_T_ARRAY) c = ']';
	else if (anyb_pointer_is(ctx->vtype)) c = ']';
	if (c) TRYR( stream_char_put(sio->s, c) );
	if (c && c != '"') TRYR( stio_stream_pretty_char_put(sio, '\n') );

	ctx->npend = 0;
	return STIO_R_OK;
}

static inline
int stio_json_write_string(StioStream* sio, unsigned long n, const char* str)
{
	for (const char *c=str, *e=c+n; c<e; ++c)
	{
		switch (*c) {
		case '"':	TRYR( stream_write_chk(sio->s, 2, "\\\"") ); break;
		case '\\':	TRYR( stream_write_chk(sio->s, 2, "\\\\") ); break;
		case '\t':	TRYR( stream_write_chk(sio->s, 2, "\\t") ); break;
		case '\r':	TRYR( stream_write_chk(sio->s, 2, "\\r") ); break;
		case '\n':	TRYR( stream_write_chk(sio->s, 2, "\\n") ); break;
		//TODO: more escapes
		default:
			TRYR( stream_char_put(sio->s, *c) );
		}
	}
	return STIO_R_OK;
}

static inline
int stio_json_write_chunk(StioStream* sio, StioCtx* ctx, const Any* value)
{
	int r;
	if (!value->len) return STIO_R_OK;
	if (!value->p.p) return STIO_E_VALUE;
	//if (value->cls) return STIO_E_VALUE;

	if (ctx->vtype == ANY_T_STRING)
	{
		if (value->t != ctx->vtype) return STIO_E_VALUE;
		TRYR( stio_json_write_string(sio, value->len, value->p.cp) );
	}
	else if (ctx->vtype == ANY_T_MAP)
	{
		if (value->t != ctx->vtype) return STIO_E_VALUE;
		StioItem itm={0};
		const Any *ac = value->p.ap;
		for (uint32_t i=0, e=value->len; i<e; ++i)
		{
			if (ctx->sflags & STIO_IF_FIRST_DONE)
				TRYR( stio_json_comma_write(sio) );

			itm.type = STIO_T_KEY;
			if (ac) itm.value = *ac++;
			TRYR( r = stio_json_write_value(sio, &itm) );
			if (r == STIO_R_CTX_BEGIN) return STIO_E_NESTING; //incomplete

			TRYR( stio_json_colon_write(sio) );

			itm.type = STIO_T_VALUE;
			if (ac) itm.value = *ac++;
			TRYR( r = stio_json_write_value(sio, &itm) );
			if (r == STIO_R_CTX_BEGIN) return STIO_E_NESTING; //incomplete

			ctx->sflags |= STIO_IF_FIRST_DONE;
		}
	}
	else if (ctx->vtype == ANY_T_ARRAY || anyb_pointer_is(ctx->vtype))
	{
		if (value->t == ANY_T_ARRAY)
		{
			if (value->t != ctx->vtype) return STIO_E_VALUE;
			StioItem itm={0};
			const Any *ac = value->p.ap;
			for (uint32_t i=0, e=value->len; i<e; ++i)
			{
				if (ctx->sflags & STIO_IF_FIRST_DONE)
					TRYR( stio_json_comma_write(sio) );

				itm.type = STIO_T_VALUE;
				if (ac) itm.value = *ac++;
				TRYR( r = stio_json_write_value(sio, &itm) );
				if (r == STIO_R_CTX_BEGIN) return STIO_E_NESTING;

				ctx->sflags |= STIO_IF_FIRST_DONE;
			}
		}
		else if (anyb_pointer_is(value->t))
		{
			AnyBaseType vtype = anyb_pointer_deref(value->t);
			const size_t vstep = anyb_size(vtype);

			StioItem itm={0};
			const uint8_t *ac = value->p.p;
			for (uint32_t i=0, e=value->len; i<e; ++i)
			{
				if (ctx->sflags & STIO_IF_FIRST_DONE)
					TRYR( stio_json_comma_write(sio) );

				itm.type = STIO_T_VALUE;
				itm.value.t = vtype;
				if (ac && vstep) {
					memcpy(&itm.value.p, ac, vstep);
					ac += vstep;
				}
				TRYR( stio_json_write_value(sio, &itm) );

				ctx->sflags |= STIO_IF_FIRST_DONE;
			}
		}
		else return STIO_E_VALUE;
	}
	else return STIO_E_CONTEXT;

	return STIO_R_OK;
}

static inline
int stio_json_write_uint64(StioStream* sio, uint64_t n)
{
	uint64_t f = 1;
	while (f*10 <= n) f *= 10;
	while (f) {
		unsigned d = n / f;
		n = n % f;
		f /= 10;
		TRYR( stream_char_put(sio->s, '0'+d) );
	}
	return STIO_R_OK;
}

static inline
int stio_json_write_int64(StioStream* sio, int64_t n)
{
	if (n < 0) {
		TRYR( stream_char_put(sio->s, '-') );
		n = -n;
	}
	return stio_json_write_uint64(sio, n);
}

static inline
int stio_json_write_float64(StioStream* sio, double n, int digits)
{
	if (n < 0) {
		TRYR( stream_char_put(sio->s, '-') );
		n = -n;
	}

	switch(fpclassify(n)) { //TODO: not standard, add flag
	case FP_INFINITE:
		TRYR( stream_write_chk(sio->s, 8, "Infinity") );
		return STIO_R_OK;
	case FP_NAN:
		TRYR( stream_write_chk(sio->s, 3, "NaN") );
		return STIO_R_OK;
    case FP_NORMAL:
    	break;
    case FP_SUBNORMAL:
    	break;
    case FP_ZERO:
		TRYR( stream_char_put(sio->s, '0') );
		return STIO_R_OK;
    default:
		return STIO_E_VALUE;
	}

	TRYR( stream_write_prep(sio->s, 22) );

	uint8_t *cur = stream_buffer_get(sio->s, NULL);

	int exp = floor(log10(n));

	bool bexp;
	int nl, nr;
	if (-4 <= exp && exp <= 0) {
		bexp = false;
		nl = 1;
		nr = digits-1-exp;
		// round to the printed number of decimal digits
		n += 5 * pow(10, exp-digits) - 1e-21;
	} else {
		n *= pow(10, -exp);
		if (0 < exp && exp <= digits-1) {
			bexp = false;
			nl = exp+1;
			nr = digits-1-exp;
		} else {
			bexp = true;
			nl = 1;
			nr = digits-1;
		}
		// round to the printed number of decimal digits
		n += 5 * pow(10, -digits) - 1e-17;
	}

	// digits to left of the decimal dot
	for (int i=0; i<nl; ++i) {
		int d = n;
		n = (n - d)*10;
		*cur++ = '0'+d;
	}

	*cur++ = '.';

	// digits to right of the decimal dot
	if (nr) {
		unsigned nzero=0;
		for (int i=0; i<nr; ++i) {
			int d = n;
			if (!d) nzero++; else nzero=0;
			n = (n - d)*10;
			*cur++ = '0'+d;
		}
		if (nzero == nr) nzero--;
		cur -= nzero;
	}
	else {
		*cur++ = '0';
	}

	if (bexp) {
		// exponent
		*cur++ = 'e';
		stream_commit(sio->s, cur);
		return stio_json_write_int64(sio, exp);
	}
	else {
		stream_commit(sio->s, cur);
		return STIO_R_OK;
	}
}

static inline
int stio_json_write_open(StioStream* sio, StioItem* itm)
{
	if (itm->value.len == 0) {
		return stio_json_write_close(sio, &(StioCtx){ .vtype=itm->value.t });
	}
	else if (itm->value.p.p) {
		StioCtx ctx={ .vtype=itm->value.t, .npend=STIO_LENGTH_INDEF };
		TRYR( stio_json_write_chunk(sio, &ctx, &itm->value) );
		return stio_json_write_close(sio, &ctx);
	}
	
	itm->npend = STIO_LENGTH_INDEF;

	return STIO_R_CTX_BEGIN;
}

static inline
int stio_json_write_value(StioStream* sio, StioItem* itm)
{
	switch (itm->value.t) {
	case ANY_T_NULL:
		TRYR( stream_write_chk(sio->s, 4, "null") );
		return STIO_R_OK;

	case ANY_T_BOOL:
		if (itm->value.p.b) TRYR( stream_write_chk(sio->s, 4, "true") );
		else TRYR( stream_write_chk(sio->s, 5, "false") );
		return STIO_R_OK;

	case ANY_T_CHAR:
		TRYR( stream_char_put(sio->s, '"') );
		TRYR( stream_char_put(sio->s, itm->value.p.c) );
		TRYR( stream_char_put(sio->s, '"') );
		return STIO_R_OK;

	case ANY_T_UINT8:  return stio_json_write_uint64(sio, itm->value.p.u8);
	case ANY_T_UINT16: return stio_json_write_uint64(sio, itm->value.p.u16);
	case ANY_T_UINT32: return stio_json_write_uint64(sio, itm->value.p.u32);
	case ANY_T_UINT64: return stio_json_write_uint64(sio, itm->value.p.u64);

	case ANY_T_INT8:   return stio_json_write_int64(sio, itm->value.p.i8);
	case ANY_T_INT16:  return stio_json_write_int64(sio, itm->value.p.i16);
	case ANY_T_INT32:  return stio_json_write_int64(sio, itm->value.p.i32);
	case ANY_T_INT64:  return stio_json_write_int64(sio, itm->value.p.i64);

	case ANY_T_FLOAT32:
		return stio_json_write_float64(sio, itm->value.p.f32, 7);
	case ANY_T_FLOAT64:
		return stio_json_write_float64(sio, itm->value.p.f64, 16);

	case ANY_T_STRING:
		TRYR( stream_char_put(sio->s, '"') );
		return stio_json_write_open(sio, itm);

	case ANY_T_ARRAY:
		TRYR( stream_char_put(sio->s, '[') );
		//TRYR( stio_stream_pretty_char_put(sio, '\n') );
		return stio_json_write_open(sio, itm);

	case ANY_T_MAP:
		TRYR( stream_char_put(sio->s, '{') );
		//TRYR( stio_stream_pretty_char_put(sio, '\n') );
		return stio_json_write_open(sio, itm);


	default:
		if (anyb_pointer_is(itm->value.t))
		{
			TRYR( stream_char_put(sio->s, '[') );
			return stio_json_write_open(sio, itm);
		}
	}
	return STIO_E_VALUE;
}

int stio_json_write(StioStream* sio, StioCtx* ctx, StioItem* itm)
{
	int r=0;

	switch (itm->type) {
	case STIO_T_NULL:
		return STIO_R_OK;

	case STIO_T_END:
		if (!ctx) return STIO_E_CONTEXT;
		TRYR( stream_write_prep(sio->s, 1) );
		TRYR( r = stio_json_write_close(sio, ctx) );
		return STIO_R_CTX_END;

	case STIO_T_TAG:
		return STIO_E_TYPE;

	case STIO_T_KEY:
		if (!ctx) return STIO_E_CONTEXT;
		if (ctx->vtype != ANY_T_MAP) return STIO_E_CONTEXT;
		if (ctx->sflags & STIO_IF_KEY_DONE) return STIO_E_CONTEXT;

		if (ctx->sflags & STIO_IF_FIRST_DONE)
			TRYR( stio_json_comma_write(sio) );

		TRYR( stream_write_prep(sio->s, 1) );
		TRYR( r = stio_json_write_value(sio, itm) );

		TRYR( stio_json_colon_write(sio) );
		ctx->sflags |= STIO_IF_KEY_DONE;
		break;

	case STIO_T_VALUE:
		if (ctx) {
			if (ctx->vtype == ANY_T_MAP && ~ctx->sflags & STIO_IF_KEY_DONE)
				return STIO_E_CONTEXT;

			if (ctx->vtype != ANY_T_MAP && ctx->sflags & STIO_IF_FIRST_DONE)
				TRYR( stio_json_comma_write(sio) );
		}

		TRYR( stream_write_prep(sio->s, 1) );
		TRYR( r = stio_json_write_value(sio, itm) );

		if (ctx) {
			if (ctx->vtype == ANY_T_MAP) ctx->sflags &= ~STIO_IF_KEY_DONE;
			ctx->sflags |= STIO_IF_FIRST_DONE;
		}
		break;

	case STIO_T_CHUNK:
		if (!ctx) return STIO_E_CONTEXT;
		TRYR( stream_write_prep(sio->s, 1) );
		return stio_json_write_chunk(sio, ctx, &itm->value);

	default:
		return STIO_E_TYPE;
	}

	return r;
}

/*
	JSON decoder
*/
static inline
int stio_json_read_chunk(StioStream* sio, StioCtx* ctx, Any* value);

static inline
int stio_json_read_close(StioStream* sio, StioCtx* ctx, StioItem* itm)
{
	ctx->npend = 0;
	itm->type = STIO_T_END;
	return STIO_R_CTX_END;
}

static inline
int stio_json_read_open(StioStream* sio, StioItem* itm, AnyBaseType vtype)
{
	if (itm->value.p.p && itm->value.len > 0 &&
		(itm->value.t == ANY_T_VOIDP || itm->value.t == vtype) )
	{
		StioCtx ctx = { .vtype = vtype, .npend = STIO_LENGTH_INDEF };
		TRYR( stio_json_read_chunk(sio, &ctx, &itm->value) );
		if (ctx.npend)
		{
			itm->npend = STIO_LENGTH_INDEF;
			return STIO_R_CTX_BEGIN;
		}
		else
		{
			return STIO_R_OK;
		}
	}
	else
	{
		itm->value = (Any){ .t = vtype };
		itm->npend = STIO_LENGTH_INDEF;
		return STIO_R_CTX_BEGIN;
	}
}

int json_parse_string(const uint8_t** pcur, const uint8_t* end,
	char** pdst, char* dend)
{
	const uint8_t *cur = *pcur;
	char *dst = *pdst;
	for (; cur < end && *cur!='"' && dst<dend; ++cur) {
		if (*cur == '\\') {
			if (cur+1 >= end) break;
			cur++;
			switch (*cur) {
			case '\\':
			case '"':
				*dst++ = *cur;
				break;
			//TODO: more
			default:
				return STIO_E_DATA;
			}
		} else {
			*dst++ = *cur;
		}
	}
	*pcur = cur;
	*pdst = dst;
	return (cur<end && *cur == '"') ? STIO_R_CTX_END : STIO_R_OK;
}

static inline
int stio_json_read_string(StioStream* sio, uint32_t* pn, char* dst)
{
	unsigned n = *pn;
	if (n < 1) return STIO_R_OK;
	n--;  //zero terminator
	//TODO: move this to code common to all structio_

	char *dst0 = dst, *dend = dst+n;
	const uint8_t *end, *cur = stream_read_buffer(sio->s, &end);
	if (!cur) return STIO_E_READ;
	int r = json_parse_string(&cur, end, &dst, dend);
	stream_commit(sio->s, (void*)cur);
	*dst = 0;
	*pn = dst - dst0;
	return r;
}

long json_parse_number(const uint8_t* cur, const uint8_t* end, Any* value)
{
	const uint8_t* cur0 = cur;
	bool float_is = false;
	int sign = +1;
	intptr_t i=0;
	int e=0, e2=0;
	double f=0;

	if (cur < end) {
		if (*cur == '-') { sign = -1; cur++; }
		else if (*cur == '+') cur++;
	}

	if (!memcmp(cur, "NaN", 3)) {
		cur += 3;
		*value = any_float64(NAN * sign);
		return cur - cur0;
	}
	if (!memcmp(cur, "Infinity", 8)) {
		cur += 8;
		*value = any_float64(INFINITY * sign);
		return cur - cur0;
	}

	while (cur < end && '0' <= *cur && *cur <= '9') {
		i = i*10 + (*cur-'0');
		cur++;
	}

	if (cur < end) {
		if (*cur == '.') {
			cur++;
			float_is = true;
			f = i;
			const uint8_t *cur1 = cur;
			while (cur < end && '0' <= *cur && *cur <= '9') {
				f = f*10 + (*cur-'0');
				cur++;
			}
			e -= cur - cur1;
		}
		if (*cur == 'e' || *cur == 'E') {
			cur++;
			if (!float_is) {
				f = i;
				float_is = true;
			}
			int esign = +1;
			if (cur < end) {
				if (*cur == '-') { esign = -1; cur++; }
				else if (*cur == '+') cur++;
			}
			while (cur < end && '0' <= *cur && *cur <= '9') {
				e2 = e2*10 + (*cur-'0');
				cur++;
			}
			e += e2 * esign;
		}
	}

	if (float_is) {
		if (f <= 33554432.0 && e2 <= 38) {
			f *= pow(10, e) * sign;
			*value = any_float32(f);
		 } else {
		 	f *= pow(10, e) * sign;
			*value = any_float64(f);
		}
	} else if (i < 0x7fffffff) {
		*value = any_int32((int32_t)i * sign);
	} else {
		*value = any_int64((int64_t)i * sign);
	}

	return cur - cur0;
}

static inline
int stio_json_read_number(StioStream* sio, StioItem* itm)
{
	uint8_t *end, *cur = stream_read_buffer(sio->s, &end);
	if (!cur) return STIO_E_READ;
	int r = json_parse_number(cur, end, &itm->value);
	cur += r;
	stream_commit(sio->s, cur);
	return r ? STIO_R_OK : STIO_E_DATA;
}

static inline
int stio_json_read_value(StioStream* sio, StioItem* itm)
{
	int c;
	TRYR( stream_space_skip_(sio->s) );
	TRYR( c = stream_char_get(sio->s) );
	DebugLog("stio_json_read_value %c", c);
	switch (c) {
	case '"':
		return stio_json_read_open(sio, itm, ANY_T_STRING);
	case '[':
		return stio_json_read_open(sio, itm, ANY_T_ARRAY);
	case '{':
		return stio_json_read_open(sio, itm, ANY_T_MAP);
	case 't':
		TRYR( stio_stream_read_check(sio, 3, "rue") );
		itm->value = any_bool(true);
		return STIO_R_OK;
	case 'f':
		TRYR( stio_stream_read_check(sio, 4, "alse") );
		itm->value = any_bool(false);
		return STIO_R_OK;
	case 'n':
		TRYR( stio_stream_read_check(sio, 3, "ull") );
		itm->value = any_null();
		return STIO_R_OK;
	default:
		stream_unget(sio->s, 1);
		return stio_json_read_number(sio, itm);
	}
}

static inline
int stio_json_read_chunk(StioStream* sio, StioCtx* ctx, Any* value)
{
	int c;
	bool end = false;
	uint32_t i, e;

	if (!value->len) return STIO_R_OK;
	//if (!value->p.p) return STIO_E_VALUE;
	//if (value->cls) return STIO_E_VALUE;

	if (ctx->vtype == ANY_T_STRING) {
		if (!value->p.p) return STIO_E_VALUE;
		if (value->t != ctx->vtype) return STIO_E_VALUE; //TODO
		int r = stio_json_read_string(sio, &value->len, value->p.cp);
		if (r < 0) return r;
		if (r == STIO_R_CTX_END) {
			TRYR( c = stream_char_get(sio->s) );
			assert(c == '"');
			end=true;
		}
	}
	else if (ctx->vtype == ANY_T_MAP) {
		if (!value->p.p) return STIO_E_VALUE;
		if (value->t != ctx->vtype) return STIO_E_VALUE; //TODO
		Any *ac=value->p.ap;
		for (i=0, e=value->len; i<e; ++i)
		{
			TRYR( stream_space_skip_(sio->s) );
			TRYR( c = stream_char_get(sio->s) );
			if (c == '}') { end=true; break; }
			else if (~ctx->sflags & STIO_IF_FIRST_DONE)
				stream_unget(sio->s, 1);
			else if (c != ',') return STIO_E_DATA;

			StioItem itm = {0};
			int r = stio_json_read_value(sio, &itm);
			if (r < 0) return r;
			if (r == STIO_R_CTX_BEGIN) return STIO_E_NESTING;

			if (ac) *ac++ = itm.value;

			TRYR( stream_space_skip_(sio->s) );
			TRYR( c = stream_char_get(sio->s) );
			if (c != ':') return STIO_E_DATA;

			itm = (StioItem){0};
			r = stio_json_read_value(sio, &itm);
			if (r < 0) return r;
			if (r == STIO_R_CTX_BEGIN) return STIO_E_NESTING;

			if (ac) *ac++ = itm.value;

			ctx->sflags |= STIO_IF_FIRST_DONE;
		}
		value->len = i;
		
		if (!end) {  //auto end
			TRYR( stream_space_skip_(sio->s) );
			TRYR( c = stream_char_get(sio->s) );
			if (c == '}') end=true;
			else if (stream_unget(sio->s, 1) != 1) return STIO_E_READ;
		}
	}
	else if (ctx->vtype == ANY_T_ARRAY) {
		if (value->t == ANY_T_ARRAY) {
			if (!value->p.p) return STIO_E_VALUE;
			Any *ac=value->p.ap;
			for (i=0, e=value->len; i<e; ++i)
			{
				TRYR( stream_space_skip_(sio->s) );
				TRYR( c = stream_char_get(sio->s) );
				if (c == ']') { end=true; break; }
				else if (~ctx->sflags & STIO_IF_FIRST_DONE)
					stream_unget(sio->s, 1);
				else if (c != ',') return STIO_E_DATA;

				StioItem itm = {0};
				int r = stio_json_read_value(sio, &itm);
				if (r < 0) return r;
				if (r == STIO_R_CTX_BEGIN) return STIO_E_NESTING;

				if (ac) *ac++ = itm.value;

				ctx->sflags |= STIO_IF_FIRST_DONE;
			}
			value->len = i;
		}
		else if (anyb_pointer_is(value->t)) {
			AnyBaseType vtype = anyb_pointer_deref(value->t);
			const size_t vstep = anyb_size(vtype);
			uint8_t *ac = value->p.p;
			for (i=0, e=value->len; i<e; ++i)
			{
				TRYR( stream_space_skip_(sio->s) );
				TRYR( c = stream_char_get(sio->s) );
				if (c == ']') { end=true; break; }
				else if (~ctx->sflags & STIO_IF_FIRST_DONE)
					stream_unget(sio->s, 1);
				else if (c != ',') return STIO_E_DATA;

				StioItem itm = {0};
				int r = stio_json_read_value(sio, &itm);
				if (r < 0) return r;
				if (r == STIO_R_CTX_BEGIN) return STIO_E_NESTING;

				if (ac && vstep) {
					anyp_cast(vtype, ac, itm.value.t, &itm.value.p);
					ac += vstep;
				}

				ctx->sflags |= STIO_IF_FIRST_DONE;
			}
			value->len = i;
		}
		else return STIO_E_VALUE;
		
		if (!end) {  //auto end
			TRYR( stream_space_skip_(sio->s) );
			TRYR( c = stream_char_get(sio->s) );
			if (c == ']') end=true;
			else if (stream_unget(sio->s, 1) != 1) return STIO_E_READ;
		}
	}
	else return STIO_E_CONTEXT;

	if (end) {
		ctx->npend = 0;
		return STIO_R_CTX_END;
	}
	return STIO_R_OK;
}

int stio_json_read(StioStream* sio, StioCtx* ctx, StioItem* itm)
{
	int c;

	TRYR( stream_space_skip_(sio->s) );

	if (!ctx)
	{
		if (itm->type == STIO_T_CHUNK) return STIO_E_TYPE;
		TRYR( stream_read_prep(sio->s, 1) );
		itm->type = STIO_T_VALUE;
		return stio_json_read_value(sio, itm);
	}
	else if (ctx && ctx->npend == 0)
	{
		return stio_json_read_close(sio, ctx, itm);
	}

	TRYR( stream_read_prep(sio->s, 1) );
	if (itm->type == STIO_T_CHUNK)
	{
		return stio_json_read_chunk(sio, ctx, &itm->value);
	}
	else if (ctx->vtype == ANY_T_STRING)
	{
		TRYR( c = stream_char_get(sio->s) );
		if (c == '"') return stio_json_read_close(sio, ctx, itm);
		return STIO_E_NEED_CHUNK;
	}
	else if (ctx->vtype == ANY_T_ARRAY)
	{
		TRYR( stream_space_skip_(sio->s) );
		TRYR( c = stream_char_get(sio->s) );
		if (c == ']') return stio_json_read_close(sio, ctx, itm);
		else if (~ctx->sflags & STIO_IF_FIRST_DONE) stream_unget(sio->s, 1);
		else if (c != ',') return STIO_E_DATA;

		itm->type = STIO_T_VALUE;
		int r = stio_json_read_value(sio, itm);

		ctx->sflags |= STIO_IF_FIRST_DONE;
		return r;
	}
	else if (ctx->vtype == ANY_T_MAP) {
		if (ctx->sflags & STIO_IF_KEY_DONE)
		{
			TRYR( stream_space_skip_(sio->s) );
			TRYR( c = stream_char_get(sio->s) );
			if (c != ':') return STIO_E_DATA;

			itm->type = STIO_T_VALUE;
			int r = stio_json_read_value(sio, itm);
			if (r < 0) return r;
			ctx->sflags &= ~STIO_IF_KEY_DONE;
			return r;
		}
		else
		{
			TRYR( stream_space_skip_(sio->s) );
			TRYR( c = stream_char_get(sio->s) );
			if (c == '}') return stio_json_read_close(sio, ctx, itm);
			else if (~ctx->sflags & STIO_IF_FIRST_DONE) stream_unget(sio->s, 1);
			else if (c != ',') return STIO_E_DATA;

			itm->type = STIO_T_KEY;
			int r = stio_json_read_value(sio, itm);
			if (r < 0) return r;

			ctx->sflags |= STIO_IF_KEY_DONE | STIO_IF_FIRST_DONE;
			return r;
		}
	}
	return STIO_E_CONTEXT;
}

/*
*/
const StioClass stio_class_json = {
	stio_json_read,
	stio_json_write,
	"json"
};
