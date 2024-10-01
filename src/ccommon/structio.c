/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#include "structio.h"
#include "byteswap.h"

/* Internal functions */

static inline
void* stio__buf_ptr(const StioStream* S) {
	// Leave space for an additional context
	return S->ctx ? S->ctx + 2 : (StioCtx*)S->buffer + 1;
}

static inline
ptrdiff_t stio__buf_sz(const StioStream* S) {
	return (char*)S->buf_end - (char*)stio__buf_ptr(S);
}

static inline
int stio__ctx_new(StioStream* S, StioItem* itm) {
	if (S->ctx) {
		if (!(S->ctx + 1 <= (StioCtx*)S->buf_end)) return STIO_E_MEM;
		S->ctx++;
	} else {
		if (!((StioCtx*)S->buffer + 1 <= (StioCtx*)S->buf_end)) return STIO_E_MEM;
		S->ctx = (StioCtx*)S->buffer;
	}
	*S->ctx = (StioCtx){ .vtype=itm->value.t, .npend=itm->npend,
		.sflags=S->itm_sflags };
	S->itm_sflags = 0;
	return 0;
}

static inline
int stio__ctx_delete(StioStream* S) {
	if (S->ctx > (StioCtx*)S->buffer)
		S->ctx--;
	else if (S->ctx == (StioCtx*)S->buffer)
		S->ctx = NULL; 
	else return STIO_E_MEM;
	return 0;
}

/* Interface */

int stio_write(StioStream* S, StioItem* itm)
{
	int r = S->cls->write(S, S->ctx, itm);
	if (r == STIO_R_CTX_BEGIN)
		TRYR( stio__ctx_new(S, itm) );
	else if (r == STIO_R_CTX_END)
		TRYR( stio__ctx_delete(S) );
	return r;
}

int stio_read(StioStream* S, StioItem* itm, int flags)
{
	if (!(flags & STIO_RF_PASS_ITEM)) {
		if (flags & STIO_RF_NO_DATA)
			*itm = (StioItem){0};
		else
			//TODO: use ANY_T_VOIDP ? see stio_read_chunk_vector
			*itm = (StioItem){ .value=any_string(
				stio__buf_sz(S), stio__buf_ptr(S)) };
	}

	int r = S->cls->read(S, S->ctx, itm);

	if (r == STIO_R_CTX_BEGIN)
		TRYR( stio__ctx_new(S, itm) );
	else if (r == STIO_R_CTX_END)
		TRYR( stio__ctx_delete(S) );

	return r;
}

int stio_read_chunk(StioStream* S, StioItem* itm)
{
	*itm = (StioItem){ .type=STIO_T_CHUNK,
		.value={ .t=S->ctx->vtype, .len=stio__buf_sz(S), .p={.p=stio__buf_ptr(S)} } };
	return stio_read(S, itm, STIO_RF_PASS_ITEM);
}

int stio_read_vector(StioStream* S, StioItem* itm,
	AnyBaseType type, uint32_t n, void* dst)
{
	int r = stio_read(S, itm, STIO_RF_NO_DATA);
	int itype = itm->type;
	if (r == STIO_R_CTX_BEGIN) {
		r = stio_read_chunk_buf(S, itm, type, n, dst);
		if (r >= 0) itm->type = itype;  //TODO: check
		if (r == STIO_R_CTX_END) r = STIO_R_OK;
		else if (r == STIO_R_OK) r = STIO_R_CTX_BEGIN;
	}
	return r;
}

int stio_read_dyn(StioStream* S, StioItem* itm, AnyBaseType type, void** pdst,
	Allocator* al)
{
	int r = stio_read(S, itm, STIO_RF_NO_DATA);
	int itype = itm->type;
	if (r == STIO_R_CTX_BEGIN) {
		unsigned tsz = anyb_size(type);
		if (itm->npend != STIO_LENGTH_INDEF)
			*pdst = alloc_realloc(al, *pdst, itm->npend*tsz);
		else if (!*pdst) *pdst = alloc_alloc(al, 1024);
		unsigned cap = alloc_size(al, *pdst) / tsz;
		unsigned i=0;
		while (1) {
			TRYR( r = stio_read_chunk_buf(S, itm, type,
				cap-i, (char*)(*pdst)+tsz*i) );
			i += itm->value.len;
			if (r == STIO_R_CTX_END) break;
			cap *= 2;
			*pdst = alloc_realloc(al, *pdst, tsz*cap);
		}
		r = STIO_R_OK;
		itm->type = itype;  //TODO: check
		itm->value.len = i;
		itm->value.p.p = *pdst;
	}
	return r;
}

int stio_checkpoint_restore(StioStream* S, const StioCheckpoint* cp)
{
	if (cp->nstack > stio_stack_size(S)+1) return STIO_E_CHECKPOINT;
	if (stream_seek(S->s, cp->stream_pos, 0) < 0) return STIO_E_SEEK;
	if (cp->nstack > 0) {
		assert(S->ctx + cp->nstack <= (StioCtx*)S->buf_end);
		S->ctx = (StioCtx*)S->buffer + cp->nstack - 1;
		*S->ctx = cp->ctx;
	} else {
		S->ctx = NULL;
	}
	return 1;
}

/* Internal functions for class implementations */

int stio_write_chunk_vector(StioStream* sio, const Any* value,
	AnyBaseType ctx_type, bool swap)
{
	//DebugLog("stio_write_chunk_vector");
	assert( anyb_pointer_is(value->t) );
	assert( value->p.p != NULL );
	assert( value->len != ANYT_LENGTH_INDEF );

	AnyBaseType stype = anyb_pointer_deref(value->t);
	const size_t sstep = anyb_size(stype);

	if (anyb_pointer_is(ctx_type))
	{
		AnyBaseType dtype = anyb_pointer_deref(ctx_type);
		const size_t dstep = anyb_size(dtype);
		assert(dstep > 0);

		if (dtype == stype)
		{
			if (dstep == 1)
			{
				size_t n = stream_write(sio->s, value->len, value->p.p);
				if (n != value->len) return STIO_E_WRITE;
			}
			else if (!swap)
			{	// This x5 faster for large vectors
				size_t n = stream_write(sio->s, value->len*sstep, value->p.p);
				if (n != value->len*sstep) return STIO_E_WRITE;
			}
			else
			{
				const unsigned char* ac = value->p.p;
				const unsigned char* ae = ac + value->len*sstep;
				while (ac < ae) {
					long rr = stream_write_prep(sio->s, (ae-ac)*sstep);
					if (rr < sstep) return STIO_E_WRITE;

					unsigned char *end, *cur = stream_buffer_get(sio->s, &end);

					while (cur+sstep < end && ac < ae) {
						memcpy(cur, ac, sstep);
						byteswap(sstep, cur);
						cur += sstep;
						ac += sstep;
					}

					stream_commit(sio->s, cur);
				}
			}
		}
		else
		{
			AnyPayload p;
			const unsigned char* ac = value->p.p;
			for (uint32_t i=0, e=value->len; i<e; ++i)
			{
				anyp_cast(dtype, &p, stype, ac);
				if (swap) byteswap(dstep, &p);
				if (stream_write(sio->s, dstep, &p) != dstep)
					return STIO_E_WRITE;
				ac += sstep;
			}
		}
	}
	else
		return STIO_E_CONTEXT;

	return STIO_R_OK;
}

int stio_read_chunk_vector(StioStream* sio, Any* value,
	AnyBaseType type, uint32_t len, bool swap)
{
	//DebugLog("stio_read_chunk_vector");
	assert( anyb_pointer_is(type) );

	AnyBaseType stype = anyb_pointer_deref(type);
	const size_t sstep = anyb_size(stype);

	if (value->t == ANY_T_VOIDP) {
		value->t = type;
		value->len /= sstep;
	}
	//TODO: value->t size < sstep ???

	if (!anyb_pointer_is(value->t))
		return STIO_E_VALUE;

	if (value->len > len)
		value->len = len;

	AnyBaseType dtype = anyb_pointer_deref(value->t);
	const size_t dstep = anyb_size(dtype);

	if (!value->p.p)
	{
		if (stream_seek(sio->s, value->len*sstep, SEEK_CUR) < 0)
			return STIO_E_SEEK;
	}
	else if (dtype == stype)
	{
		if (dstep == 1)
		{
			size_t n = stream_read(sio->s, value->len, value->p.p);
			if (n != value->len) return STIO_E_READ;
		}
		else if (!swap)
		{	// This x5 faster for large vectors
			size_t n = stream_read(sio->s, value->len*dstep, value->p.p);
			if (n != value->len*dstep) return STIO_E_WRITE;
		}
		else
		{
			unsigned char* ac = value->p.p;
			unsigned char* ae = ac + value->len*sstep;
			while (ac < ae) {
				long rr = stream_read_prep(sio->s, (ae-ac)*sstep);
				if (rr < sstep) return STIO_E_WRITE;

				unsigned char *end, *cur = stream_buffer_get(sio->s, &end);

				while (cur+sstep < end && ac < ae) {
					memcpy(ac, cur, sstep);
					byteswap(sstep, ac);
					cur += sstep;
					ac += sstep;
				}

				stream_commit(sio->s, cur);
			}
		}
	}
	else
	{
		AnyPayload p;
		unsigned char* ac = value->p.p;
		for (uint32_t i=0, e=value->len; i<e; ++i)
		{
			stream_read(sio->s, sstep, &p);
			if (swap) byteswap(sstep, ac);
			anyp_cast(stype, ac, dtype, &p);
			ac += dstep;
		}
	}

	return STIO_R_OK;
}
