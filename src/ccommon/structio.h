/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 *
 * Structured data I/O.
 *
 * Example:
Stream stm={0};
StioStream sio={0};
char buffer[128];

TRY( stream_open_file(&stm, "test.json", SOF_CREATE|SOF_READ) );
TRY( stio_init(&sio, &stm, &stio_class_json, 0, sizeof(buffer), buffer) );
StioCheckpoint cp = stio_checkpoint(&sio);
TRY( stio_write_value(&sio, &any_map_indef()) );
TRY( stio_write_key(&sio, &any_stringz("pages")) );
TRY( stio_write_value(&sio, &any_int32(213)) );
TRY( stio_write_end(&sio) );
TRY( stio_close_check(&sio) );

StioItem itm;
TRY( stio_checkpoint_restore(&sio, &cp) );
TRY( stio_read(&sio, &itm, 0) );
TRY( stio_item_open_check(&itm, STIO_T_VALUE, ANY_T_MAP) );
TRY( stio_read(&sio, &itm, 0) );
TRY( stio_item_check(&itm, STIO_T_KEY, &any_stringz("pages")) );
TRY( stio_read(&sio, &itm, 0) );
TRY( stio_item_check(&itm, STIO_T_VALUE, &any_int32(213)) );
TRY( stio_read(&sio, &itm, 0) );
TRY( stio_item_check(&itm, STIO_T_END, NULL) );
TRY( stio_close_check(&sio) );
 */
#pragma once
#include "ccommon.h"
#include "any.h"
#include "stream.h"

//TODO: different mechanism to trigger the type casting (now: itm->type = STIO_T_CHUNK)
//TODO: class detection from file name and from file data (a la imgio)
//TODO: move code to .c ?

enum StioResult {
	STIO_R_OK			= 2,
	STIO_R_CTX_BEGIN	= 1,
	STIO_R_CTX_END		= 0,
	STIO_E_UNKNOWN		= -0x201,
	STIO_E_NOT_IMPL 	= -0x202,
	STIO_E_READ			= -0x203,
	STIO_E_WRITE		= -0x204,
	STIO_E_SEEK			= -0x205,
	STIO_E_EOF			= -0x206,
	STIO_E_DATA			= -0x207,
	STIO_E_TYPE			= -0x208,
	STIO_E_VALUE		= -0x209,
	STIO_E_CONTEXT		= -0x20a,
	STIO_E_OVERFLOW		= -0x20b,
	STIO_E_UNDERFLOW	= -0x20c,
	STIO_E_NESTING		= -0x20d,
	STIO_E_NEED_CHUNK	= -0x20e,
	STIO_E_MEM			= -0x20f,
	STIO_E_OPEN			= -0x210,
	STIO_E_UNEXPECTED	= -0x211, //user, unexpected item read
	STIO_E_CHECKPOINT	= -0x212,
	STIO_E_INCOMPLETE	= -0x213,
};

enum StioStreamFlag {
	STIO_SF_NO_BYTE_SWAP		= 0x0001,
	STIO_SF_CUSTOM				= 0x0100,
};

enum StioCtxFlag {
	STIO_IF_KEY_DONE			= 0x0001,
	STIO_IF_FIRST_DONE			= 0x0002,
	STIO_IF_DIRECT_ACCESS		= 0x0004,
	STIO_IF_DIRECT_BYTE_SWAP	= 0x0008,

	STIO_IF_FREE_OBJ			= 0x0010, //this ctx memory should be free'd

	STIO_IF_CUSTOM				= 0x0100,
};

// stio_read flags
enum StioReadFlag {
	// Do not read data even if available, just return the length
	STIO_RF_NO_DATA				= 0x0001,
	// Keep itm as is. Will use the buffer in itm->value to store the data.
	STIO_RF_PASS_ITEM			= 0x0002,
};

typedef enum StioItemType {
	STIO_T_NULL = 0,  //no-op or padding
	STIO_T_VALUE,
	STIO_T_KEY,
	STIO_T_TAG,
	STIO_T_CHUNK,
	STIO_T_END,
	STIO_T_LAST = STIO_T_END
} StioItemType;

typedef struct StioCtx StioCtx;
typedef struct StioItem StioItem;
typedef struct StioClass StioClass;
typedef struct StioStream StioStream;
typedef struct StioCheckpoint StioCheckpoint;

struct StioClass {
	int (*read)(StioStream*, StioCtx*, StioItem*);
	int (*write)(StioStream*, StioCtx*, StioItem*);
	const char* name;
};

struct StioCtx {
	AnyBaseType			vtype;		// value type (map, array)
	uint32_t			npend;		// number of array elements pending
	uint32_t			sflags;		// internal state flags
};

struct StioItem {
	StioItemType		type;
	uint32_t			npend;		// number of array elements pending
	Any					value;
};

struct StioStream {
	const StioClass *	cls;
	Stream *			s;
	StioCtx	*			ctx;
	void 				*buffer, *buf_end;
	uint32_t			cflags;  //config flags
	uint32_t			itm_sflags;  //tmp, new ctx sflags
};

struct StioCheckpoint {
	StioCtx				ctx;
	uint64_t			stream_pos;
	unsigned			nstack;
};

#define STIO_LENGTH_INDEF  ((uint32_t)(-1))

/* Interface */

static inline
bool stio_good(const StioStream* S) {
	return S && S->cls && S->s;
}

// Check if nothing is pending and the stream can be closed.
static inline
int stio_close_check(const StioStream* S) {
	return !S->ctx ? 0 : STIO_E_INCOMPLETE;
}

static inline
int stio_init(StioStream* S, Stream* s, const StioClass* cls, int cflags,
	size_t bufsz, void* buffer)
{
	*S = (StioStream){ .s=s, .cls=cls, .cflags=cflags,
		.buffer=buffer, .buf_end=(char*)buffer+bufsz };
	//TODO: check buffer alignment?
	return 0;
}

static inline
void stio_clear(StioStream* S) {
	*S = (StioStream){0};
}

// Write
int stio_write(StioStream* S, StioItem* itm);

static inline
int stio_write_value(StioStream* S, const Any* v) {
	StioItem itm = { .type=STIO_T_VALUE, .value=*v };
	return stio_write(S, &itm);
}

static inline
int stio_write_key(StioStream* S, const Any* v) {
	StioItem itm = { .type=STIO_T_KEY, .value=*v };
	return stio_write(S, &itm);
}

static inline
int stio_write_chunk(StioStream* S, const Any* v) {
	StioItem itm = { .type=STIO_T_CHUNK, .value=*v };
	return stio_write(S, &itm);
}

static inline
int stio_write_end(StioStream* S) {
	StioItem itm = { .type = STIO_T_END };
	return stio_write(S, &itm);
}

// Read
int stio_read(StioStream* S, StioItem* itm, int flags);

static inline
int stio_read_chunk(StioStream* S, StioItem* itm,
	AnyBaseType type, uint32_t n, void* dst)
{
	*itm = (StioItem){ .type=STIO_T_CHUNK,
		.value={ .t=anyb_pointer_get(type), .len=n, .p={.p=dst} } };
	return stio_read(S, itm, STIO_RF_PASS_ITEM);
}

// Read a vector to buffer <dst> with a maximum of <n> elements of type <type>.
// Perform type casting if necesary.
// If the source has more than <n> elements, returns STIO_R_CTX_BEGIN.
// Use stio_read_chunk to read more.
int stio_read_vector(StioStream* S, StioItem* itm,
	AnyBaseType type, uint32_t n, void* dst);

// Read a vector with element type <type>, allocating memory as needed with allocator <al>.
// Reuses memory block pointed by <pdst>, if not NULL.
// The final block of memory will be in <*pdst>.
// The allocator must be able to return the block size.
int stio_read_dyn(StioStream* S, StioItem* itm, AnyBaseType type, void** pdst,
	Allocator* al);

//int stio_read_dyn_any(StioStream* S, StioItem* itm, int any_al);

static inline
int stio_item_check(const StioItem* itm, StioItemType type, const Any* value)
{
	if (itm->type != type) return STIO_E_TYPE;
	if (itm->npend && itm->value.t != ANY_T_ARRAY && itm->value.t != ANY_T_MAP)
		return STIO_E_INCOMPLETE;  //be careful, the item is incomplete
	if (value && !any_equal(&itm->value, value)) return STIO_E_VALUE;
	return 0;
}

static inline
int stio_item_type_check(const StioItem* itm, StioItemType type, AnyBaseType vtype) {
	if (itm->type != type) return STIO_E_TYPE;
	if (itm->npend && itm->value.t != ANY_T_ARRAY && itm->value.t != ANY_T_MAP)
		return STIO_E_INCOMPLETE;  //be careful, the item is incomplete
	if (vtype && itm->value.t != vtype) return STIO_E_VALUE;
	return 0;
}

static inline
int stio_item_open_check(const StioItem* itm, StioItemType type, AnyBaseType vtype) {
	if (itm->type != type) return STIO_E_TYPE;
	if (!itm->npend) return STIO_E_VALUE;
	if (vtype && itm->value.t != vtype) return STIO_E_VALUE;
	if (itm->value.len) return STIO_E_INCOMPLETE;  //be careful, partial data received
	return 0;
}

/*static inline
long stio_read_skip(StioStream* S, uint32_t n) {
	return stio_read_chunk(S, ANY_T_NULL, n, NULL);
}

static inline
long stio_write_skip(StioStream* S, uint32_t n) {
	return stio_write_chunk(S, ANY_T_NULL, n, NULL);
}

static inline
int stio_read_end(StioStream* S) {
	if (S->ctx->npend && S->ctx->npend != STIO_LENGTH_INDEF) {
		long r = stio_read_skip(S, S->ctx->npend);
		if (r < 0 || r == STIO_R_CTX_END) return r;
	}
	char buffer[128];  //TODO: ok?
	StioItem itm = {
		.type  = anyb_pointer_is(S->ctx->vtype) ? STIO_T_CHUNK : STIO_T_VALUE,
		.value = any_string(sizeof(buffer), buffer) };
	do {
		int r = stio_read(S, &itm, STIO_RF_PASS_ITEM);
		if (r < 0) return r;
		if (r == STIO_R_CTX_BEGIN) {
			r = stio_read_end(S);
			if (r < 0) return r;
		}
	} while (itm.type != STIO_T_END);
	return STIO_R_CTX_END;
}*/

static inline
uint64_t stio_stream_pos_get(const StioStream* S) {
	return stream_pos_get(S->s);
}

// Return the number of parent contexts stored
static inline
unsigned stio_stack_size(const StioStream* S) {
	return S->ctx ? S->ctx + 1 - (StioCtx*)S->buffer : 0;
}

static inline
StioCheckpoint stio_checkpoint(const StioStream* S)
{
	assert(!S->ctx || S->ctx >= (StioCtx*)S->buffer);
	return (StioCheckpoint){
		.stream_pos = stio_stream_pos_get(S),
		.ctx = S->ctx ? *S->ctx : (StioCtx){0},
		.nstack = stio_stack_size(S),
	};
}

int stio_checkpoint_restore(StioStream* S, const StioCheckpoint* cp);

/* Utility functions for class implementations. */
#ifdef STRUCTIO_IMPLEMENTATION
#include <string.h>
#include "byteswap.h"

#ifdef STRUCTIO_LOG_DEBUG
	#include "logging.h"
	#define DebugLog(...) log_info(__VA_ARGS__)
#else
	#define DebugLog(...) {}
#endif

static inline
void stio_copy_be(StioStream*restrict sio, unsigned sz, void*restrict dst,
	const void*restrict src)
{
	bool swap = !(sio->cflags & STIO_SF_NO_BYTE_SWAP) && !big_endian_is();
	if (swap) byteswap_swap(sz, dst, src);
	else byteswap_copy(sz, dst, src);
}

// Write
static inline
int stio_stream_write_be(StioStream*restrict sio, size_t sz,
	const void*restrict data)
{
	assert(sz <= 8);
	uint8_t buffer[8];
	stio_copy_be(sio, sz, buffer, data);
	return stream_write_chk(sio->s, sz, buffer);
}

static inline
int stio_stream_write_be16(StioStream*restrict sio, const void*restrict data)
{
	return stio_stream_write_be(sio, 2, data);
}

static inline
int stio_stream_write_be32(StioStream*restrict sio, const void*restrict data)
{
	return stio_stream_write_be(sio, 4, data);
}

static inline
int stio_stream_write_be64(StioStream*restrict sio, const void*restrict data)
{
	return stio_stream_write_be(sio, 8, data);
}

// Read
static inline
int stio_stream_read_check(StioStream*restrict sio, size_t sz,
	const void*restrict chk)
{
	TRYR( stream_read_prep(sio->s, sz) );
	const char *cur = stream_buffer_get(sio->s, NULL);
	if (memcmp(cur, chk, sz)) return STIO_E_DATA;
	stream_commit(sio->s, cur+sz);
	return 0;
}

static inline
int stio_stream_read_be(StioStream*restrict sio, size_t sz, void*restrict dst)
{
	assert(sz <= 8);
	char buffer[8];
	TRYR( stream_read_chk(sio->s, sz, buffer) );
	stio_copy_be(sio, sz, dst, buffer);
	return 0;
}

#define stio_stream_read_var_be(S, D) \
	stio_stream_read_be((S), sizeof(D), &(D))

// Write a vector of binary elements
// swap: perform byte-order correction
// May perform type casting.
int stio_write_chunk_vector(StioStream* sio, const Any* value,
	AnyBaseType ctx_type, bool swap);

// Read a vector of binary elements
// swap: perform byte-order correction
// May perform type casting.
int stio_read_chunk_vector(StioStream* sio, Any* value,
	AnyBaseType type, uint32_t len, bool swap);

#endif  //STRUCTIO_IMPLEMENTATION

// vim: noet ts=4 sw=4
