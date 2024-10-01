/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 */
#include "tensorstore_gguf.h"
#include "ccommon/logging.h"

#ifndef TENSORSTORE_ALLOCATOR
#define TENSORSTORE_ALLOCATOR  g_allocator
#endif

#define GGUF_MAGIC "GGUF"
//#define GGUF_VERSION 3
#define GGUF_ALIGNMENT 32

static
uint64_t gguf_align(uint64_t offset)
{
    return offset + (GGUF_ALIGNMENT - (offset % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT;
}

static const int g_gguf_to_any_type[] = {
	ANY_T_UINT8 , ANY_T_INT8 ,
	ANY_T_UINT16, ANY_T_INT16,
	ANY_T_UINT32, ANY_T_INT32,
	ANY_T_FLOAT32,
	ANY_T_BOOL,
	ANY_T_STRING,
	ANY_T_ARRAY,
	ANY_T_UINT64, ANY_T_INT64,
	ANY_T_FLOAT64,
};

static
int gguf_meta_type_to_any(uint32_t gguf_type)
{
	return gguf_type < COUNTOF(g_gguf_to_any_type) ?
		g_gguf_to_any_type[gguf_type] : 0;
}

static
int gguf_read_string(Stream* stm, Allocator* al, Any* out, uint64_t limit)
{
	uint64_t len;
	TRYR( stream_read_var(stm, len) );
	TRYRB(TS_E_OVERFLOW, len <= limit);  //sanity check
	char *p = alloc_alloc(al, len+1);
	TRYR( stream_read_chk(stm, len, p) );
	p[len] = 0;
	*out = any_string(len, p);
	return 1;
}

static
int gguf_read_key(Stream* stm, Allocator* al, StringStore* ss, const char** pname)
{
	Any key={0};
	TRYR( gguf_read_string(stm, al, &key, 256) );
	TRYRB( TS_E_FORMAT, key.len > 0 );
	*pname = key.p.cp;
	assert( al->ctx == &ss->al );
	return strsto_add2(ss, strsl_froma(key), -1, true);
}

static
int gguf_read_array(Stream* stm, Allocator* al, Any* out)
{
	int R=1;

	uint32_t type;
	TRY( stream_read_var(stm, type) );
	int atype = gguf_meta_type_to_any(type);
	if (!(atype > 0)) ERROR_LOG(TS_E_METADATA, "unknown metadata type %u", type);

	uint64_t len;
	TRY( stream_read_var(stm, len) );
	TRYB(TS_E_OVERFLOW, len <= 0xffffff);  //sanity check

	if (anyb_scalar_is(atype)) {
		size_t sz = anyb_size(atype) * len;
		void *p = alloc_alloc(al, sz);
		TRY( stream_read_chk(stm, sz, p) );
		*out = any_vector(atype, len, p);
	}
	else if (atype == ANY_T_STRING) {
		size_t sz = sizeof(Any) * len;
		Any *p = alloc_alloc(al, sz);
		for (uint64_t i=0; i<len; ++i) {
			TRY( gguf_read_string(stm, al, &p[i], 0xffff) );
		}
		*out = any_array(len, p);
	}
	else
		return TS_E_METADATA;

end:
	return R;
}

static
int gguf_read_meta(Stream* stm, Allocator* al, Any* value, const char* name)
{
	int R=1;

	uint32_t type;
	TRY( stream_read_var(stm, type) );
	int atype = gguf_meta_type_to_any(type);
	if (!(atype > 0)) ERROR_LOG(TS_E_METADATA, "unknown metadata type %u", type);

	if (anyb_scalar_is(atype)) {
		*value = (Any){ atype };
		TRY( stream_read_chk(stm, anyb_size(atype), &value->p) );
	}
	else if (atype == ANY_T_STRING) {
		TRY( gguf_read_string(stm, al, value, 0xffffff) );
	}
	else if (atype == ANY_T_ARRAY) {
		TRY( gguf_read_array(stm, al, value) );
	}
	else
		return TS_E_METADATA;
	
	//log_debug("gguf meta '%s' %s", name, anyb_name(atype));

end:
	if (R<0) log_error("gguf load metadata '%s': %x", name, -R);
	return R;
}

static
int gguf_read_tensor(Stream* stm, TSTensorEntry* entry, const char* name)
{
	int R=1;

	uint32_t n_dim;
	TRY( stream_read_var(stm, n_dim) );
	TRYB( TS_E_OVERFLOW, n_dim <= 4 );  //sanity check

	uint64_t dims[4]={1,1,1,1};
	TRY( stream_read_chk(stm, sizeof(*dims)*n_dim, dims) );
	TRYB( TS_E_OVERFLOW, dims[0] <= 0xffffff );
	TRYB( TS_E_OVERFLOW, dims[1] <= 0xffffff );
	TRYB( TS_E_OVERFLOW, dims[2] <= 0xffffff );
	TRYB( TS_E_OVERFLOW, dims[3] <= 0xffffff );

	uint32_t ggml_type;
	TRY( stream_read_var(stm, ggml_type) );
	int dtype = tstore_dtype_from_ggml(ggml_type);
	if (!(dtype > 0)) ERROR_LOG(TS_E_DTYPE, "unknown tensor type %u", ggml_type);

	uint64_t offset;
	TRY( stream_read_var(stm, offset) );

	// Store
	entry->dtype = dtype;
	entry->shape_n = n_dim;
	entry->shape[0] = dims[0];
	entry->shape[1] = dims[1];
	entry->shape[2] = dims[2];
	entry->shape[3] = dims[3];
	entry->offset = offset;  // needs to be updated
	entry->stm = stm;
	entry->size = tstore_tensor_size(entry);
	
	//log_debug("gguf tensor '%s' %s " TSTENSOR_SHAPE4_FMT,
	//	name, tstore_dtype_str(dtype), TSTENSOR_SHAPE4_UNPACK(*entry));

end:
	if (R<0) log_error("gguf load tensor '%s': %x", name, -R);
	return R;
}

int tstore_read_gguf(TensorStore* S, Stream* stm)
{
	int R=1, key;
	const char* name=NULL;
	Allocator al = allocator_arena(&S->ss->al);
	
	// Header
	uint32_t magic;
	if (stream_read_var(stm, magic) < 0)
		ERROR_LOG(TS_E_READ, "could not read" );
	if (memcmp(&magic, GGUF_MAGIC, 4))
		ERROR_LOG(TS_E_FORMAT, "bad magic: %08xh", magic);
	
	uint32_t version;
	TRY( stream_read_var(stm, version) );
	if (version != 2 && version != 3)
		ERROR_LOG(TS_E_FORMAT, "unsupported version: %u", version);

	uint64_t n_tensor, n_meta;
	TRY( stream_read_var(stm, n_tensor) );
	TRY( stream_read_var(stm, n_meta) );
	TRYB(TS_E_OVERFLOW, n_tensor <= 65535);  //sanity check
	TRYB(TS_E_OVERFLOW, n_meta   <= 65535);  //sanity check

	log_debug("gguf n_meta:%u n_tensor:%u",
		(unsigned)n_meta, (unsigned)n_tensor);

	// Reserve memory
	vec_realloc(S->meta, vec_count(S->meta) + n_meta);
	vec_realloc(S->meta_idx, vec_count(S->meta_idx) + n_meta);
	vec_realloc(S->tensors, vec_count(S->tensors) + n_tensor);
	vec_realloc(S->tensors_idx, vec_count(S->tensors_idx) + n_tensor);

	// Metadata
	for (uint64_t i=0; i<n_meta; ++i) {
		TRY( key = gguf_read_key(stm, &al, S->ss, &name) );
		Any value={0};
		TRY( gguf_read_meta(stm, &al, &value, name) );
		TRY( tstore_meta_addk(S, key, &value) );
	}
	
	// Tensors
	for (uint64_t i=0; i<n_tensor; ++i) {
		TRY( key = gguf_read_key(stm, &al, S->ss, &name) );
		TSTensorEntry e={0};
		TRY( gguf_read_tensor(stm, &e, name) );
		TRY( tstore_tensor_addk(S, key, &e) );
	}

	uint64_t offset = stream_pos_get(stm);
	offset = gguf_align(offset);

	// Make tensors offsets absolute
	vec_for(S->tensors, i, 0) {
		if (S->tensors[i].stm != stm) continue;
		S->tensors[i].offset += offset;
	}

end:
	if (R<0) log_error("gguf read: %x", -R);
	return R;
}

//TODO: write

int tstore_detect_gguf(Stream* stm)
{
	uint8_t *end, *cur = stream_read_buffer(stm, &end);
	if (!(end-cur >= 4)) return 0;
	return !memcmp(cur, GGUF_MAGIC, 4);
}

const TensorStoreFormat ts_cls_gguf = {
	"gguf", "gguf",
	tstore_detect_gguf,
	tstore_read_gguf,
};
