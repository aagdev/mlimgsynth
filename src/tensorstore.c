/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 */
#include "tensorstore.h"
#include "ccommon/logging.h"
#include "ccommon/vector.h"
#include "ccommon/bisect.h"
#include <inttypes.h>

#ifndef TENSORSTORE_ALLOCATOR
#define TENSORSTORE_ALLOCATOR  g_allocator
#endif

#define id_fromz(X)  strsto_add(S->ss, strsl_fromz(X))
#define id_str(X)  strsto_get(S->ss, X).b

/* Data conversion */

#ifdef TENSORSTORE_USE_GGML
#include "ggml.h"

#define tstore_fp32_from_fp16(D, S, N) \
	ggml_fp16_to_fp32_row((S), (D), (N))

#define tstore_fp16_from_fp32(D, S, N) \
	ggml_fp32_to_fp16_row((S), (D), (N))

#else
enum {
	GGML_TYPE_F64=28, GGML_TYPE_F32=0, GGML_TYPE_F16=1,
	GGML_TYPE_I64=27, GGML_TYPE_I32=26, GGML_TYPE_I16=25, GGML_TYPE_I8=24,
};

static inline
void tstore_fp32_from_fp16(float* dst, const void* src, size_t n) {
	for (size_t i=0; i<n; ++i) dst[i] = ((_Float16*)src)[i];
}

static inline
void tstore_fp16_from_fp32(void* dst, const float* src, size_t n) {
	for (size_t i=0; i<n; ++i) ((_Float16*)dst)[i] = src[i];
}
#endif

static inline
void tstore_fp32_from_fp64(float* dst, const double* src, size_t n) {
	for (size_t i=0; i<n; ++i) dst[i] = src[i];
}

/* Data types */

const char * g_tstore_dtype_str[TS_DTYPE__END] = {
	"none", "F64", "F32", "F16", "I64", "I32", "I16", "I8"
};

size_t g_tstore_dtype_size[TS_DTYPE__END] = {
	0, 8, 4, 2, 8, 4, 2, 1
};

int g_tstore_dtype_ggml_types[TS_DTYPE__END] = {
	-1,
	GGML_TYPE_F64, GGML_TYPE_F32, GGML_TYPE_F16,
	GGML_TYPE_I64, GGML_TYPE_I32, GGML_TYPE_I16, GGML_TYPE_I8,
};

int tstore_dtype_fromz(const char* s) {
	for (unsigned i=1; i<TS_DTYPE__END; ++i)
		if (!strcmp(s, g_tstore_dtype_str[i])) return i;
	return -1;
}

const char * tstore_dtype_str(int i) {
	return (0 <= i && i < TS_DTYPE__END) ? g_tstore_dtype_str[i] : "???";
}

size_t tstore_dtype_size(int i) {
	return (0 <= i && i < TS_DTYPE__END) ? g_tstore_dtype_size[i] : 0;
}

int tstore_dtype_from_ggml(int t)
{
	for (unsigned i=1; i<TS_DTYPE__END; ++i)
		if (g_tstore_dtype_ggml_types[i] == t) return i;
	return -1;
}

int tstore_dtype_to_ggml(int i)
{
	return (0 <= i && i < TS_DTYPE__END) ? g_tstore_dtype_ggml_types[i] : -1;
}

/* Tensor data */

void tstore_tdata_free(TSTensorData* S)
{
	if (S->ownmem) {
		alloc_free(TENSORSTORE_ALLOCATOR, (void*) S->data);
		S->data = NULL;
	}
}

/* Tensor entry */

void tstore_tensor_free(TSTensorEntry* S)
{
	vec_for(S->cache,i,0)
		tstore_tdata_free(&S->cache[i]);	
	vec_free(S->cache);
}

uint64_t tstore_tensor_count(const TSTensorEntry* S)
{
	uint64_t size = 1;
	for (unsigned i=0; i<S->shape_n; ++i) size *= S->shape[i];
	return size;	
}

uint64_t tstore_tensor_size(const TSTensorEntry* S)
{
	uint64_t size = tstore_dtype_size(S->dtype);
	for (unsigned i=0; i<S->shape_n; ++i) size *= S->shape[i];
	return size;	
}

int tstore_tensor_data_get(TSTensorEntry* S, TSDType dtype, int flags, 
	TSTensorData* out)
{
	int R=1;
	bool f_perm = flags & TSTDG_F_PERM;
		
	BISECT_RIGHT_DECL(found, idx, 0, vec_count(S->cache),
		S->cache[i_].dtype - dtype);
	if (found) {
		*out = S->cache[idx];
		return 1;
	}

	TRY_LOG( stream_seek(S->stm, S->offset, 0), "seek to %"PRIu64, S->offset );
	if (stream_read_prep(S->stm, S->size) < S->size)
		ERROR_LOG(-1, "read %"PRIu64" bytes", S->size);
	const void *cur = stream_buffer_get(S->stm, NULL);

	if (dtype == S->dtype) {  //direct
		if (stream_mmap_is(S->stm)) {
			*out = (TSTensorData){ dtype, cur, S->size, .perm=true };
		}
		else if (f_perm) {
			size_t sz = S->size;
			void *data = alloc_alloc(TENSORSTORE_ALLOCATOR, sz);
			memcpy(data, cur, sz);
			*out = (TSTensorData){ dtype, data, sz, .ownmem=true, .perm=true };
		}
		else {
			*out = (TSTensorData){ dtype, cur, S->size };
		}
	}
	// Data type conversion
	else if (dtype == TS_DTYPE_F32 && S->dtype == TS_DTYPE_F16)
	{
		size_t n = tstore_tensor_count(S), sz=n*4;
		void *data = alloc_alloc(TENSORSTORE_ALLOCATOR, sz);
		tstore_fp32_from_fp16(data, cur, n);
		*out = (TSTensorData){ dtype, data, sz, .ownmem=true, .perm=true };
		R = TSTDG_R_CONVERT;
	}
	else if (dtype == TS_DTYPE_F16 && S->dtype == TS_DTYPE_F32)
	{
		size_t n = tstore_tensor_count(S), sz=n*2;
		void *data = alloc_alloc(TENSORSTORE_ALLOCATOR, sz);
		tstore_fp16_from_fp32(data, cur, n);
		*out = (TSTensorData){ dtype, data, sz, .ownmem=true, .perm=true };
		R = TSTDG_R_CONVERT;
	}
	else if (dtype == TS_DTYPE_F32 && S->dtype == TS_DTYPE_F64)
	{
		size_t n = tstore_tensor_count(S), sz=n*4;
		float *data = alloc_alloc(TENSORSTORE_ALLOCATOR, sz);
		tstore_fp32_from_fp64(data, (double*)cur, n);
		*out = (TSTensorData){ dtype, data, sz, .ownmem=true, .perm=true };
		R = TSTDG_R_CONVERT;
	}
	else {
		ERROR_LOG(-1, "unsupported conversion from %s to %s",
			tstore_dtype_str(S->dtype), tstore_dtype_str(dtype));
	}
	
	if (f_perm)
		vec_insert(S->cache, idx, 1, out);
	
end:
	return R;
}

/* Tensor storage */

void tstore_free(TensorStore* S)
{
	vec_for(S->tensors,i,0) tstore_tensor_free(&S->tensors[i]);
	vec_free(S->tensors_idx);
	vec_free(S->tensors);
	vec_free(S->meta_idx);
	vec_free(S->meta);
}

void tstore_copy_from(TensorStore* dst, const TensorStore* src)
{
	vec_copyv(dst->meta, src->meta);
	vec_copyv(dst->meta_idx, src->meta_idx);
	vec_copyv(dst->tensors, src->tensors);
	vec_for(dst->tensors,i,0) dst->tensors[i].cache = NULL;
	vec_copyv(dst->tensors_idx, src->tensors_idx);
	dst->os_data = src->os_data;
	dst->os_end = src->os_end;
}

int tstore_info_dump(const TensorStore* S, Stream* stm)
{
	stream_printf(stm, "Metadata (%u):\n", vec_count(S->meta));
	vec_forp(TSMetaEntry,S->meta,e,0) {
		stream_printf(stm, "%s: %s\n", id_str(e->key), id_str(e->value));
	}
	stream_printf(stm, "Tensors (%u):\n", vec_count(S->tensors));
	vec_forp(TSTensorEntry,S->tensors,e,0) {
		stream_printf(stm, "%s: %s ", id_str(e->key), tstore_dtype_str(e->dtype));
		for (unsigned i=0; i<e->shape_n; ++i) {
			if (i) stream_char_put(stm, 'x');
			stream_printf(stm, "%u", e->shape[i]);
		}
		stream_printf(stm, " %"PRIu64" %"PRIu64"\n", e->offset, e->size);
	}
	return 1;
}

int tstore_info_dump_path(const TensorStore* S, const char* path)
{
	int R=1;
	Stream stm={0};
	TRY_LOG( stream_open_file(&stm, path, SOF_CREATE),
		"could not open '%s'", path);
	TRY( tstore_info_dump(S, &stm) );
end:
	stream_close(&stm, 0);
	return R;
}
		
#define INDEX_INSERT(VEC, KEY) do { \
	BISECT_RIGHT_DECL(found, ipos, 0, vec_count(S->VEC##_idx), \
		S->VEC[S->VEC##_idx[i_]].key - (KEY)); \
	if (found) { \
		log_debug("%s duplicate '%s'", #VEC, id_str(KEY)); \
	} else { \
		vec_insert(S->VEC##_idx, ipos, 1, NULL); \
	} \
	S->VEC##_idx[ipos] = vec_count(S->VEC)-1; \
} while (0)

void tstore_meta_add(TensorStore* S, const char* key, const char* value)
{
	TSMetaEntry e = { id_fromz(key), id_fromz(value) };
	vec_push(S->meta, e);
	INDEX_INSERT(meta, e.key);
}

void tstore_tensor_add(TensorStore* S, DynStr* name,
	const TSTensorEntry* e)
{
	if (S->cb_add) {
		int r = S->cb_add(S->cb_user, name);
		if (r) return;  //ignore
	}
	int key = id_fromz(*name);
	vec_push(S->tensors, *e);
	vec_last(S->tensors,0).key = key;
	INDEX_INSERT(tensors, key);
}

const char* tstore_meta_get(const TensorStore* S, const char* keystr)
{
	StringInt key = id_fromz(keystr);
	BISECT_RIGHT_DECL(found, ipos, 0, vec_count(S->meta_idx),
		S->meta[S->meta_idx[i_]].key - key);
	TSMetaEntry* e = &S->meta[S->meta_idx[ipos]];
	return found ? id_str(e->value) : NULL;
}

TSTensorEntry* tstore_tensor_getk(const TensorStore* S, StringInt key)
{
	BISECT_RIGHT_DECL(found, ipos, 0, vec_count(S->tensors_idx),
		S->tensors[S->tensors_idx[i_]].key - key);
	return found ? &S->tensors[S->tensors_idx[ipos]] : NULL;
}
