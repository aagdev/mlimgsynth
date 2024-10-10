/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 */
#include "tensorstore.h"
#include "ccommon/logging.h"
#include "ccommon/vector.h"
#include "ccommon/bisect.h"
#include <inttypes.h>

#ifdef TENSORSTORE_USE_GGML
#include "ggml.h"
#else
#include "hpc_ll.h"
#endif

#define ALIGNMENT_CHECK(PTR,BYTES) \
	((((intptr_t)(PTR)) & ((BYTES)-1)) == 0)

#ifndef TENSORSTORE_ALLOCATOR
#define TENSORSTORE_ALLOCATOR  g_allocator
#endif

#define id_fromz(X)  strsto_add(S->ss, strsl_fromz(X))
#define id_str(X)  strsto_get(S->ss, X).b

/* Data types */

#ifndef TENSORSTORE_USE_GGML
// Copied from ggml.h
enum ggml_type {
	GGML_TYPE_F32     = 0,
	GGML_TYPE_F16     = 1,
	GGML_TYPE_Q4_0    = 2,
	GGML_TYPE_Q4_1    = 3,
	// GGML_TYPE_Q4_2 = 4, support has been removed
	// GGML_TYPE_Q4_3 = 5, support has been removed
	GGML_TYPE_Q5_0    = 6,
	GGML_TYPE_Q5_1    = 7,
	GGML_TYPE_Q8_0    = 8,
	GGML_TYPE_Q8_1    = 9,
	GGML_TYPE_Q2_K    = 10,
	GGML_TYPE_Q3_K    = 11,
	GGML_TYPE_Q4_K    = 12,
	GGML_TYPE_Q5_K    = 13,
	GGML_TYPE_Q6_K    = 14,
	GGML_TYPE_Q8_K    = 15,
	GGML_TYPE_IQ2_XXS = 16,
	GGML_TYPE_IQ2_XS  = 17,
	GGML_TYPE_IQ3_XXS = 18,
	GGML_TYPE_IQ1_S   = 19,
	GGML_TYPE_IQ4_NL  = 20,
	GGML_TYPE_IQ3_S   = 21,
	GGML_TYPE_IQ2_S   = 22,
	GGML_TYPE_IQ4_XS  = 23,
	GGML_TYPE_I8      = 24,
	GGML_TYPE_I16     = 25,
	GGML_TYPE_I32     = 26,
	GGML_TYPE_I64     = 27,
	GGML_TYPE_F64     = 28,
	GGML_TYPE_IQ1_M   = 29,
	GGML_TYPE_BF16    = 30,
	GGML_TYPE_Q4_0_4_4 = 31,
	GGML_TYPE_Q4_0_4_8 = 32,
	GGML_TYPE_Q4_0_8_8 = 33,
	GGML_TYPE_COUNT,
};
#endif

// Copied from mdarray.h
typedef enum {
	MDA_DT_NONE = 0,
	MDA_DT_F32  = 1,
	MDA_DT_F16  = 2,
	MDA_DT_I32  = 3,
	MDA_DT_I8   = 4,
	MDA_DT_Q8_0 = 5,
	MDA_DT_Q4_1 = 6,
	MDA_DT__END
} MdaDataType;

TSDTypeAttr g_tstore_dtype_attr[TS_DTYPE__END] = {
	{ "none",   0,   1, -1, -1 },
	{ "f64" ,   8,   1, GGML_TYPE_F64 , -1, true },
	{ "f32" ,   4,   1, GGML_TYPE_F32 , MDA_DT_F32 , true },
	{ "f16" ,   2,   1, GGML_TYPE_F16 , MDA_DT_F16 , true },
	{ "bf16",   2,   1, GGML_TYPE_BF16, -1, true },
	{ "i64" ,   8,   1, GGML_TYPE_I64 , -1, true },
	{ "i32" ,   4,   1, GGML_TYPE_I32 , MDA_DT_I32 , true },
	{ "i16" ,   2,   1, GGML_TYPE_I16 , -1, true },
	{ "i8"  ,   1,   1, GGML_TYPE_I8  , MDA_DT_I8  , true },
	{ "q8_0",  34,  32, GGML_TYPE_Q8_0, MDA_DT_Q8_0, true },
	{ "q4_1",  20,  32, GGML_TYPE_Q4_1, MDA_DT_Q4_1, true },
	{ "q6_k", 209, 256, GGML_TYPE_Q6_K, -1, true },
	{ "q5_k", 176, 256, GGML_TYPE_Q5_K, -1, true },
	{ "q4_k", 144, 256, GGML_TYPE_Q4_K, -1, true },
};

const TSDTypeAttr* tstore_dtype_attr(int dt) {
	if (!(0 <= dt && dt < TS_DTYPE__END)) dt = 0;
	return &g_tstore_dtype_attr[dt];
}

int tstore_dtype_fromz(const char* s) {
	if (!s) return -1;
	for (unsigned i=1; i<TS_DTYPE__END; ++i)
		if (!strcmp(s, g_tstore_dtype_attr[i].name)) return i;
	return -1;
}

const char * tstore_dtype_str(int dt) {
	return tstore_dtype_attr(dt)->name;
}

int tstore_dtype_from_ggml(int t)
{
	for (unsigned i=1; i<TS_DTYPE__END; ++i)
		if (g_tstore_dtype_attr[i].ggml == t) return i;
	return -1;
}

int tstore_dtype_to_ggml(int dt)
{
	return tstore_dtype_attr(dt)->ggml;
}

int tstore_dtype_from_mda(int t)
{
	for (unsigned i=1; i<TS_DTYPE__END; ++i)
		if (g_tstore_dtype_attr[i].mda == t) return i;
	return -1;
}

int tstore_dtype_to_mda(int dt)
{
	return tstore_dtype_attr(dt)->mda;
}

/* Tensor data */

void tstore_tdata_free_force(TSTensorData* S)
{
	if (S->ownmem) {
		alloc_free(TENSORSTORE_ALLOCATOR, (void*) S->data);
		S->data = NULL;
	}
}

void tstore_tdata_free(TSTensorData* S)
{
	if (!S->perm) tstore_tdata_free_force(S);
}

/* Metadata entry */

void tstore_meta_free(TSMetaEntry* S)
{
	any_free(&S->value);
}

/* Tensor entry */

void tstore_tensor_free(TSTensorEntry* S)
{
	vec_for(S->cache,i,0)
		tstore_tdata_free_force(&S->cache[i]);	
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
	uint64_t size = tstore_tensor_count(S);
	const TSDTypeAttr *attr = tstore_dtype_attr(S->dtype);
	return size * attr->sz_m / attr->sz_d;
}

static
int data_convert(int dtype, int stype, size_t n, void* dst, const void* src)
{
#ifdef TENSORSTORE_USE_GGML
	//TODO: one ggml_init is required for some of this to work
	const TSDTypeAttr *dta = tstore_dtype_attr(dtype);
	if (dtype == TS_DTYPE_F32 && stype == TS_DTYPE_F16) {
		ggml_fp16_to_fp32_row(src, dst, n);
	}
	else if (dtype == TS_DTYPE_F16 && stype == TS_DTYPE_F32) {
		ggml_fp32_to_fp16_row(src, dst, n);
	}
	else if (dtype == TS_DTYPE_F32 && stype == TS_DTYPE_BF16) {
		ggml_bf16_to_fp32_row(src, dst, n);
	}
	else if (dtype == TS_DTYPE_F16 && stype == TS_DTYPE_BF16) {
		void *tmp = alloc_alloc(TENSORSTORE_ALLOCATOR, n*4);
		ggml_bf16_to_fp32_row(src, tmp, n);
		ggml_fp32_to_fp16_row(tmp, dst, n);
		alloc_free(TENSORSTORE_ALLOCATOR, tmp);
	}
	else if (dtype == TS_DTYPE_F32 && stype == TS_DTYPE_F64) {
		for (size_t i=0; i<n; ++i)
			((float*)dst)[i] = ((double*)src)[i];
	}
	else if (dta->sz_d > 1 && stype == TS_DTYPE_F32) {
		size_t r = ggml_quantize_chunk(dta->ggml, src, dst, 0, 1, n, NULL);
		if (r == 0) return -1;
	}
	else if (dta->sz_d > 1 && stype == TS_DTYPE_F16) {
		void *tmp = alloc_alloc(TENSORSTORE_ALLOCATOR, n*4);
		ggml_fp16_to_fp32_row(src, tmp, n);
		size_t r = ggml_quantize_chunk(dta->ggml, tmp, dst, 0, 1, n, NULL);
		alloc_free(TENSORSTORE_ALLOCATOR, tmp);
		if (r == 0) return -1;
	}
#else
	if (dtype == TS_DTYPE_F32 && stype == TS_DTYPE_F16) {
		hpc_set_f32_f16(n, dst, src);
	}
	else if (dtype == TS_DTYPE_F16 && stype == TS_DTYPE_F32) {
		hpc_set_f16_f32(n, dst, src);
	}
	else if (dtype == TS_DTYPE_F32 && stype == TS_DTYPE_F64) {
		hpc_set_f32_f64(n, dst, src);
	}
	else if (dtype == TS_DTYPE_F32 && stype == TS_DTYPE_Q8_0) {
		hpc_set_f32_q8_0(n, dst, src);
	}
	else if (dtype == TS_DTYPE_F16 && stype == TS_DTYPE_Q8_0) {
		hpc_set_f16_q8_0(n, dst, src);
	}
	else if (dtype == TS_DTYPE_Q8_0 && stype == TS_DTYPE_F16) {
		hpc_set_q8_0_f16(n, dst, src);
	}
	else if (dtype == TS_DTYPE_Q4_1 && stype == TS_DTYPE_Q8_0) {
		hpc_set_q4_1_q8_0(n, dst, src);
	}
	else if (dtype == TS_DTYPE_Q8_0 && stype == TS_DTYPE_Q6_K) {
		hpc_set_q8_0_q6_k(n, dst, src);
	}
#endif
	else return -1;
	return 1;
}

int tstore_tensor_data_get(TSTensorEntry* S, TSDType dtype, int flags, 
	TSTensorData* out)
{
	int R=1;
	const bool f_perm  = flags & TSTDG_F_PERM;
	const bool f_write = flags & TSTDG_F_WRITE;

	tstore_tdata_free(out);
		
	//TODO: write flag?
	BISECT_RIGHT_DECL(found, idx, 0, vec_count(S->cache),
		S->cache[i_].dtype - dtype);
	if (found) {
		*out = S->cache[idx];
		return 1;
	}

	TRY_LOG( stream_seek(S->stm, S->offset, 0), "seek to %"PRIu64, S->offset );
	if (stream_read_prep(S->stm, S->size) < S->size)
		ERROR_LOG(-1, "read %"PRIu64" bytes", S->size);
	void *cur = stream_buffer_get(S->stm, NULL);

	//TODO: configure required alignment
	bool f_aligned = ALIGNMENT_CHECK(cur, 32);

	if (dtype == S->dtype) {  //direct
		if (stream_mmap_is(S->stm) && f_aligned && !f_write) {
			*out = (TSTensorData){ dtype, cur, S->size, .perm=true };
		}
		else if (f_perm || !f_aligned || f_write) {
			size_t sz = S->size;
			void *data = alloc_alloc(TENSORSTORE_ALLOCATOR, sz);
			memcpy(data, cur, sz);
			*out = (TSTensorData){ dtype, data, sz, .ownmem=true, .perm=f_perm };
		}
		else {
			*out = (TSTensorData){ dtype, cur, S->size };
		}
	}
	// Data type conversion
	else {
		const TSDTypeAttr *dta = tstore_dtype_attr(dtype);
		const TSDTypeAttr *sta = tstore_dtype_attr(S->dtype);
		if (!(dta->valid))
			ERROR_LOG(-1, "invalid target tensor type %u", dtype);
		
		size_t n  = tstore_tensor_count(S),
		       sz = n * dta->sz_m / dta->sz_d;
		
		void *data = alloc_alloc(TENSORSTORE_ALLOCATOR, sz);

		if (data_convert(dtype, S->dtype, n, data, cur) < 0)
		{
			alloc_free(TENSORSTORE_ALLOCATOR, data);
			ERROR_LOG(-1, "unsupported conversion from %s to %s",
				sta->name, dta->name);
		}

		*out = (TSTensorData){ dtype, data, sz, .ownmem=true, .perm=f_perm };
	}
	
	if (f_perm) {  //store permanent data for later deallocation
		vec_insert(S->cache, idx, 1, out);
	}
	
end:
	return R;
}

/* Formats register */

#ifdef TENSORSTORE_FMT_GGUF
extern const TensorStoreFormat ts_cls_gguf;
#endif
#ifdef TENSORSTORE_FMT_SAFET
extern const TensorStoreFormat ts_cls_safet;
#endif

const TensorStoreFormat * g_tstore_formats[32] = {
#ifdef TENSORSTORE_FMT_GGUF
	&ts_cls_gguf,
#endif
#ifdef TENSORSTORE_FMT_SAFET
	&ts_cls_safet,
#endif
	NULL,
};

int tstore_format_register(const TensorStoreFormat* fmt)
{
	unsigned i=0;
	for (; i<COUNTOF(g_tstore_formats); ++i)
		if (g_tstore_formats[i] == fmt) return i;
	if (i < COUNTOF(g_tstore_formats)) {
		g_tstore_formats[i] = fmt;
		return i;
	}
	return TS_E_UNKNOWN;
}

const TensorStoreFormat* tstore_format_get(int idx)
{
	if (!(0 <= idx && idx < COUNTOF(g_tstore_formats))) return NULL;
	return g_tstore_formats[idx];
}

/* Tensor storage */

void tstore_free(TensorStore* S)
{
	vec_for(S->tensors,i,0) tstore_tensor_free(&S->tensors[i]);
	vec_free(S->tensors_idx);
	vec_free(S->tensors);
	vec_for(S->meta,i,0) tstore_meta_free(&S->meta[i]);
	vec_free(S->meta_idx);
	vec_free(S->meta);
}

void tstore_copy_from(TensorStore* dst, const TensorStore* src)
{
	vec_copyv(dst->meta, src->meta);
	//TODO: check metadata entries with for dynamic memory
	vec_copyv(dst->meta_idx, src->meta_idx);
	vec_copyv(dst->tensors, src->tensors);
	vec_for(dst->tensors,i,0) dst->tensors[i].cache = NULL;
	vec_copyv(dst->tensors_idx, src->tensors_idx);
}

const TensorStoreFormat* tstore_format_detect(Stream* stm)
{
	for (unsigned i=0; i<COUNTOF(g_tstore_formats); ++i) {
		if (!g_tstore_formats[i]) break;
		if (!g_tstore_formats[i]->detect) continue;
		if (g_tstore_formats[i]->detect(stm) > 0)
			return g_tstore_formats[i];
	}
	return NULL;
}

int tstore_read(TensorStore* S, Stream* stm, const TensorStoreFormat* fmt,
	TSCallback* cb)
{
	if (!fmt) {
		fmt = tstore_format_detect(stm);
		if (!fmt) {
			log_error("tensorstore: unknown format");
			return TS_E_FORMAT;
		}
	}
	if (!fmt->read) return TS_E_FORMAT;
	return fmt->read(S, stm, cb);
}

int tstore_write(TensorStore* S, Stream* stm, const TensorStoreFormat* fmt,
	TSCallback* cb)
{
	if (!(fmt && fmt->write)) return TS_E_FORMAT;
	return fmt->write(S, stm, cb);
}

int tstore_info_dump(const TensorStore* S, Stream* stm)
{
	char buf[64];

	stream_printf(stm, "Metadata (%u):\n", vec_count(S->meta));
	vec_forp(TSMetaEntry,S->meta,e,0)
	{
		stream_printf(stm, "%s: %s %s\n", id_str(e->key),
			anyb_name(e->value.t),
			(any_tostr(&e->value, sizeof(buf), buf), buf) );
	}
	
	stream_printf(stm, "Tensors (%u):\n", vec_count(S->tensors));
	vec_forp(TSTensorEntry,S->tensors,e,0)
	{
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
		
#define INDEX_INSERT(VEC, IDX, KEY) do { \
	BISECT_RIGHT_DECL(found, ipos, 0, vec_count(S->VEC##_idx), \
		S->VEC[S->VEC##_idx[i_]].key - (KEY)); \
	if (found) { \
		log_debug("%s duplicate '%s'", #VEC, id_str(KEY)); \
	} else { \
		vec_insert(S->VEC##_idx, ipos, 1, NULL); \
	} \
	S->VEC##_idx[ipos] = (IDX); \
} while (0)

#define INDEX_INSERT_LAST(VEC) \
	INDEX_INSERT(VEC, vec_count(S->VEC)-1, vec_last(S->VEC,0).key)

int tstore_meta_addk(TensorStore* S, StringInt key, Any* value)
{
	TSMetaEntry e = { key, *value };
	vec_push(S->meta, e);
	INDEX_INSERT_LAST(meta);
	*value = (Any){0};  //prevents further external modification
	return vec_count(S->meta)-1;
}

int tstore_meta_adds(TensorStore* S, const char* name, const char* value)
{
	size_t len = strlen(value);
	char *p = alloc_arena_alloc(&S->ss->al, len+1);
	memcpy(p, value, len);
	p[len] = 0;
	Any v = any_string(len, p);
	return tstore_meta_add(S, name, &v);
}

int tstore_tensor_addk(TensorStore* S, StringInt key,
	const TSTensorEntry* e)
{
	vec_push(S->tensors, *e);
	vec_last(S->tensors,0).key = key;
	INDEX_INSERT_LAST(tensors);
	return vec_count(S->tensors)-1;
}

const Any tstore_meta_getk(const TensorStore* S, StringInt key)
{
	BISECT_RIGHT_DECL(found, ipos, 0, vec_count(S->meta_idx),
		S->meta[S->meta_idx[i_]].key - key);
	TSMetaEntry* e = &S->meta[S->meta_idx[ipos]];
	return found ? e->value : (Any){0};
}

TSTensorEntry* tstore_tensor_getk(const TensorStore* S, StringInt key)
{
	BISECT_RIGHT_DECL(found, ipos, 0, vec_count(S->tensors_idx),
		S->tensors[S->tensors_idx[i_]].key - key);
	return found ? &S->tensors[S->tensors_idx[ipos]] : NULL;
}

int tstore_tensor_index_remake(TensorStore* S)
{
	vec_resize(S->tensors_idx, 0);
	vec_for(S->tensors,i,0) {
		int key = S->tensors[i].key;
		INDEX_INSERT(tensors, i, key);
	}
	return 1;
}
