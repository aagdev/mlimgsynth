/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * Parse and store information from common tensor storage formats used for
 * machine learning.
 *
 * Example:
	StringStore ss={0};
	TensorStore ts={ .ss=&ss };
	Stream stm={0};
	TRY( stream_open_file(&stm, "model.gguf") );
	TRY( tstore_read(&tsp, &stm, NULL) );
	TRY( tstore_info_dump_path(&sp, "model-info.txt") );
end:
	tstore_free(&tsp);
	stream_close(&stm, 0);
 */
#pragma once
#include "ccommon/stream.h"
#include "ccommon/vector.h"
#include "ccommon/any.h"
#include "ccommon/stringstore.h"

typedef struct TensorStore TensorStore;

/* Error codes */

typedef enum {
	TS_E_UNKNOWN		= -0x3001,
	TS_E_OVERFLOW		= -0x3002,
	TS_E_FORMAT			= -0x3003,
	TS_E_READ			= -0x3004,
	TS_E_METADATA		= -0x3005,
	TS_E_DTYPE			= -0x3006,
	TS_E_WRITE			= -0x3007,
} TSError;

/* Data types */

typedef enum {
	TS_DTYPE_NONE,
	TS_DTYPE_F64,
	TS_DTYPE_F32,
	TS_DTYPE_F16,
	TS_DTYPE_BF16,
	TS_DTYPE_I64,
	TS_DTYPE_I32,
	TS_DTYPE_I16,
	TS_DTYPE_I8,
	// GGML quantization
	TS_DTYPE_Q8_0,
	TS_DTYPE_Q4_1,
	TS_DTYPE_Q6_K,
	TS_DTYPE_Q5_K,
	TS_DTYPE_Q4_K,
	TS_DTYPE__END,
} TSDType;

typedef struct {
	const char	*name;
	uint64_t	sz_m, sz_d;  // tensor size = count * sz_m / sz_d
	int			ggml,
				mda;
	unsigned	valid:1;
} TSDTypeAttr;

const TSDTypeAttr* tstore_dtype_attr(int dt);

int tstore_dtype_fromz(const char* s);

const char * tstore_dtype_str(int dt);

// Returns -1 if not found
int tstore_dtype_from_ggml(int ggml_type);
int tstore_dtype_to_ggml(int dt);

// Returns -1 if not found
int tstore_dtype_from_mda(int mda_dtype);
int tstore_dtype_to_mda(int dt);

/* Tensor data */

typedef struct {
	TSDType dtype;  	//data type
	void *data;
	size_t size;
	unsigned ownmem:1,
	         perm:1;	//<data> remains valid for the lifetime of the tensor store
} TSTensorData;

void tstore_tdata_free(TSTensorData*);

/* Meta data entry */

typedef struct {
	int key;  //str_id
	Any value;
} TSMetaEntry;

/* Tensor entry */

typedef struct {
	int key;  //str_id
	TSDType dtype;
	unsigned shape_n, shape[4];
	uint64_t offset, size;
	Stream *stm;
	TSTensorData *cache;  //converted tensor cache, vector, sorted
} TSTensorEntry;

uint64_t tstore_tensor_count(const TSTensorEntry* S);
uint64_t tstore_tensor_size(const TSTensorEntry* S);

#define TSTENSOR_SHAPE4_FMT  "%ux%ux%ux%u"
#define TSTENSOR_SHAPE4_UNPACK(T) \
	(T).shape[0], (T).shape[1], (T).shape[2], (T).shape[3]

/* Return a TSTensorData object with the tensor data with type dtype.
 * If flags & TSTDG_F_PERM, the data pointer is permanent, otherwise,
 * The TSTensorData object must be free'd after use.
 */
int tstore_tensor_data_get(TSTensorEntry* S, TSDType dtype, int flags,
	TSTensorData* out);

enum tstore_tensor_data_get_flags_t {
	TSTDG_F_PERM  = 1,  // out->data is in permanent storage
	TSTDG_F_WRITE = 2,  // Returns memory that can be written
};

/* IO CallBack */

typedef struct {
	int (*func)(void* user, TensorStore* ts, TSTensorEntry* te, DynStr* pname);
	void *user;
} TSCallback;

static inline
int tstore_cb_call(TSCallback* cb, TensorStore* ts, TSTensorEntry* te,
	DynStr* pname)
{
	if (!cb || !cb->func) return 1;
	return cb->func(cb->user, ts, te, pname);
}

/* Parser */

typedef struct {
	const char *name, *ext;
	int (*detect)(Stream*);
	int (*read)(TensorStore*, Stream*, TSCallback*);
	int (*write)(TensorStore*, Stream*, TSCallback*);
} TensorStoreFormat;

int tstore_format_register(const TensorStoreFormat*);

const TensorStoreFormat* tstore_format_get(int idx);

/* Store */

struct TensorStore {
	TSTensorEntry * tensors;  //vector, source order
	TSMetaEntry * meta;  //vector, source order
	unsigned * tensors_idx;  //vector, key sorted
	unsigned * meta_idx;  //vector, key sorted
	StringStore *ss;  //external store for tensor names strings, fill before use
};

void tstore_free(TensorStore*);

/* Read tensors information from a stream.
 * Does not read the tensors data.
 * fmt: data format. If NULL, tries to guess from the data.
 * cb: Optional. Function called before adding each tensor. If it returns non
 *     positive, the tensor is not added. May change the name.
 */
int tstore_read(TensorStore* S, Stream* stm, const TensorStoreFormat* fmt,
	TSCallback* cb);

/* Write tensors information to a stream.
 * Does not writes the tensors data.
 * fmt: data format.
 * cb: Optional. Function called before writing each tensor. If it returns non
 *     positive, the tensor is not written. May store a new name in *pname.
 */
int tstore_write(TensorStore* S, Stream* stm, const TensorStoreFormat* fmt,
	TSCallback* cb);

/* Tries to detect the data format of a stream.
 */
const TensorStoreFormat* tstore_format_detect(Stream* stm);

/* Make copy of the store src in dst.
 * Useful for conversion and for other manipulations.
 */
void tstore_copy_from(TensorStore* dst, const TensorStore* src);

/* Write human readable information about the store.
 */
int tstore_info_dump(const TensorStore*, Stream* out);

/* Write human readable information about the store.
 */
int tstore_info_dump_path(const TensorStore*, const char* path);

/* Add a new key-value metadata entry.
 * Takes ownership of value.
 */
int tstore_meta_addk(TensorStore* S, StringInt key, Any* value);

/* Add a new key-value metadata entry.
 * Takes ownership of value.
 */
static inline
int tstore_meta_add(TensorStore* S, const char* name, Any* value)
{
	int key = strsto_add(S->ss, strsl_fromz(name));
	return tstore_meta_addk(S, key, value);
}

/* Add a new key-value metadata entry.
 * String value.
 */
int tstore_meta_adds(TensorStore* S, const char* name, const char* value);

/* Find and return a metadata entry.
 * Returns empty (t=0) if not found.
 */
const Any tstore_meta_getk(const TensorStore* S, StringInt key);

/* Find and return a metadata entry.
 * Return empty (t=0) if not found.
 */
static inline
const Any tstore_meta_get(const TensorStore* S, const char* name) {
	int key = strsto_add(S->ss, strsl_fromz(name));
	return tstore_meta_getk(S, key);
}

/* Add a new tensor entry.
 * entry->key is ignored.
 */
int tstore_tensor_addk(TensorStore* S, StringInt key,
	const TSTensorEntry* entry);

/* Add a new tensor entry.
 * entry->key is ignored.
 */
static inline
int tstore_tensor_add(TensorStore* S, const char* name,
	const TSTensorEntry* entry)
{
	int key = strsto_add(S->ss, strsl_fromz(name));
	return tstore_tensor_addk(S, key, entry);
}

/* Find and return a tensor entry.
 * Return NULL if not found.
 */
TSTensorEntry* tstore_tensor_getk(const TensorStore*, StringInt key);

/* Find and return a tensor entry.
 * Return NULL if not found.
 */
static inline
TSTensorEntry* tstore_tensor_get(const TensorStore* S, const char* name) {
	int key = strsto_add(S->ss, strsl_fromz(name));
	return tstore_tensor_getk(S, key);
}

/* Remake the tensors index.
 * Call after changing the tensor manually.
 */
int tstore_tensor_index_remake(TensorStore* S);

/* Free all stored tensor data.
 */
int tstore_cache_clear(TensorStore* S);
