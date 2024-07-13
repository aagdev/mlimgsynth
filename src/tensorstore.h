/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * Store the information required to load model parameter tensors.
 */
#pragma once
#include "ccommon/stream.h"
#include "ccommon/vector.h"
#include "ggml.h"
#include "ids.h"

typedef enum { 
	TS_DTYPE_F64 = 1,
	TS_DTYPE_F32,
	TS_DTYPE_F16,
	TS_DTYPE_I64,
	TS_DTYPE_I32,
	TS_DTYPE_I16,
	TS_DTYPE_I8,
	TS_DTYPE__END,
} TSDType;

int tstore_dtype_fromz(const char* s);
const char * tstore_dtype_str(int t);

size_t tstore_dtype_size(int t);

// Returns -1 if not found
int tstore_dtype_from_ggml(enum ggml_type t);
int tstore_dtype_to_ggml(int t);

typedef struct {
	TSDType dtype;  //target dtype
	const void *data;
	size_t size;
} TSTensorData;

typedef struct {
	int key;  //str_id
	TSDType dtype;
	unsigned shape_n, shape[8];
	uint64_t offset, size;
	Stream *stm;
	TSTensorData *cache;  //converted tensor cache, vector, sorted
} TSTensorEntry;

uint64_t tstore_tensor_count(const TSTensorEntry* S);
uint64_t tstore_tensor_size(const TSTensorEntry* S);

int tstore_tensor_data_get(TSTensorEntry* S, TSDType dtype, TSTensorData* out);
int tstore_tensor_read(TSTensorEntry* S, struct ggml_tensor* t);
enum { TSTG_R_DIRECT=1, TSTG_R_CONVERT=2 };

#define TSTENSOR_SHAPE4_FMT  "%ux%ux%ux%u"
#define TSTENSOR_SHAPE4_UNPACK(T) \
	(T).shape[0], (T).shape[1], (T).shape[2], (T).shape[3]

typedef struct {
	int key;  //str_id
	int value;  //str_id
} TSMetaEntry;

typedef struct {
	TSTensorEntry * tensors;  //vector, source order
	TSMetaEntry * meta;  //vector, source order
	unsigned * tensors_idx;  //key sorted
	unsigned * meta_idx;  //key sorted
	uint64_t os_data, os_end;

	// Callback before adding a new tensor
	// Allows to transform tensor names
	int (*cb_add)(void*, DynStr*);
	void *cb_user;
} TensorStore;

void tstore_free(TensorStore*);

void tstore_copy_from(TensorStore* dst, const TensorStore* src);

int tstore_info_dump(const TensorStore*, Stream* out);

int tstore_info_dump_path(const TensorStore*, const char* path);

void tstore_meta_add(TensorStore*, const char* key, const char* value);

// Returns NULL if not found
const char* tstore_meta_get(const TensorStore*, const char* key);

void tstore_tensor_add(TensorStore*, DynStr* name, const TSTensorEntry*);

// Returns NULL if not found
TSTensorEntry* tstore_tensor_getk(const TensorStore*, StringInt key);

static inline
TSTensorEntry* tstore_tensor_get(const TensorStore* S, const char* name) {
	return tstore_tensor_getk(S, id_fromz(name));
}
