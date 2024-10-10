/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "ccommon/stream.h"
#include "ggml.h"
#include <inttypes.h>

#define GGML_SHAPE_FMT  "%"PRId64"x%"PRId64"x%"PRId64"x%"PRId64
#define GGML_SHAPE_UNPACK(T) \
	(T)->ne[0], (T)->ne[1], (T)->ne[2], (T)->ne[3]

#define GGML_TYPESHAPE_FMT  "%s " GGML_SHAPE_FMT
#define GGML_TYPESHAPE_ARGS(T) \
	ggml_type_name((T)->type), (T)->ne[0], (T)->ne[1], (T)->ne[2], (T)->ne[3]

#define GGML_TENSOR_FMT  "%s: %s %s " GGML_SHAPE_FMT
#define GGML_TENSOR_ARGS(T) \
	ggml_get_name(T), ggml_op_desc(T), ggml_type_name((T)->type), \
	(T)->ne[0], (T)->ne[1], (T)->ne[2], (T)->ne[3]

#define GGML_TENSOR_VARS_N(X,L) \
	L##0n=(X)->ne[0], L##1n=(X)->ne[1], L##2n=(X)->ne[2], L##3n=(X)->ne[3]

#define GGML_TENSOR_VARS_B(X,L) \
	L##0b=(X)->nb[0], L##1b=(X)->nb[1], L##2b=(X)->nb[2], L##3b=(X)->nb[3]

#define GGML_TENSOR_VARS_S(X,L) \
	L##eb=ggml_element_size(X),\
	L##0s=(X)->nb[0]/L##eb, \
	L##1s=(X)->nb[1]/L##eb, \
	L##2s=(X)->nb[2]/L##eb, \
	L##3s=(X)->nb[3]/L##eb

struct ggml_tensor* ggml_name_prefix(struct ggml_tensor* x, const char* pre);

const char* ggml_tensor_typeshape_desc(const struct ggml_tensor* x);

size_t ggml_ctx_tensors_total_size(const struct ggml_context* ctx);

void ggml_ctx_tensors_dump(const struct ggml_context* ctx, Stream* out);

void ggml_tensor_graph_dump(const struct ggml_tensor* result, Stream* out);

void ggml_tensor_export(const struct ggml_tensor* T, const char* path);

typedef struct {
	double asum, first;
	char hash[9];
	char valid;
} ggml_tensor_stat_st;

ggml_tensor_stat_st ggml_tensor_stat(const struct ggml_tensor* T);

// Operations

void ggml_chunk_(struct ggml_context* ctx,
	struct ggml_tensor* x, int n_chunk, int n_dim, struct ggml_tensor*** out);
#define ggml_chunk(C, X, N, D, ...) \
	ggml_chunk_((C), (X), (N), (D), (struct ggml_tensor**[]){__VA_ARGS__});

// Debug operations
// Only works on CPU

struct ggml_tensor*
ggml_debug_print(struct ggml_context* ctx, struct ggml_tensor* t,
	const char* desc, int loglvl);

#define ggml_debug4_print(...) \
	ggml_debug_print(__VA_ARGS__, LOG_LVL_DEBUG4)

struct ggml_tensor*
ggml_debug_export(struct ggml_context* ctx, struct ggml_tensor* t,
	const char* fname);

// Neural networks operations

struct ggml_tensor* ggml_nn_attention(struct ggml_context* ctx,
	struct ggml_tensor* q, struct ggml_tensor* k, struct ggml_tensor* v, 
	bool mask);
