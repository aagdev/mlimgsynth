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

struct ggml_tensor* ggml_name_prefix(struct ggml_tensor* x, const char* pre);

const char* ggml_tensor_typeshape_desc(const struct ggml_tensor* x);

size_t ggml_ctx_tensors_total_size(const struct ggml_context* ctx);

void ggml_ctx_tensors_dump(const struct ggml_context* ctx, Stream* out);

void ggml_tensor_graph_dump(const struct ggml_tensor* result, Stream* out);

void ggml_chunk_(struct ggml_context* ctx,
	struct ggml_tensor* x, int n_chunk, int n_dim, struct ggml_tensor*** out);
#define ggml_chunk(C, X, N, D, ...) \
	ggml_chunk_((C), (X), (N), (D), (struct ggml_tensor**[]){__VA_ARGS__});

// Neural networks operations
struct ggml_tensor* ggml_nn_attention(struct ggml_context* ctx,
	struct ggml_tensor* q, struct ggml_tensor* k, struct ggml_tensor* v, 
	bool mask);
