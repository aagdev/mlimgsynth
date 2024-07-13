/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * Machine learning blocks of operations.
 */
#pragma once
#include "ccommon/vector.h"
#include "ccommon/stream.h"
#include "ccommon/logging.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml_extend.h"
#include "ids.h"
#include "tensorstore.h"
#include "localtensor.h"

//TODO: ggml_backend_sched?

typedef struct ggml_tensor MLTensor;

typedef struct {
	MLTensor *tensor;
	StringInt name,
	          key;  //full name to load from the tensor store
} MLCtxTensor;

typedef struct {
	ggml_backend_t backend, backend2;  //Fill
	TensorStore *tstore;  //Fill
	const char *tprefix;  //Set to load tensors with a prefix
	
	struct ggml_context *cp, *cc; //params, compute
	struct ggml_cgraph *graph;
    ggml_gallocr_t allocr;
	ggml_backend_sched_t sched;
	ggml_backend_buffer_t bkbuf;
	
	MLCtxTensor * tensors;  //vector
	MLTensor ** inputs;  //vector
	MLTensor * result;

	// Config
	struct {
		enum ggml_type wtype;  //weights type (default F16)
		unsigned n_tensor_max;
		unsigned multi_compute:1,  //allow multiple calls to compute
		         quiet:1;  //do not output information
		const char *name;
	} c;

	// Information/statistics
	struct MLCtxInfo {
		size_t mem_params, mem_compute, mem_total;
		double t_load, t_compute;
		unsigned n_compute, n_conv;
	} info;
} MLCtx;

void mlctx_free(MLCtx* C);

void mlctx_begin(MLCtx* C, const char* name);

// All in one
int mlctx_run_(MLCtx* C, MLTensor* result, LocalTensor* out,
	const LocalTensor** inputs);
#define mlctx_run(C,R,O,...) \
	mlctx_run_((C), (R), (O), (const LocalTensor*[]){ __VA_ARGS__, NULL })

// Build, alloc and load
// Pending: set input, compute, get output, free
int mlctx_prep(MLCtx* C, MLTensor* result);

/* Step by step interface */

// No need to call build
void mlctx_block_graph_dump(const MLCtx* C, Stream* out);
int mlctx_block_graph_dump_path(const MLCtx* C, const char* path);

int mlctx_build_alloc(MLCtx* C, MLTensor* result);

int mlctx_tstore_load(MLCtx* C, TensorStore* ts);

int mlctx_compute(MLCtx* C);

/* Functions to define blocks */

static inline
void mlctx_block_begin(MLCtx* C)
{
	vec_push(C->tensors, ((MLCtxTensor){ NULL, ID_ML_BLOCK_BEGIN }));
	log_debug2("ML block begin");
}

static inline
MLTensor* mlctx_tensor_add(MLCtx* C, const char* name, MLTensor* tensor)
{
	ggml_name_prefix(tensor, name);
	bool param = (tensor->op == GGML_OP_NONE);
	vec_push(C->tensors, ((MLCtxTensor){ tensor, id_fromz(name) }));
	log_debug2("ML %s: %s " GGML_TYPESHAPE_FMT, param ? "param" : "op",
		name, GGML_TYPESHAPE_ARGS(tensor));
	return tensor;
}

static inline
MLTensor* mlctx_input_add(MLCtx* C, const char* name, enum ggml_type dtype,
	int n0, int n1, int n2, int n3)
{
	MLTensor *T = ggml_new_tensor_4d(C->cp, dtype, n0,n1,n2,n3);
	ggml_set_name(T, name);
	ggml_set_input(T);
	vec_push(C->inputs, T);
	return T;
}
