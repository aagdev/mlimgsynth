/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * Simple storage for tensor.
 */
#pragma once
#include "ccommon/alloc.h"
#include "ccommon/logging.h"
#include "ccommon/image.h"
#include "ggml.h"
#include "ggml-backend.h"

//TODO: replace with general >ndarray> inspired by <vector>.

typedef struct {
	float* d;  //data
	int s[4];  //shape
} LocalTensor;

#define LT_SHAPE_FMT		"%dx%dx%dx%d"
#define LT_SHAPE_UNPACK(X)	(X).s[0], (X).s[1], (X).s[2], (X).s[3]

static inline
size_t ltensor_good(const LocalTensor* S) { return S && S->d; }

static inline
size_t ltensor_nelements(const LocalTensor* S) {
	return (size_t)S->s[0] * S->s[1] * S->s[2] * S->s[3];
}

static inline
size_t ltensor_nbytes(const LocalTensor* S) {
	return sizeof(*S->d) * ltensor_nelements(S);
}

static inline
void ltensor_free(LocalTensor* S) {
	alloc_free(g_allocator, S->d);
	*S = (LocalTensor){0};
}

static inline
void ltensor_resize(LocalTensor* S, int n0, int n1, int n2, int n3) {
	S->s[0] = n0;  S->s[1] = n1;  S->s[2] = n2;  S->s[3] = n3;
	S->d = alloc_realloc(g_allocator, S->d, ltensor_nbytes(S));
}

static inline
void ltensor_resize_like(LocalTensor* S, const LocalTensor* T) {
	ltensor_resize(S, LT_SHAPE_UNPACK(*T));
}

static inline
void ltensor_to_backend(const LocalTensor* S, struct ggml_tensor* out) {
	assert(ltensor_nbytes(S) == ggml_nbytes(out));
	ggml_backend_tensor_set(out, S->d, 0, ltensor_nbytes(S));
}

static inline
void ltensor_from_backend(LocalTensor* S, struct ggml_tensor* out) {
	ltensor_resize(S, out->ne[0], out->ne[1], out->ne[2], out->ne[3]);
	assert(ltensor_nbytes(S) == ggml_nbytes(out));
	ggml_backend_tensor_get(out, S->d, 0, ltensor_nbytes(S));
}

static inline
bool ltensor_shape_equal(const LocalTensor* A, const LocalTensor* B) {
	return (A->s[0] == B->s[0] && A->s[1] == B->s[1] && A->s[2] == B->s[2] &&
		A->s[3] == B->s[3]);
}

static inline
int ltensor_shape_check(const LocalTensor* S, int n0, int n1, int n2, int n3) {
	if (n0>0 && n0 != S->s[0]) return -1;
	if (n1>0 && n1 != S->s[1]) return -1;
	if (n2>0 && n2 != S->s[2]) return -1;
	if (n3>0 && n3 != S->s[3]) return -1;
	return 1;
}

static inline
int ltensor_shape_check_log(const LocalTensor* S, const char* desc,
	int n0, int n1, int n2, int n3)
{
	int r = ltensor_shape_check(S, n0, n1, n2, n3);
	if (r < 0) log_error("%s wrong shape: " LT_SHAPE_FMT,
				desc, LT_SHAPE_UNPACK(*S));
	return r;
}

float ltensor_minmax(const LocalTensor* S, float* min);
float ltensor_sum(const LocalTensor* S);
float ltensor_mean(const LocalTensor* S);

int ltensor_save_path(const LocalTensor* S, const char* path);
int ltensor_load_path(LocalTensor* S, const char* path);

void ltensor_from_image(LocalTensor* S, const Image* img);
void ltensor_to_image(const LocalTensor* S, Image* img);

int ltensor_img_redblue(const LocalTensor* S, Image* img);
int ltensor_img_redblue_path(const LocalTensor* S, const char* path);

#define ltensor_for(T,V,I) \
	for (unsigned V=(I), V##e_=ltensor_nelements(&(T)); V<V##e_; ++V)
