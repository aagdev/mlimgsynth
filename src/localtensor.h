/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * Simple storage for tensors.
 */
#pragma once
#include "ccommon/alloc.h"
#include "ccommon/logging.h"
#include "ggml.h"
#include "ggml-backend.h"

#ifdef LOCALTENSOR_USE_IMAGE
#include "ccommon/image.h"
#endif

typedef struct LocalTensor {
	float	*d;  //data
	int		n[4];  //shape
	int		flags;
} LocalTensor;

enum {
	// Memory owned by the tensor
	LT_F_OWNMEM = 1,
	// User-specified ready state
	LT_F_READY  = 2,
};

#define LT_SHAPE_FMT		"%dx%dx%dx%d"
#define LT_SHAPE_UNPACK(X)	(X).n[0], (X).n[1], (X).n[2], (X).n[3]

static inline
size_t ltensor_good(const LocalTensor* S) { return S && S->d; }

static inline
size_t ltensor_nelements(const LocalTensor* S) {
	return (size_t)S->n[0] * S->n[1] * S->n[2] * S->n[3];
}

static inline
size_t ltensor_nbytes(const LocalTensor* S) {
	return sizeof(*S->d) * ltensor_nelements(S);
}

static inline
void ltensor_free(LocalTensor* S) {
	if (S->flags & LT_F_OWNMEM)
		alloc_free(g_allocator, S->d);
	*S = (LocalTensor){0};
}

static inline
void ltensor_resize(LocalTensor* S, int n0, int n1, int n2, int n3) {
	if (!(S->flags & LT_F_OWNMEM)) S->d = NULL;
	S->n[0] = n0;  S->n[1] = n1;  S->n[2] = n2;  S->n[3] = n3;
	S->d = alloc_realloc(g_allocator, S->d, ltensor_nbytes(S));
	S->flags |= LT_F_OWNMEM;
}

static inline
void ltensor_resize_like(LocalTensor* S, const LocalTensor* T) {
	ltensor_resize(S, LT_SHAPE_UNPACK(*T));
}

static inline
void ltensor_copy(LocalTensor* dst, const LocalTensor* src) {
	ltensor_resize_like(dst, src);
	memcpy(dst->d, src->d, ltensor_nbytes(dst));
}

/* Copy an slice of src into an slice of dst.
 * n#: slice size en elements (#: 0-3 dimension)
 * Li#: slice start (L: d=dst or s=src)
 * Ls#: slice step (L: d=dst or s=src)
 */
void ltensor_copy_slice(LocalTensor* dst, const LocalTensor* src,
	int n0 , int n1 , int n2 , int n3 ,
	int di0, int di1, int di2, int di3,
	int si0, int si1, int si2, int si3,
	int ds0, int ds1, int ds2, int ds3,
	int ss0, int ss1, int ss2, int ss3 );

static inline
void ltensor_copy_slice2(LocalTensor* dst, const LocalTensor* src,
	int n0 , int n1 ,
	int di0, int di1,
	int si0, int si1,
	int ds0, int ds1,
	int ss0, int ss1 )
{
	ltensor_copy_slice(dst, src, n0,n1,src->n[2],src->n[3],
		di0,di1,0,0, si0,si1,0,0, ds0,ds1,1,1, ss0,ss1,1,1);
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
	return (A->n[0] == B->n[0] && A->n[1] == B->n[1] && A->n[2] == B->n[2] &&
		A->n[3] == B->n[3]);
}

static inline
int ltensor_shape_check(const LocalTensor* S, int n0, int n1, int n2, int n3) {
	if (n0>0 && n0 != S->n[0]) return -1;
	if (n1>0 && n1 != S->n[1]) return -1;
	if (n2>0 && n2 != S->n[2]) return -1;
	if (n3>0 && n3 != S->n[3]) return -1;
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

int ltensor_finite_check(const LocalTensor* S);

float ltensor_minmax(const LocalTensor* S, float* min);
float ltensor_sum(const LocalTensor* S);
float ltensor_mean(const LocalTensor* S);

typedef struct {
	float asum, first, min, max;
	char hash[9];
	char valid;
} LocalTensorStats;

LocalTensorStats ltensor_stat(const LocalTensor* S);

void log_ltensor_stats(int loglvl, const LocalTensor* S, const char* desc);

#define log_debug2_ltensor(T, D) \
	log_ltensor_stats(LOG_LVL_DEBUG2, (T), (D))

#define log_debug3_ltensor(T, D) \
	log_ltensor_stats(LOG_LVL_DEBUG3, (T), (D))

// Reduces the sizes by the factors.
// Can be done inplace (dst = src).
void ltensor_downsize(LocalTensor* dst, const LocalTensor* src,
	int f0, int f1, int f2, int f3);

int ltensor_save_stream(const LocalTensor* S, Stream *stm);
int ltensor_save_path(const LocalTensor* S, const char* path);
int ltensor_load_stream(LocalTensor* S, Stream *stm);
int ltensor_load_path(LocalTensor* S, const char* path);

#ifdef LOCALTENSOR_USE_IMAGE
void ltensor_from_image(LocalTensor* S, const Image* img);
void ltensor_to_image(const LocalTensor* S, Image* img);

// Load separately the last channel (usually the transparancy)
void ltensor_from_image_alpha(LocalTensor* S, LocalTensor* alpha, const Image* img);

int ltensor_img_redblue(const LocalTensor* S, Image* img);
int ltensor_img_redblue_path(const LocalTensor* S, const char* path);
#endif

#define ltensor_for(T,V,I) \
	for (unsigned V=(I), V##e_=ltensor_nelements(&(T)); V<V##e_; ++V)
