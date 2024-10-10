/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 */
#include "ggml_extend.h"
#include "ccommon/ccommon.h"
#include "ccommon/logging.h"
#include "ccommon/vector.h"
#include <inttypes.h>
#include <string.h>
#include <math.h>

#include "ggml-backend.h"

struct ggml_tensor* ggml_name_prefix(struct ggml_tensor* x, const char* pre)
{
	if (x->name[0]) {
		unsigned lp=strlen(pre), ln=strlen(x->name);
		if (lp+1+ln+1 > sizeof(x->name))
			FATAL_LOG("ggml tensor name too long");
		memmove(x->name+lp+1, x->name, ln+1);
		memcpy(x->name, pre, lp);
		x->name[lp] = '.';
	} else {
		strncpy(x->name, pre, sizeof(x->name)-1);
		x->name[sizeof(x->name)-1] = 0;
	}
	return x;
}

const char* ggml_tensor_typeshape_desc(const struct ggml_tensor* x)
{
	static DynStr out=NULL;
	dstr_printf(out, "%s ", ggml_type_name(x->type));

	for (unsigned i=0; i<GGML_MAX_DIMS && x->ne[i]; ++i) {
		if (i) dstr_push(out, 'x');
		dstr_printfa(out, "%"PRId64, x->ne[i]);
	}

	return out;
}

size_t ggml_ctx_tensors_total_size(const struct ggml_context* ctx)
{
	size_t s=0;
	struct ggml_tensor *t = ggml_get_first_tensor(ctx);
	for (; t; t=ggml_get_next_tensor(ctx, t)) s += ggml_nbytes(t);
	return s;
}

void ggml_ctx_tensors_dump(const struct ggml_context* ctx, Stream* out)
{
	struct ggml_tensor *t=ggml_get_first_tensor(ctx);
	for (; t; t=ggml_get_next_tensor(ctx, t)) {
		stream_printf(out, GGML_TENSOR_FMT "\n", GGML_TENSOR_ARGS(t));
	}
}

static const char g_base64_chars[] =
	"ABCDEFGHIJKLMNOPQRSTUVWXYZ" "abcdefghijklmnopqrstuvwxyz" "0123456789" "+/";

#define ggml_tensor_stat_CODE(TYPE,CONV) do { \
	const int64_t GGML_TENSOR_VARS_N(T,t), GGML_TENSOR_VARS_S(T,t); \
	const TYPE *tp = T->data; \
    stat.first = *tp; \
    int64_t hsep = (t3n *t2n *t1n *t0n) / 8; \
	for (int64_t i3=0, i=0; i3<t3n; ++i3) \
	for (int64_t i2=0; i2<t2n; ++i2) \
	for (int64_t i1=0; i1<t1n; ++i1) \
	for (int64_t i0=0; i0<t0n; ++i0, ++i) { \
        double v = (double)CONV(tp[i3*t3s +i2*t2s +i1*t1s +i0*t0s]); \
		stat.asum += fabs(v); \
		hsum[i/hsep] += v; \
    } \
} while(0)

ggml_tensor_stat_st ggml_tensor_stat(const struct ggml_tensor* T)
{
    ggml_tensor_stat_st stat={0};
    if (!T->data) return stat;

    double hsum[8]={0};
    if      (T->type == GGML_TYPE_F32)
		ggml_tensor_stat_CODE(float,);
    else if (T->type == GGML_TYPE_F16)
		ggml_tensor_stat_CODE(ggml_fp16_t,ggml_fp16_to_fp32);
    else return stat;

	// hsum: partial sums of 8 segments
    double hmn=hsum[0], hmx=hmn;
	for (unsigned i=1; i<8; ++i) {
		MINSET(hmn, hsum[i]);
		MAXSET(hmx, hsum[i]);
	}
	// Convert each sum to a character to fast checking by a human
	double f = (hmx > hmn) ? (64 / (hmx - hmn)) : 0;
	f = nextafter(f, 0);
	for (unsigned i=0; i<8; ++i) {
		int idx = (hsum[i] - hmn) * f;
		assert( 0 <= idx && idx < 64 );
		stat.hash[i] = g_base64_chars[idx];
	}
	stat.hash[8] = 0;

    return stat;
}

#define ggml_tensor_export_CODE(TYPE,CONV) do { \
	const TYPE *tp = T->data; \
	for (int64_t i3=0; i3<t3n; ++i3) \
	for (int64_t i2=0; i2<t2n; ++i2) \
	for (int64_t i1=0; i1<t1n; ++i1) \
	for (int64_t i0=0; i0<t0n; ++i0) \
		fprintf(f, "%g\n", (double)CONV(tp[i3*t3s +i2*t2s +i1*t1s +i0*t0s])); \
} while(0)

void ggml_tensor_export(const struct ggml_tensor* T, const char* path)
{
    if (!T->data) return;

    FILE *f = fopen(path, "w");
    if (!f) return;

	const int64_t GGML_TENSOR_VARS_N(T,t), GGML_TENSOR_VARS_S(T,t);
	fprintf(f, "TENSOR ASCII %zd %zd %zd %zd\n", t0n,t1n,t2n,t3n);

    if      (T->type == GGML_TYPE_F32)
		ggml_tensor_export_CODE(float,);
    else if (T->type == GGML_TYPE_F16)
		ggml_tensor_export_CODE(ggml_fp16_t,ggml_fp16_to_fp32);

	fclose(f);
}

void ggml_chunk_(struct ggml_context* ctx,
	struct ggml_tensor* x, int n_chunk, int n_dim, struct ggml_tensor*** out)
{
	GGML_ASSERT( GGML_MAX_DIMS == 4 );
	GGML_ASSERT( 0 <= n_dim && n_dim < GGML_MAX_DIMS );
	GGML_ASSERT( n_dim == 0 );  //TODO
	int64_t ne[GGML_MAX_DIMS];
	memcpy(ne, x->ne, sizeof(ne));
	size_t  nb[GGML_MAX_DIMS];
	memcpy(nb, x->nb, sizeof(nb));
	ne[n_dim] /= n_chunk;
	GGML_ASSERT( n_chunk * ne[n_dim] == x->ne[n_dim] );
	
	size_t offset = ggml_type_size(x->type) * ne[n_dim];

	for (int i=0; i<n_chunk; ++i) {
		*out[i] = ggml_view_4d(ctx, x, ne[0], ne[1], ne[2], ne[3],
					x->nb[1], x->nb[2], x->nb[3], offset*i);
	}
}

void ggml_tensor_debug_print(struct ggml_tensor* dst,
	const struct ggml_tensor* src, int ith, int nth, void* userdata)
{
	if (ith != 0) return;
	GGML_ASSERT( dst->data == src->data );

	const char *desc = userdata;
    ggml_tensor_stat_st stat = ggml_tensor_stat(src);
    char buffer[32];
    sprintf(buffer, GGML_SHAPE_FMT, GGML_SHAPE_UNPACK(src) );
	log_debug("%-12s: %s %-16s %.2e %s %+.2e",
		desc ? desc : src->name, ggml_type_name(src->type),
        buffer, stat.asum, stat.hash, stat.first);
}

struct ggml_tensor*
ggml_debug_print(struct ggml_context* ctx, struct ggml_tensor* t, const char* desc,
	int loglvl)
{
	if (!log_level_check(loglvl)) return t;
	if (!ggml_backend_buffer_is_host(t->buffer)) return t;
	return ggml_map_custom1_inplace(ctx, t, ggml_tensor_debug_print, 1,
		(void*)desc);
}

void ggml_tensor_debug_export(struct ggml_tensor* dst,
	const struct ggml_tensor* src, int ith, int nth, void* userdata)
{
	if (ith != 0) return;
	GGML_ASSERT( dst->data == src->data );

	const char *path = userdata;
    ggml_tensor_export(src, path);
}

struct ggml_tensor*
ggml_debug_export(struct ggml_context* ctx, struct ggml_tensor* t,
	const char* fname)
{
	if (!ggml_backend_buffer_is_host(t->buffer)) return t;
	return ggml_map_custom1_inplace(ctx, t, ggml_tensor_debug_export, 1,
		(void*)fname);
}

struct ggml_tensor* ggml_nn_attention(struct ggml_context* ctx,
	struct ggml_tensor* q, struct ggml_tensor* k, struct ggml_tensor* v, 
	bool mask)
{
//#ifdef USE_FLASH_ATTENTION
//	assert(q->ne[0] == v->ne[0]);
//	return ggml_flash_attn_ext(ctx, q, k, v, NULL, 1.0f, 0.0f);
//	// [N * n_head, n_token, d_head]
//#else
    float d_head = (float)q->ne[0];
	struct ggml_tensor *kq;

    kq = ggml_mul_mat(ctx, k, q);  // [N * n_head, n_token, n_k]
    kq = ggml_scale_inplace(ctx, kq, 1.0f / sqrt(d_head));
    if (mask)
        kq = ggml_diag_mask_inf_inplace(ctx, kq, 0);

    kq = ggml_soft_max_inplace(ctx, kq);

	return ggml_mul_mat(ctx, v, kq);
	// [N * n_head, n_token, d_head]
//#endif
}
