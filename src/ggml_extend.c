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
