/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 */
#include "mlblock_nn.h"

#define T  true
#define MLN(NAME,X)  mlctx_tensor_add(C, (NAME), (X))

// The GGML scheduler have problems with inplace operations (2024-07-13)
#if USE_GGML_SCHED
	#define ggml_gelu_inplace  ggml_gelu
	#define ggml_silu_inplace  ggml_silu
#endif

//ref: pytorch.nn.Linear
MLTensor* mlb_nn_linear(MLCtx* C, MLTensor* x, int n_out, bool bias)
{
	MLTensor *w, *b=NULL;
	mlctx_block_begin(C);
	int n_in = x->ne[0];
	w = MLN("weight", ggml_new_tensor_2d(C->cp, C->c.wtype, n_in, n_out));
    x = ggml_mul_mat(C->cc, w, x);
    if (bias) {
		b = MLN("bias", ggml_new_tensor_1d(C->cp, GGML_TYPE_F32, n_out));
        x = ggml_add(C->cc, x, b);
    }
	return x;
}

//ref: pytorch.nn.Conv2d
MLTensor* mlb_nn_conv2d(MLCtx* C, MLTensor* x,
	int ch_out,
	int k0, int k1, int s0, int s1, int p0, int p1, int d0, int d1,
	bool bias)
{
	MLTensor *w, *b;
	mlctx_block_begin(C);
	int ch_in = x->ne[2];
	// x: [N, ch_in, h, w]

	// Warning: conv_2d works only with F16
	w = MLN("weight",
		ggml_new_tensor_4d(C->cp, GGML_TYPE_F16, k0, k1, ch_in, ch_out));
    x = ggml_conv_2d(C->cc, w, x, s0,s1, p0,p1, d0,d1);

	if (bias) {
		b = MLN("bias", ggml_new_tensor_1d(C->cp, GGML_TYPE_F32, ch_out));
        b = ggml_reshape_4d(C->cc, b, 1, 1, ch_out, 1);
        //b = ggml_repeat(C->cc, b, x);
        x = ggml_add(C->cc, x, b);
    }
    // x: [N, ch_out, h, w]

	return x;
}

//ref: pytorch.nn.LayerNorm
MLTensor* mlb_nn_layer_norm(MLCtx* C, MLTensor* x,
	bool affine, bool bias, float eps)
{
	MLTensor *w=NULL, *b=NULL;
	mlctx_block_begin(C);
	int n = x->ne[0];
	if (!(eps>0)) eps = 1e-5;
	x = ggml_norm(C->cc, x, eps);
	if (affine) {
		w = MLN("weight", ggml_new_tensor_1d(C->cp, GGML_TYPE_F32, n));
        x = ggml_mul(C->cc, x, w);
		if (bias) {
			b = MLN("bias", ggml_new_tensor_1d(C->cp, GGML_TYPE_F32, n));
            x = ggml_add(C->cc, x, b);
		}
	}
	return x;
}

//ref: pytorch.nn.GroupNorm
MLTensor* mlb_nn_groupnorm(MLCtx* C, MLTensor* x,
	int n_grp, bool affine, float eps)
{
	MLTensor *w=NULL, *b=NULL;
	mlctx_block_begin(C);
	int n = x->ne[2];
	if (!(eps>0)) eps = 1e-5;

	x = ggml_group_norm(C->cc, x, n_grp, eps);

	if (affine) {
		w = MLN("weight", ggml_new_tensor_1d(C->cp, GGML_TYPE_F32, n));
		b = MLN("bias", ggml_new_tensor_1d(C->cp, GGML_TYPE_F32, n));
		
		if (ggml_n_dims(x) >= 3) {
			w = ggml_reshape_4d(C->cc, w, 1, 1, w->ne[0], 1);
			b = ggml_reshape_4d(C->cc, b, 1, 1, b->ne[0], 1);
		}

        x = ggml_mul(C->cc, x, w);
        // b = ggml_repeat(C->cc, b, x);
        x = ggml_add(C->cc, x, b);
	}

	return x;
}

MLTensor* mlb_downsample(MLCtx* C, MLTensor* x, int ch_out, bool vae)
{
	mlctx_block_begin(C);
	// x: [N, ch_in, h, w]
	if (vae) {
		x = ggml_pad(C->cc, x, 1, 1, 0, 0);
		x = MLN("conv", mlb_nn_conv2d(C, x, ch_out, 3,3, 2,2, 0,0, 1,1, T));
	} else
		x = MLN("conv", mlb_nn_conv2d(C, x, ch_out, 3,3, 2,2, 1,1, 1,1, T));
	// x: [N, ch_out, h/2, w/2]
	return x;
}

MLTensor* mlb_upsample(MLCtx* C, MLTensor* x, int ch_out)
{
	mlctx_block_begin(C);
	// x: [N, ch_in, h, w]
	x = ggml_upscale(C->cc, x, 2, GGML_SCALE_MODE_NEAREST);
	x = MLN("conv", mlb_nn_conv2d(C, x, ch_out, 3,3, 1,1, 1,1, 1,1, T));
	// x: [N, ch_out, h*2, w*2]
	return x;
}

//ref: diffusers/models/resnet.py: class ResnetBlock2D
MLTensor* mlb_resnet(MLCtx* C, MLTensor* x, MLTensor* emb, int ch_out)
{
	MLTensor *x0=x;
	int ch_in = x->ne[2];
	mlctx_block_begin(C);

	x = MLN("norm1", mlb_nn_groupnorm32(C, x));
	x = ggml_silu_inplace(C->cc, x);
	x = MLN("conv1", mlb_nn_conv2d(C, x, ch_out, 3,3, 1,1, 1,1, 1,1, T));
	
	if (emb) {
		emb = ggml_silu(C->cc, emb);
		emb = MLN("emb_proj", mlb_nn_linear(C, emb, ch_out, T));
		emb = ggml_reshape_4d(C->cc, emb, 1, 1, emb->ne[0], emb->ne[1]);
		x = ggml_add(C->cc, x, emb);
	}
	
	x = MLN("norm2", mlb_nn_groupnorm32(C, x));
	x = ggml_silu_inplace(C->cc, x);
	x = MLN("conv2", mlb_nn_conv2d(C, x, ch_out, 3,3, 1,1, 1,1, 1,1, T));

	if (ch_in != ch_out)
		x0 = MLN("skip_conv", mlb_nn_conv2d(C, x0,
				ch_out, 1,1, 1,1, 0,0, 1,1, T));

	x = ggml_add(C->cc, x, x0);
	return x;
}

//ref: diffusers/models/activations.py: class GEGLU
MLTensor* mlb_GEGLU(MLCtx* C, MLTensor* x, int d_out)
{
	MLTensor *g;
	mlctx_block_begin(C);
	// x: [ne3, ne2, ne1, d_in]
	x = MLN("proj", mlb_nn_linear(C, x, d_out*2, true));
	// [ne3, ne2, ne1, d_out*2]
	ggml_chunk(C->cc, x, 2, 0, &x, &g);
	g = ggml_cont(C->cc, g);
	g = ggml_gelu_inplace(C->cc, g);
	x = ggml_mul(C->cc, x, g);
	// [ne3, ne2, ne1, d_out]
	return x;
}

//ref: diffusers/models/attention.py: class FeedForward
MLTensor* mlb_feed_forward(MLCtx* C, MLTensor* x, int d_out, int mult)
{
	mlctx_block_begin(C);
	// x: [ne3, ne2, ne1, d_in]
	int d_in = x->ne[0],
	    d_inner = d_in * mult;
	x = MLN("net.0", mlb_GEGLU(C, x, d_inner));
	// [ne3, ne2, ne1, d_inner]
	//net.1: dropout, skipped for inference
	x = MLN("net.2", mlb_nn_linear(C, x, d_out, true));
	// [ne3, ne2, ne1, d_out]
	return x;
}

//ref: diffusers/models/attention_processor.py: class Attention
MLTensor* mlb_attn_mhead(MLCtx* C, MLTensor* q, MLTensor* k, MLTensor* v,
	int d_out, int d_embed, int n_head, bool mask, bool bias, bool bias_out)
{
	GGML_ASSERT( q->ne[3] == 1 );
	GGML_ASSERT( k->ne[3] == 1 );
	GGML_ASSERT( v->ne[3] == 1 );
	int nq1=q->ne[1], nq2=q->ne[2];
	int nk1=k->ne[1], nk2=k->ne[2];
	int nv1=v->ne[1], nv2=v->ne[2];
	int d_head  = d_embed / n_head;
	GGML_ASSERT( d_head * n_head == d_embed );

	mlctx_block_begin(C);
	q = MLN("q_proj", mlb_nn_linear(C, q, d_embed, bias));
	q = ggml_reshape_4d(C->cc, q, d_head, n_head, nq1, nq2);
	q = ggml_cont(C->cc, ggml_permute(C->cc, q, 0, 2, 1, 3));
	q = ggml_reshape_3d(C->cc, q, d_head, nq1, n_head * nq2);

	k = MLN("k_proj", mlb_nn_linear(C, k, d_embed, bias));
	k = ggml_reshape_4d(C->cc, k, d_head, n_head, nk1, nk2);
	k = ggml_cont(C->cc, ggml_permute(C->cc, k, 0, 2, 1, 3));
	k = ggml_reshape_3d(C->cc, k, d_head, nk1, n_head * nk2);

	v = MLN("v_proj", mlb_nn_linear(C, v, d_embed, bias));
	v = ggml_reshape_4d(C->cc, v, d_head, n_head, nv1, nv2);

#ifdef USE_FLASH_ATTENTION
	v = ggml_cont(C->cc, ggml_permute(C->cc, v, 0, 2, 1, 3));
	v = ggml_reshape_3d(C->cc, v, d_head, nv1, n_head * nv2);
	v = ggml_flash_attn_ext(C->cc, q, k, v, NULL, 1.0f, 0.0f);
#else
	v = ggml_cont(C->cc, ggml_permute(C->cc, v, 1, 2, 0, 3));
	v = ggml_reshape_3d(C->cc, v, nv1, d_head, n_head * nv2);
	v = ggml_nn_attention(C->cc, q, k, v, mask);
	v = ggml_reshape_4d(C->cc, v, d_head, nq1, n_head, nq2);
	v = ggml_cont(C->cc, ggml_permute(C->cc, v, 0, 2, 1, 3));
#endif
	v = ggml_reshape_3d(C->cc, v, d_embed, nq1, nq2);
	
	v = MLN("out_proj", mlb_nn_linear(C, v, d_out, bias_out));
	return v;
}

//ref: diffusers/models/activations.py: class BasicTransformerBlock
MLTensor* mlb_basic_transf(MLCtx* C, MLTensor* x, MLTensor* c,
	int d_out, int d_embed, int n_head)
{
	mlctx_block_begin(C);
	MLTensor *r = x;
	x = MLN("norm1", mlb_nn_layer_norm(C, x, true, true, 0));
	x = MLN("attn1", mlb_attn_mhead(C, x,x,x,
		d_out, d_embed, n_head, false, false, true));
	x = ggml_add(C->cc, x, r);
	r = x;
	x = MLN("norm2", mlb_nn_layer_norm(C, x, true, true, 0));
	x = MLN("attn2", mlb_attn_mhead(C, x,c,c,
		d_out, d_embed, n_head, false, false, true));
	x = ggml_add(C->cc, x, r);
	r = x;
	x = MLN("norm3", mlb_nn_layer_norm(C, x, true, true, 0));
	x = MLN("ff", mlb_feed_forward(C, x, d_out, 4));
	x = ggml_add(C->cc, x, r);
	return x;
}
