/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * Blocks commonly used in neural networks.
 */
#pragma once
#include "mlblock.h"

MLTensor* mlb_nn_linear(MLCtx* C, MLTensor* x, int n_out, bool bias);

MLTensor* mlb_nn_conv2d(MLCtx* C, MLTensor* x,
	int ch_out,
	int k0, int k1, int s0, int s1, int p0, int p1, int d0, int d1,
	bool bias);

MLTensor* mlb_nn_layer_norm(MLCtx* C, MLTensor* x,
	bool affine, bool bias, float eps);

MLTensor* mlb_nn_groupnorm(MLCtx* C, MLTensor* x,
	int n_grp, bool affine, float eps);

static inline
MLTensor* mlb_nn_groupnorm32(MLCtx* C, MLTensor* x) {
	return mlb_nn_groupnorm(C, x, 32, true, 1e-6);
}

MLTensor* mlb_downsample(MLCtx* C, MLTensor* x, int ch_out, bool vae);

MLTensor* mlb_upsample(MLCtx* C, MLTensor* x, int ch_out);

MLTensor* mlb_resnet(MLCtx* C, MLTensor* x, MLTensor* emb, int ch_out);

MLTensor* mlb_GEGLU(MLCtx* C, MLTensor* x, int d_out);

MLTensor* mlb_feed_forward(MLCtx* C, MLTensor* x, int d_out, int mult);

MLTensor* mlb_attn_mhead(MLCtx* C, MLTensor* q, MLTensor* k, MLTensor* v,
	int d_out, int d_embed, int n_head, bool mask, bool bias, bool bias_out);

MLTensor* mlb_basic_transf(MLCtx* C, MLTensor* x, MLTensor* c,
	int d_out, int d_embed, int n_head);
