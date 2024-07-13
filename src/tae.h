/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * Tiny auto-encoder
 *
 * References:
 *   https://github.com/madebyollin/taesd/blob/main/taesd.py 
 */
#pragma once
#include "mlblock.h"
#include "localtensor.h"

typedef struct {
	int ch_x, ch_inner, ch_z, n_blk;
} SdTaeParams;

extern const SdTaeParams g_sdtae_sd1;

MLTensor* mlb_sdtae_encoder(MLCtx* C, MLTensor* x, const SdTaeParams* P);

MLTensor* mlb_sdtae_decoder(MLCtx* C, MLTensor* x, const SdTaeParams* P);

/*static inline
void sdtae_encoder_post(LocalTensor* out, const LocalTensor* latent)
{	// [0,1] -> [-1,1]
	ltensor_resize_like(out, latent);
	ltensor_for(*out,i,0) out->d[i] = latent->d[i]*2 -1;
}

static inline
void sdtae_decoder_pre(LocalTensor* out, const LocalTensor* latent)
{	// [-1,1] -> [0,1]
	ltensor_resize_like(out, latent);
	ltensor_for(*out,i,0) out->d[i] = (latent->d[i]+1)/2;
}*/

int sdtae_encode(MLCtx* C, const SdTaeParams* P,
	const LocalTensor* img, LocalTensor* latent);

int sdtae_decode(MLCtx* C, const SdTaeParams* P,
	const LocalTensor* latent, LocalTensor* img);
