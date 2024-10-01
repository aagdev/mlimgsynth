/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * Variational auto-encoder.
 */
#pragma once
#include "mlblock.h"
#include "localtensor.h"

typedef struct {
	int ch_x,
	    ch_z,
		ch,
		n_res,
		n_res_blk,
		ch_mult[5],
		d_embed,
		f_down;  //downsampling total factor
	float scale_factor;
} VaeParams;

extern const VaeParams g_vae_sd1;	//SD 1.x & 2.x
extern const VaeParams g_vae_sdxl;	//SDXL

MLTensor* mlb_sdvae_encoder(MLCtx* C, MLTensor* x, const VaeParams* P);

MLTensor* mlb_sdvae_decoder(MLCtx* C, MLTensor* x, const VaeParams* P);

void sdvae_latent_mean(LocalTensor* latent, const LocalTensor* moments,
	const VaeParams* P);

void sdvae_latent_sample(LocalTensor* latent, const LocalTensor* moments,
	const VaeParams* P);

static inline
void sdvae_encoder_pre(LocalTensor* out, const LocalTensor* img)
{	// [0,1] -> [-1,1]
	ltensor_resize_like(out, img);
	ltensor_for(*out,i,0) out->d[i] = img->d[i]*2 -1;
}

static inline
void sdvae_decoder_post(LocalTensor* out, const LocalTensor* img)
{	// [-1,1] -> [0,1]
	ltensor_resize_like(out, img);
	ltensor_for(*out,i,0) out->d[i] = (img->d[i]+1)/2;
}

int sdvae_encode(MLCtx* C, const VaeParams* P,
	const LocalTensor* img, LocalTensor* latent, int tile_px);

int sdvae_decode(MLCtx* C, const VaeParams* P,
	const LocalTensor* latent, LocalTensor* img, int tile_px);
