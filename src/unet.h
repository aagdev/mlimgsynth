/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * UNet implementation for denoising in SD.
 */
#pragma once
#include "mlblock.h"
#include "localtensor.h"

typedef struct {
	int n_ch_in;
	int n_ch_out;
	int n_res_blk;
	int attn_res[4];
	int ch_mult[5];
	int transf_depth[5];
	int n_te;  //time embedding dimensions
	int n_head;
	int d_head;
	int n_ctx;
	int n_ch;
	int ch_adm_in;

	unsigned clip_norm:1,
	         uncond_empty_zero:1,
			 vparam:1;
	
	int n_step_train;
	float sigma_min;
	float sigma_max;
	float *log_sigmas;  //[n_step_train]
} UnetParams;

extern const UnetParams g_unet_sd1;		//SD 1.x
extern const UnetParams g_unet_sd2;		//SD 2.x
extern const UnetParams g_unet_sdxl;	//SDXL
//extern const UnetParams g_unet_svd;		//SVD (stable video diffusion)

MLTensor* mlb_unet_denoise(MLCtx* C, MLTensor* x, MLTensor* time, MLTensor* c,
	MLTensor* label, const UnetParams* P);

void unet_params_init();  //fill global log_sigmas

float unet_sigma_to_t(const UnetParams* P, float sigma);

float unet_t_to_sigma(const UnetParams* P, float t);

typedef struct {
	MLCtx *ctx;
	const UnetParams *par;
	unsigned nfe, i_step, n_step, split:1;
} UnetState;

int unet_denoise_init(UnetState* S, MLCtx* C, const UnetParams* P,
	unsigned lw, unsigned lh, bool split);

int unet_denoise_run(UnetState* S,
	const LocalTensor* x, const LocalTensor* cond, const LocalTensor* label,
	float sigma, LocalTensor* dx);
