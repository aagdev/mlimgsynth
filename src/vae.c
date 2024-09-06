/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 */
#include "vae.h"
#include "rng_philox.h"
#include "ccommon/ccommon.h"
#include "ccommon/logging.h"
#include "ggml_extend.h"
#include "mlblock_nn.h"
#include <stdlib.h>
#include <math.h>

#define T  true
#define MLN(NAME,X)  mlctx_tensor_add(C, (NAME), (X))

// The GGML scheduler have problems with inplace operations (2024-07-13)
#if USE_GGML_SCHED
	#define ggml_silu_inplace  ggml_silu
#endif

const VaeParams g_vae_sd1 = {
	.ch_x			= 3, 
	.ch_z			= 4,
	.ch				= 128,
	.n_res			= 4,
	.n_res_blk		= 2,
	.ch_mult		= {1,2,4,4},
	.d_embed		= 4,
	.f_down			= 8,  //(n_res-1)**2
	.scale_factor	= 0.18215f,
};

const VaeParams g_vae_sdxl = {
	.ch_x			= 3, 
	.ch_z			= 4,
	.ch				= 128,
	.n_res			= 4,
	.n_res_blk		= 2,
	.ch_mult		= {1,2,4,4},
	.d_embed		= 4,
	.f_down			= 8,  //(n_res-1)**2
	.scale_factor	= 0.13025f,
};

MLTensor* mlb_attn_2d_self(MLCtx* C, MLTensor* x)
{
	MLTensor *x0=x, *q, *k, *v;
	mlctx_block_begin(C);
    // x: [N, ch_in, h, w]

	x = MLN("norm", mlb_nn_groupnorm32(C, x));

	const int64_t w=x->ne[0], h=x->ne[1], c=x->ne[2], n=x->ne[3];

	q = MLN("q", mlb_nn_conv2d(C, x, c, 1,1, 1,1, 0,0, 1,1, T));
	q = ggml_cont(C->cc, ggml_permute(C->cc, q, 1, 2, 0, 3));  //[N, h, w, c]
	q = ggml_reshape_3d(C->cc, q, c, h * w, n);              //[N, h*w, c]

	k = MLN("k", mlb_nn_conv2d(C, x, c, 1,1, 1,1, 0,0, 1,1, T));
	k = ggml_cont(C->cc, ggml_permute(C->cc, k, 1, 2, 0, 3));  //[N, h, w, c]
	k = ggml_reshape_3d(C->cc, k, c, h * w, n);              //[N, h*w, c]

	v = MLN("v", mlb_nn_conv2d(C, x, c, 1,1, 1,1, 0,0, 1,1, T));
	v = ggml_reshape_3d(C->cc, v, h * w, c, n);  //[N, c, h*w]

	x = ggml_nn_attention(C->cc, q, k, v, false);  //[N, h*w, c]
	x = ggml_cont(C->cc, ggml_permute(C->cc, x, 1, 0, 2, 3));  //[N, c, h*w]
	x = ggml_reshape_4d(C->cc, x, w, h, c, n);               //[N, c, h, w]

	x = MLN("proj_out", mlb_nn_conv2d(C, x, c, 1,1, 1,1, 0,0, 1,1, T));
	x = ggml_add(C->cc, x, x0);
	return x;
}

MLTensor* mlb_kl_encoder(MLCtx* C, MLTensor* x,
	int ch_out, int ch, int n_res, int n_res_blk, const int ch_mult[n_res])
{
	char name[64];
	mlctx_block_begin(C);
    // x: [N, ch_in, h, w]

	x = MLN("conv_in", mlb_nn_conv2d(C, x, ch, 3,3, 1,1, 1,1, 1,1, T));
	// x: [N, ch, h, w]

	// downsampling
	int ch_blk = ch;
	for (unsigned i=0; i<n_res; ++i) {
		int ch_blk_out = ch * ch_mult[i];
		for (unsigned j=0; j<n_res_blk; ++j) {
			sprintf(name, "down.%d.block.%d", i, j);
			x = MLN(name, mlb_resnet(C, x, NULL, ch_blk_out));
			ch_blk = ch_blk_out;
		}
		if (i+1 != n_res) {
			sprintf(name, "down.%d.downsample", i);
			x = MLN(name, mlb_downsample(C, x, ch_blk, T));
		}
	}
	// x: [N, ch_blk, h/8, w/8]
	
	// middle
	x = MLN("mid.block_1", mlb_resnet(C, x, NULL, ch_blk));
	x = MLN("mid.attn_1", mlb_attn_2d_self(C, x));
	x = MLN("mid.block_2", mlb_resnet(C, x, NULL, ch_blk));
	// x: [N, ch_blk, h/8, w/8]
	
	// end
	x = MLN("norm_out", mlb_nn_groupnorm32(C, x));
    x = ggml_silu_inplace(C->cc, x);  // nonlinearity/swish
	x = MLN("conv_out", mlb_nn_conv2d(C, x, ch_out, 3,3, 1,1, 1,1, 1,1, T));
    // x: [N, ch_out, h/8, w/8]
	
	return x;
}

MLTensor* mlb_sdvae_encoder(MLCtx* C, MLTensor* x, const VaeParams* P)
{
	GGML_ASSERT( x->ne[2] == P->ch_x );
	x = MLN("encoder", mlb_kl_encoder(C, x,
		P->ch_z*2, P->ch, P->n_res, P->n_res_blk, P->ch_mult));
	x = MLN("quant_conv", mlb_nn_conv2d(C, x,
		P->ch_z*2, 1,1, 1,1, 0,0, 1,1, T));
	return x;
}

MLTensor* mlb_kl_decoder(MLCtx* C, MLTensor* x,
	int ch_out, int ch, int n_res, int n_res_blk, const int ch_mult[n_res])
{
	char name[64];
	mlctx_block_begin(C);
    // x: [N, ch_z, h, w]
	
	int ch_blk = ch * ch_mult[n_res-1];
	x = MLN("conv_in", mlb_nn_conv2d(C, x, ch_blk, 3,3, 1,1, 1,1, 1,1, T));
	// x: [N, ch_blk, h, w]
	
	// middle
	x = MLN("mid.block_1", mlb_resnet(C, x, NULL, ch_blk));
	x = MLN("mid.attn_1", mlb_attn_2d_self(C, x));
	x = MLN("mid.block_2", mlb_resnet(C, x, NULL, ch_blk));
	// x: [N, ch_blk, h, w]

	// upsampling
	for (int i=n_res-1; i>=0; --i) {
		int ch_blk_out = ch * ch_mult[i];
		for (unsigned j=0; j<n_res_blk+1; ++j) {
			sprintf(name, "up.%d.block.%d", i, j);
			x = MLN(name, mlb_resnet(C, x, NULL, ch_blk_out));
			ch_blk = ch_blk_out;
		}
		if (i != 0) {
			sprintf(name, "up.%d.upsample", i);
			x = MLN(name, mlb_upsample(C, x, ch_blk));
		}
	}
	// x: [N, ch_blk, h*8, w*8]
	
	// end
	x = MLN("norm_out", mlb_nn_groupnorm32(C, x));
    x = ggml_silu_inplace(C->cc, x);  // nonlinearity/swish
	x = MLN("conv_out", mlb_nn_conv2d(C, x, ch_out, 3,3, 1,1, 1,1, 1,1, T));
    // x: [N, ch_out, h*8, w*8]
	
	return x;
}

MLTensor* mlb_sdvae_decoder(MLCtx* C, MLTensor* x, const VaeParams* P)
{
	GGML_ASSERT( x->ne[2] == P->ch_z );
	x = ggml_scale(C->cc, x, 1 / P->scale_factor);
	x = MLN("post_quant_conv", mlb_nn_conv2d(C, x,
		P->d_embed, 1,1, 1,1, 0,0, 1,1, T));
	x = MLN("decoder", mlb_kl_decoder(C, x,
		P->ch_x, P->ch, P->n_res, P->n_res_blk, P->ch_mult));
	return x;
}

void sdvae_latent_mean(LocalTensor* latent, const LocalTensor* mom,
	const VaeParams* P)
{
	assert(mom->s[3] == 1 && mom->s[2]%2 == 0);
	if (latent == mom) {
		latent->s[2] /= 2;
	} else {
		ltensor_resize(latent, mom->s[0], mom->s[1], mom->s[2]/2, 1);
		memcpy(latent->d, mom->d, ltensor_nbytes(latent));
	}

	ltensor_for(*latent,i,0) latent->d[i] *= P->scale_factor;
}

// ldm.modules.distributions.distributions.DiagonalGaussianDistribution.sample
void sdvae_latent_sample(LocalTensor* latent, const LocalTensor* mom,
	const VaeParams* P)
{
	assert(mom->s[3] == 1 && mom->s[2]%2 == 0);
	int n = mom->s[0] * mom->s[1] * mom->s[2]/2;
	const float *mean   = mom->d,
	            *logvar = mom->d + n;

	if (latent == mom)
		latent->s[2] /= 2;
	else
		ltensor_resize(latent, mom->s[0], mom->s[1], mom->s[2]/2, 1);

	float *rand = alloc_alloc(g_allocator, n*sizeof(float));
	rng_randn(n, rand);

	float *out = latent->d;
	for (int i=0; i<n; ++i)
		out[i] = mean[i] + exp(ccCLAMPED(logvar[i], -30, 20) * 0.5) * rand[i];

	alloc_free(g_allocator, rand);
	
	ltensor_for(*latent,i,0) latent->d[i] *= P->scale_factor;
}

int sdvae_encode(MLCtx* C, const VaeParams* P,
	const LocalTensor* img, LocalTensor* latent)
{
	int R=1;
	
	TRY( ltensor_shape_check_log(img, "img", 0,0,3,1) );
	
	// Prepare computation
	mlctx_begin(C, "VAE encode");
	MLTensor *input = mlctx_input_new(C, "img", GGML_TYPE_F32,
		LT_SHAPE_UNPACK(*img) );
	MLTensor *output = mlb_sdvae_encoder(C, input, P);
	TRY( mlctx_prep(C) );

	// Set input
	sdvae_encoder_pre(latent, img);  //uses latent as temporal
	ltensor_to_backend(latent, input);

	// Compute
	TRY( mlctx_compute(C) );

	// Get output
	ltensor_from_backend(latent, output);

end:
	mlctx_free(C);
	return R;
}

int sdvae_decode(MLCtx* C, const VaeParams* P,
	const LocalTensor* latent, LocalTensor* img)
{
	int R=1;
	MLCtx ctx={0};

	TRY( ltensor_shape_check_log(latent, "latent", 0,0,4,1) );
	
	// Prepare computation
	mlctx_begin(C, "VAE decode");
	MLTensor *input = mlctx_input_new(C, "latent", GGML_TYPE_F32,
		LT_SHAPE_UNPACK(*latent));
	MLTensor *output = mlb_sdvae_decoder(C, input, P);
	TRY( mlctx_prep(C) );

	// Set input
	ltensor_to_backend(latent, input);

	// Compute
	TRY( mlctx_compute(C) );

	// Get output
	ltensor_from_backend(img, output);
	sdvae_decoder_post(img, img);

end:
	mlctx_free(&ctx);
	return R;
}
