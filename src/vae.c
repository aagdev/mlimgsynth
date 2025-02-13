/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 */
#include "vae.h"
#include "ccommon/ccommon.h"
#include "ccommon/timing.h"
#include "ccommon/logging.h"
#include "ccommon/rng_philox.h"
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
	//x = ggml_debug4_print(C->cc, x, "vae enc down");
	
	// middle
	x = MLN("mid.block_1", mlb_resnet(C, x, NULL, ch_blk));
	x = MLN("mid.attn_1", mlb_attn_2d_self(C, x));
	x = MLN("mid.block_2", mlb_resnet(C, x, NULL, ch_blk));
	// x: [N, ch_blk, h/8, w/8]
	//x = ggml_debug4_print(C->cc, x, "vae enc mid");
	
	// end
	x = MLN("norm_out", mlb_nn_groupnorm32(C, x));
    x = ggml_silu_inplace(C->cc, x);  // nonlinearity/swish
	x = MLN("conv_out", mlb_nn_conv2d(C, x, ch_out, 3,3, 1,1, 1,1, 1,1, T));
    // x: [N, ch_out, h/8, w/8]
	//x = ggml_debug4_print(C->cc, x, "vae enc end");
	
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
	assert(mom->n[3] == 1 && mom->n[2]%2 == 0);
	if (latent == mom) {
		latent->n[2] /= 2;
	} else {
		ltensor_resize(latent, mom->n[0], mom->n[1], mom->n[2]/2, 1);
		memcpy(latent->d, mom->d, ltensor_nbytes(latent));
	}

	ltensor_for(*latent,i,0) latent->d[i] *= P->scale_factor;
}

// ldm.modules.distributions.distributions.DiagonalGaussianDistribution.sample
void sdvae_latent_sample(LocalTensor* latent, const LocalTensor* mom,
	const VaeParams* P)
{
	assert(mom->n[3] == 1 && mom->n[2]%2 == 0);
	int n = mom->n[0] * mom->n[1] * mom->n[2]/2;
	const float *mean   = mom->d,
	            *logvar = mom->d + n;

	if (latent == mom)
		latent->n[2] /= 2;
	else
		ltensor_resize(latent, mom->n[0], mom->n[1], mom->n[2]/2, 1);

	float *rand = alloc_alloc(g_allocator, n*sizeof(float));
	rng_randn(n, rand);

	float *out = latent->d;
	for (int i=0; i<n; ++i)
		out[i] = mean[i] + exp(ccCLAMPED(logvar[i], -30, 20) * 0.5) * rand[i];

	alloc_free(g_allocator, rand);
	
	ltensor_for(*latent,i,0) latent->d[i] *= P->scale_factor;
}

int sdvae_encode(MLCtx* C, const VaeParams* P,
	const LocalTensor* img, LocalTensor* latent, int tile_px)
{
	int R=1;
	LocalTensor ltmp={0}, itmp={0};
	
	const int f = P->f_down,  //latent to image scale (8 for SD)
	          k = f*8;  //overlap margin to prevent border effects when tiling

	if (!(img->n[0]%f==0 && img->n[1]%f==0 && img->n[2]==3 && img->n[3]==1))
		ERROR_LOG(-1, "invalid input image shape: " LT_SHAPE_FMT,
			LT_SHAPE_UNPACK(*img));
	
	int img_n0 = img->n[0],  n0 = img_n0,
		img_n1 = img->n[1],  n1 = img_n1;

	if (tile_px > 0) {
		tile_px = ((tile_px + 63) / 64) * 64;  //rounding up
		n0 = ccMIN( tile_px +k*2, img_n0 );  
		n1 = ccMIN( tile_px +k*2, img_n1 );
		if (n0 == img_n0 && n1 == img_n1)  //one tile
			tile_px = 0;  //disable
	};
	
	// Prepare computation
	C->c.multi_compute = (tile_px > 0);
	mlctx_begin(C, "VAE encode");
	MLTensor *input = mlctx_input_new(C, "img", GGML_TYPE_F32, n0, n1, 3, 1);
	MLTensor *output = mlb_sdvae_encoder(C, input, P);
	TRY( mlctx_prep(C) );

	if (tile_px > 0) {
		double t = timing_time();
		int lat_n0  = img_n0 / f,
		    lat_n1  = img_n1 / f,
			step0  = n0 - k*2,  //overlapping
			step1  = n1 - k*2,
			n_tile0 = (img_n0 + step0 - 1) / step0,
			n_tile1 = (img_n1 + step1 - 1) / step1,
			n_tile  = n_tile0 * n_tile1,
			i_tile  = 0;
		
		log_debug("VAE encode tiling: size:%d,%d step:%d,%d", n0,n1, step0,step1);
		
		// Temporal latent tensor in case latent == img
		ltensor_resize(&ltmp, lat_n0, lat_n1, 8, 1);

		for (int t1=0; t1<n_tile1; ++t1) {
			int i1 = ccMIN(t1 * step1, img_n1 - n1);
			for (int t0=0; t0<n_tile0; ++t0, ++i_tile) {
				int i0 = ccMIN(t0 * step0, img_n0 - n0);

				log_info("VAE tile %d/%d", i_tile+1, n_tile);

				ltensor_resize(&itmp, n0, n1, 3, 1);
				ltensor_copy_slice2(&itmp, img, n0,n1, 0,0, i0,i1, 1,1, 1,1);

				sdvae_encoder_pre(&itmp, &itmp);
				ltensor_to_backend(&itmp, input);
				if (i_tile > 0) C->c.quiet = true;
				TRY( mlctx_compute(C) );
				ltensor_from_backend(&itmp, output);
				log_debug3_ltensor(&itmp, "vae enc");
				
				int d0 = i0 ? k : 0,
				    d1 = i1 ? k : 0;
				ltensor_copy_slice2(&ltmp, &itmp, (n0-k)/f, (n1-k)/f,
					(i0+d0)/f,(i1+d1)/f, d0/f,d1/f, 1,1, 1,1);
			}
		}

		ltensor_copy(latent, &ltmp);
		t = timing_time() - t;
		log_info("VAE encode done {%.3fs}", t);
	}
	else {
		// Set input
		sdvae_encoder_pre(latent, img);  //uses latent as temporal
		ltensor_to_backend(latent, input);

		// Compute
		TRY( mlctx_compute(C) );

		// Get output
		ltensor_from_backend(latent, output);
	}
	
	log_debug2_ltensor(latent, "vae enc");

end:
	C->c.quiet = false;
	ltensor_free(&itmp);
	ltensor_free(&ltmp);
	mlctx_free(C);
	return R;
}

int sdvae_decode(MLCtx* C, const VaeParams* P,
	const LocalTensor* latent, LocalTensor* img, int tile_px)
{
	int R=1;
	MLCtx ctx={0};
	LocalTensor ltmp={0}, itmp={0};

	assert( isfinite( ltensor_sum(latent) ) );

	TRY( ltensor_shape_check_log(latent, "latent", 0,0,4,1) );
	int lat_n0 = latent->n[0],  n0 = lat_n0,
		lat_n1 = latent->n[1],  n1 = lat_n1;
	
	const int f = P->f_down,  //latent to image scale (8 for SD)
	          k = 8;  //overlap margin to prevent border effects when tiling

	if (tile_px > 0) {
		tile_px = ((tile_px + 63) / 64) * 64;  //rounding up
		n0 = ccMIN( tile_px/f +k*2, lat_n0 );  
		n1 = ccMIN( tile_px/f +k*2, lat_n1 );
		if (n0 == lat_n0 && n1 == lat_n1)  //one tile
			tile_px = 0;  //disable
	};

	// Prepare computation
	C->c.multi_compute = (tile_px > 0);
	mlctx_begin(C, "VAE decode");
	MLTensor *input = mlctx_input_new(C, "latent", GGML_TYPE_F32, n0, n1, 4, 1);
	MLTensor *output = mlb_sdvae_decoder(C, input, P);
	TRY( mlctx_prep(C) );

	if (tile_px > 0) {
		double t = timing_time();
		int f = P->f_down,  
		    img_n0  = lat_n0 * f,
		    img_n1  = lat_n1 * f,
			step0  = n0 - k*2,  //overlapping
			step1  = n1 - k*2,
			n_tile0 = (lat_n0 + step0 - 1) / step0,
			n_tile1 = (lat_n1 + step1 - 1) / step1,
			n_tile  = n_tile0 * n_tile1,
			i_tile  = 0;
		
		log_debug("VAE decode tiling: size:%d,%d step:%d,%d", n0,n1, step0,step1);
		
		// Temporal image tensor in case latent == img
		ltensor_resize(&itmp, img_n0, img_n1, 3, 1);

		for (int t1=0; t1<n_tile1; ++t1) {
			int i1 = ccMIN(t1 * step1, lat_n1 - n1);
			for (int t0=0; t0<n_tile0; ++t0, ++i_tile) {
				int i0 = ccMIN(t0 * step0, lat_n0 - n0);

				log_info("VAE tile %d/%d", i_tile+1, n_tile);

				ltensor_resize(&ltmp, n0, n1, 4, 1);
				ltensor_copy_slice2(&ltmp, latent, n0,n1, 0,0, i0,i1, 1,1, 1,1);

				ltensor_to_backend(&ltmp, input);
				if (i_tile > 0) C->c.quiet = true;
				TRY( mlctx_compute(C) );
				ltensor_from_backend(&ltmp, output);
				log_debug3_ltensor(&ltmp, "vae dec");
				
				int d0 = i0 ? k : 0,
				    d1 = i1 ? k : 0;
				ltensor_copy_slice2(&itmp, &ltmp, (n0-k)*f, (n1-k)*f,
					(i0+d0)*f,(i1+d1)*f, d0*f,d1*f, 1,1, 1,1);
			}
		}

		ltensor_copy(img, &itmp);
		t = timing_time() - t;
		log_info("VAE decode done {%.3fs}", t);
	}
	else {
		// Set input
		ltensor_to_backend(latent, input);

		// Compute
		TRY( mlctx_compute(C) );

		// Get output
		ltensor_from_backend(img, output);
	}
	
	sdvae_decoder_post(img, img);
	log_debug2_ltensor(img, "vae dec");

end:
	C->c.quiet = false;
	ltensor_free(&ltmp);
	ltensor_free(&itmp);
	mlctx_free(&ctx);
	return R;
}
