/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 */
#include "tae.h"
#include "mlblock_nn.h"

#define T  1  //true
#define F  0  //false
#define MLN(NAME,X)  mlctx_tensor_add(C, (NAME), (X))

// The GGML scheduler have problems with inplace operations (2024-07-13)
#if USE_GGML_SCHED
	#define ggml_relu_inplace  ggml_relu
	#define ggml_tanh_inplace  ggml_tanh
#endif

const SdTaeParams g_sdtae_sd1 = {
	.ch_x     = 3,
	.ch_inner = 64,
	.ch_z     = 4,
	.n_blk    = 3,
};

MLTensor* mlb_sdtae_block(MLCtx* C, MLTensor* x, int ch_out)
{
	MLTensor *x0=x;
	mlctx_block_begin(C);
	int ch_in = x->ne[2];
	x = MLN("conv.0", mlb_nn_conv2d(C, x, ch_out, 3,3, 1,1, 1,1, 1,1, T));
	x = ggml_relu_inplace(C->cc, x);
	x = MLN("conv.2", mlb_nn_conv2d(C, x, ch_out, 3,3, 1,1, 1,1, 1,1, T));
	x = ggml_relu_inplace(C->cc, x);
	x = MLN("conv.4", mlb_nn_conv2d(C, x, ch_out, 3,3, 1,1, 1,1, 1,1, T));
	if (ch_in != ch_out)
		x0 = MLN("skip", mlb_nn_conv2d(C, x0, ch_out, 1,1, 1,1, 1,1, 1,1, T));
	x = ggml_add(C->cc, x, x0);
	x = ggml_relu_inplace(C->cc, x);
	return x;
}

#define IDX2NAME(I)  (sprintf(name, "%d", (I)), name)

MLTensor* mlb_sdtae_encoder(MLCtx* C, MLTensor* x, const SdTaeParams* P)
{
	int iblk=0;
	char name[32];
	mlctx_block_begin(C);

	x = MLN(IDX2NAME(iblk++), mlb_nn_conv2d(C, x,
		P->ch_inner, 3,3, 1,1, 1,1, 1,1, true));
	x = MLN(IDX2NAME(iblk++), mlb_sdtae_block(C, x, P->ch_inner));
	
	for (int j=0; j<3; ++j) {
		x = MLN(IDX2NAME(iblk++), mlb_nn_conv2d(C, x,
			P->ch_inner, 3,3, 2,2, 1,1, 1,1, false));
		for (int i=0; i<P->n_blk; ++i)
			x = MLN(IDX2NAME(iblk++), mlb_sdtae_block(C, x, P->ch_inner));
	}
	
	x = MLN(IDX2NAME(iblk++), mlb_nn_conv2d(C, x,
		P->ch_z, 3,3, 1,1, 1,1, 1,1, true));
	return x;
}

MLTensor* mlb_sdtae_decoder(MLCtx* C, MLTensor* x, const SdTaeParams* P)
{
	int iblk=0;
	char name[32];
	mlctx_block_begin(C);

	x = ggml_scale(C->cc, x, 1.0f / 3.0f);
    x = ggml_tanh_inplace(C->cc, x);
    x = ggml_scale(C->cc, x, 3.0f);
	
	x = MLN(IDX2NAME(iblk++), mlb_nn_conv2d(C, x,
		P->ch_inner, 3,3, 1,1, 1,1, 1,1, true));
	x = ggml_relu_inplace(C->cc, x);  iblk++;
	
	for (int j=0; j<3; ++j) {
		for (int i=0; i<P->n_blk; ++i)
			x = MLN(IDX2NAME(iblk++), mlb_sdtae_block(C, x, P->ch_inner));
		x = ggml_upscale(C->cc, x, 2);  iblk++;
		x = MLN(IDX2NAME(iblk++), mlb_nn_conv2d(C, x,
			P->ch_inner, 3,3, 1,1, 1,1, 1,1, false));
	}
	
	x = MLN(IDX2NAME(iblk++), mlb_sdtae_block(C, x, P->ch_inner));
	x = MLN(IDX2NAME(iblk++), mlb_nn_conv2d(C, x,
		P->ch_x, 3,3, 1,1, 1,1, 1,1, true));

	return x;
}

int sdtae_encode(MLCtx* C, const SdTaeParams* P,
	const LocalTensor* img, LocalTensor* latent)
{
	int R=1;
	
	const int f = 8;  //latent to image scale (8 for SD)
	if (!(img->n[0]%f==0 && img->n[1]%f==0 && img->n[2]==3 && img->n[3]==1))
		ERROR_LOG(-1, "invalid input image shape: " LT_SHAPE_FMT,
			LT_SHAPE_UNPACK(*img));
	
	mlctx_begin(C, "TAE encode");
	
	MLTensor *input = mlctx_input_new(C, "img", GGML_TYPE_F32,
		LT_SHAPE_UNPACK(*img) );
	MLTensor *output = mlb_sdtae_encoder(C, input, P);
	mlctx_tensor_add(C, "encoder.layers", output);

	TRY( mlctx_run(C, latent, img) );

end:
	return R;
}

int sdtae_decode(MLCtx* C, const SdTaeParams* P,
	const LocalTensor* latent, LocalTensor* img)
{
	int R=1;

	TRY( ltensor_shape_check_log(latent, "latent", 0,0,4,1) );
	
	mlctx_begin(C, "TAE decode");
	
	MLTensor *input = mlctx_input_new(C, "latent", GGML_TYPE_F32,
		LT_SHAPE_UNPACK(*latent));
	MLTensor *output = mlb_sdtae_decoder(C, input, P);
	mlctx_tensor_add(C, "decoder.layers", output);

	TRY( mlctx_run(C, img, latent) );

end:
	return R;
}
