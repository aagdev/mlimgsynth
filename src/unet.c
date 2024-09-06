/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 */
#include "unet.h"
#include "mlblock_nn.h"
#include "ccommon/bisect.h"
#include "ccommon/timing.h"
#include <math.h>

#define T  true
#define MLN(NAME,X)  mlctx_tensor_add(C, (NAME), (X))

// The GGML scheduler have problems with inplace operations (2024-07-13)
#if USE_GGML_SCHED
	#define ggml_silu_inplace  ggml_silu
#endif

float g_log_sigmas_sd[1000];

const UnetParams g_unet_sd1 = {
	.n_ch_in		= 4,
	.n_ch_out		= 4,
	.n_res_blk		= 2,
	.attn_res		= {4,2,1},
	.ch_mult		= {1,2,4,4},
	.transf_depth	= {1,1,1,1},
	.n_te			= 1280,
	.n_head			= 8,
	.n_ctx			= 768,
	.n_ch			= 320,

	.clip_norm		= true,
	
	.n_step_train   = 1000,
	.sigma_min      = 0.029167158f,
	.sigma_max      = 14.614641f,
	.log_sigmas     = g_log_sigmas_sd,
};

const UnetParams g_unet_sd2 = {
	.n_ch_in		= 4,
	.n_ch_out		= 4,
	.n_res_blk		= 2,
	.attn_res		= {4,2,1},
	.ch_mult		= {1,2,4,4},
	.transf_depth	= {1,1,1,1},
	.n_te			= 1280,
	.d_head			= 64,
	.n_ctx			= 1024,
	.n_ch			= 320,

	.clip_norm		= true,
	.vparam			= true,
	
	.n_step_train   = 1000,
	.sigma_min      = 0.029167158f,
	.sigma_max      = 14.614641f,
	.log_sigmas     = g_log_sigmas_sd,
};

const UnetParams g_unet_sdxl = {
	.n_ch_in		= 4,
	.n_ch_out		= 4,
	.n_res_blk		= 2,
	.attn_res		= {4,2},
	.ch_mult		= {1,2,4},
	.transf_depth	= {1,2,10},
	.n_te			= 1280,
	.d_head			= 64,
	.n_ctx			= 2048,
	.n_ch			= 320,
	.ch_adm_in		= 2816,

	.clip_norm		= false,
	.uncond_empty_zero = true,
	
	.n_step_train   = 1000,
	.sigma_min      = 0.029167158f,
	.sigma_max      = 14.614641f,
	.log_sigmas     = g_log_sigmas_sd,
};

/*const UnetParams g_unet_svd = {
	.n_ch_in		= 8,
	.n_ch_out		= 4,
	.n_res_blk		= 2,
	.attn_res		= {4,2},
	.ch_mult		= {1,2,4},
	.transf_depth	= {1,2,10},
	.n_te			= 1280,
	.d_head			= 64,
	.n_ctx			= 1024,
	.n_ch			= 320,
	.ch_adm_in		= 768,
	
	.n_step_train   = 1000,
	.sigma_min      = 0.029167158f,
	.sigma_max      = 14.614641f,
	.log_sigmas     = g_log_sigmas_sd,
};*/

static inline
bool static_vector_in(const int* svec, int v) {
	for (unsigned i=0; svec[i]; ++i) if (v == svec[i]) return true;
	return false;
}

MLTensor* mlb_spatial_transf(MLCtx* C, MLTensor* x, MLTensor* ctx,
	int d_embed, int d_head, int n_head, int n_depth)
{
	MLTensor *x0=x;
	char name[64];
	mlctx_block_begin(C);
	// x: [N, d_in, h, w]

	int w=x->ne[0], h=x->ne[1], ch_in=x->ne[2], n_batch=x->ne[3];
	if (!n_head)  n_head  = d_embed / d_head;
	if (!d_head)  d_head  = d_embed / n_head;
	if (!d_embed) d_embed = d_head * n_head;
	
	x = MLN("norm", mlb_nn_groupnorm32(C, x));
	x = MLN("proj_in", mlb_nn_conv2d(C, x, d_embed, 1,1, 1,1, 0,0, 1,1, T));
	// [N, d_embed, h, w]
	x = ggml_cont(C->cc, ggml_permute(C->cc, x, 1, 2, 0, 3));
	x = ggml_reshape_3d(C->cc, x, d_embed, w * h, n_batch);
	// [N, h * w, d_embed]

	for (int i=0; i<n_depth; ++i) {
		sprintf(name, "transformer_blocks.%d", i);
		x = MLN(name, mlb_basic_transf(C, x, ctx, d_embed, d_embed, n_head));
	}

	x = ggml_cont(C->cc, ggml_permute(C->cc, x, 1, 0, 2, 3));
	// [N, d_embed, h * w]
	x = ggml_reshape_4d(C->cc, x, w, h, d_embed, n_batch);
	// [N, d_embed, h, w]

	x = MLN("proj_out", mlb_nn_conv2d(C, x, ch_in, 1,1, 1,1, 0,0, 1,1, T));
	// [N, ch_in, h, w]

	x = ggml_add(C->cc, x, x0);
	return x;
}

MLTensor* mlb_unet__embed(MLCtx* C, MLTensor* time, MLTensor* label,
	const UnetParams* P)
{
	MLTensor *emb = ggml_timestep_embedding(C->cc, time, P->n_ch, 10000);
	// [N, n_ch]
	emb = MLN("time_embed.0", mlb_nn_linear(C, emb, P->n_te, T));
	emb = ggml_silu_inplace(C->cc, emb);	
	emb = MLN("time_embed.2", mlb_nn_linear(C, emb, P->n_te, T));
	// [N, n_te]
	
	if (P->ch_adm_in && label) {
		MLTensor *le = MLN("label_emb.0.0", mlb_nn_linear(C, label, P->n_te, T));
		le = ggml_silu_inplace(C->cc, le);	
		le = MLN("label_emb.0.2", mlb_nn_linear(C, le, P->n_te, T));
		emb = ggml_add(C->cc, emb, le);
	}

	return emb;
}

MLTensor* mlb_unet__in(MLCtx* C, MLTensor* x, MLTensor* emb, MLTensor* ctx,
	const UnetParams* P, MLTensor*** pstack)
{
	char name[64];

	x = MLN("input_blocks.0.0", mlb_nn_conv2d(C, x,
		P->n_ch, 3,3, 1,1, 1,1, 1,1, T));
	
	MLTensor ** stack = NULL;
	vec_push(stack, x);
	int im=0, i_blk=0, ds=1, ch=P->n_ch;
	for (; P->ch_mult[im]; ++im) {
		if (im) {
			ds *= 2;
			i_blk++;
			sprintf(name, "input_blocks.%d.0", i_blk);
			x = MLN(name, mlb_downsample(C, x, ch, false));
			vec_push(stack, x);
		}
		for (unsigned j=0; j<P->n_res_blk; ++j) {
			i_blk++;
			sprintf(name, "input_blocks.%d.0", i_blk);
			ch = P->n_ch * P->ch_mult[im];
			x = MLN(name, mlb_resnet(C, x, emb, ch));

			if (static_vector_in(P->attn_res, ds)) {
				sprintf(name, "input_blocks.%d.1", i_blk);
				x = MLN(name, mlb_spatial_transf(C, x, ctx,
					ch, P->d_head, P->n_head, P->transf_depth[im]));
			}
			vec_push(stack, x);
		}
	}
	
	*pstack = stack;
	return x;
}

MLTensor* mlb_unet__mid(MLCtx* C, MLTensor* x, MLTensor* emb, MLTensor* ctx,
	const UnetParams* P)
{
	int im=0;
	while (P->ch_mult[im+1]) im++;
	int ch = P->n_ch * P->ch_mult[im];

	x = MLN("middle_block.0", mlb_resnet(C, x, emb, ch));
	x = MLN("middle_block.1", mlb_spatial_transf(C, x, ctx,
		ch, P->d_head, P->n_head, P->transf_depth[im]));
	x = MLN("middle_block.2", mlb_resnet(C, x, emb, ch));
	return x;
}

MLTensor* mlb_unet__out(MLCtx* C, MLTensor* x, MLTensor* emb, MLTensor* ctx,
	const UnetParams* P, MLTensor*** pstack)
{
	char name[64];

	int im=0, ds=1;
	while (P->ch_mult[im+1]) { im++; ds*=2; }
	int ch = P->n_ch * P->ch_mult[im];
	
	MLTensor ** stack = *pstack;
	for (unsigned i_oblk=0; im>=0; --im) {
		for (unsigned j=0; j<P->n_res_blk+1; ++j, ++i_oblk) {
			assert(vec_count(stack) > 0);
			MLTensor *h = vec_pop(stack);
			x = ggml_concat(C->cc, x, h, 2);

			unsigned i_sub=0;
			ch = P->n_ch * P->ch_mult[im];
			sprintf(name, "output_blocks.%d.%d", i_oblk, i_sub++);
			x = MLN(name, mlb_resnet(C, x, emb, ch));

			if (static_vector_in(P->attn_res, ds)) {
				sprintf(name, "output_blocks.%d.%d", i_oblk, i_sub++);
				x = MLN(name, mlb_spatial_transf(C, x, ctx,
					ch, P->d_head, P->n_head, P->transf_depth[im]));
			}

			if (im != 0 && j == P->n_res_blk) {
				sprintf(name, "output_blocks.%d.%d", i_oblk, i_sub++);
				x = MLN(name, mlb_upsample(C, x, ch));
				ds /= 2;
			}
		}
	}
	assert(vec_count(stack) == 0);
	
	x = MLN("out.0", mlb_nn_groupnorm32(C, x));
	x = ggml_silu_inplace(C->cc, x);
	x = MLN("out.2", mlb_nn_conv2d(C, x,
		P->n_ch_out, 3,3, 1,1, 1,1, 1,1, T));

	return x;
}

MLTensor* mlb_unet_denoise(MLCtx* C, MLTensor* x, MLTensor* time, MLTensor* ctx,
	MLTensor* label, const UnetParams* P)
{
	//char name[64];
	// x: [N, n_ch_in, h, w]
	// tsteps: [N]
	// ctx: [N, n_token, n_embed]
	mlctx_block_begin(C);

	MLTensor *emb = mlb_unet__embed(C, time, label, P);
	MLTensor ** stack=NULL;
	x = mlb_unet__in(C, x, emb, ctx, P, &stack);
	x = mlb_unet__mid(C, x, emb, ctx, P);
	x = mlb_unet__out(C, x, emb, ctx, P, &stack);
	vec_free(stack);
	
	// [N, n_ch_out, h, w]
	return x;
}

void unet_params_init()
{
	if (g_log_sigmas_sd[0]) return;

	unsigned n=1000;
	double linear_start = 0.00085,
	       linear_end   = 0.0120,
	       b = sqrt(linear_start),
		   e = sqrt(linear_end),
		   f = (e - b) / (n - 1),
		   alpha_cumprod = 1.0;

	for (unsigned i=0; i<n; ++i) {
		double beta = b+f*i,
			   alpha = 1.0 - beta * beta;
		alpha_cumprod *= alpha;
		double sigma = sqrt((1 - alpha_cumprod) / alpha_cumprod);
		//printf("sigma_%d=%.8g\n", i, sigma);
		g_log_sigmas_sd[i] = log(sigma);
	}
}

float linear_interp(unsigned n, float* vec, float t)
{
	int ti = t;
	ccCLAMP(ti, 0, (int)n-1);
	float v1 = vec[ti],
	      v2 = ti+1<n ? vec[ti+1] : v1;
	return v1*(ti+1-t) + v2*(t-ti);
}

// Estimates to position where vec crosses v
float linear_est(unsigned n, float* vec, float v)
{
	assert( vec[0] < vec[n-1] );  //must be sorted ascending
	BISECT_RIGHT_DECL(found, idx, 0, n, SIGNg(vec[i_] - v) );
	if (idx+1 >= n) return n-1;
	float v1 = vec[idx], v2 = vec[idx+1];
	return idx + (v - v1) / (v2 - v1);
}

float unet_sigma_to_t(const UnetParams* P, float sigma)
{
	float ls=log(sigma);
	return linear_est(P->n_step_train, P->log_sigmas, ls);
}

float unet_t_to_sigma(const UnetParams* P, float t)
{
	float ls = linear_interp(P->n_step_train, P->log_sigmas, t);
	return exp(ls);
}

int unet_denoise_init(UnetState* S, MLCtx* C, const UnetParams* P,
	unsigned lw, unsigned lh, bool split)
{
	int R=1;

	unet_params_init();  //global
		
	C->c.n_tensor_max = 10240;

	if (!split) {
		// Prepare computation
		C->c.multi_compute = true;
		mlctx_begin(C, "UNet");

		MLTensor *t_x, *t_t, *t_c, *t_l=NULL;
		t_x = mlctx_input_new(C, "x", GGML_TYPE_F32, lw, lh, 4, 1);
		t_t = mlctx_input_new(C, "t", GGML_TYPE_F32, 1,1,1,1);
		t_c = mlctx_input_new(C, "c", GGML_TYPE_F32, P->n_ctx, 77, 1, 1);
		if (P->ch_adm_in)
			t_l = mlctx_input_new(C, "l", GGML_TYPE_F32, P->ch_adm_in, 1,1,1);
		mlb_unet_denoise(C, t_x, t_t, t_c, t_l, P);
		TRY( mlctx_prep(C) );
	}

	S->ctx = C;
	S->par = P;
	S->split = split;

end:
	return R;
}

int unet_compute(MLCtx* C, const UnetParams* P,
	const LocalTensor* x, const LocalTensor* cond, const LocalTensor* label,
	float t, LocalTensor* dx)
{
	int R=1;

	// Set input
	ltensor_to_backend(x, C->inputs[0]);
	ggml_backend_tensor_set(C->inputs[1], &t, 0, sizeof(t));
	ltensor_to_backend(cond, C->inputs[2]);
	if (P->ch_adm_in) ltensor_to_backend(label, C->inputs[3]);
		
	// Compute
	TRY( mlctx_compute(C) );

	// Get output
	ltensor_from_backend(dx, C->result);

end:
	return R;
}

int unet_compute_split(MLCtx* C, const UnetParams* P,
	const LocalTensor* x, const LocalTensor* cond, const LocalTensor* label,
	float t, LocalTensor* dx)
{
	int R=1;
	MLTensor *t_x, *t_t, *t_c, *t_l=NULL, *t_e, *out,
	         **tstack=NULL;
	LocalTensor *lstack=NULL, emb={0};

	// First half
	mlctx_begin(C, "UNet 1/2");

	t_x = mlctx_input_new(C, "x", GGML_TYPE_F32, x->s[0], x->s[1], 4, 1);
	t_t = mlctx_input_new(C, "t", GGML_TYPE_F32, 1,1,1,1);
	t_c = mlctx_input_new(C, "c", GGML_TYPE_F32, P->n_ctx, 77, 1, 1);
	if (P->ch_adm_in)
		t_l = mlctx_input_new(C, "l", GGML_TYPE_F32, P->ch_adm_in, 1,1,1);
	
	mlctx_block_begin(C);
	t_e = mlb_unet__embed(C, t_t, t_l, P);
	out = mlb_unet__in(C, t_x, t_e, t_c, P, &tstack);
	out = mlb_unet__mid(C, out, t_e, t_c, P);
	ggml_set_output(t_e);
	vec_for(tstack,i,0) ggml_set_output(tstack[i]);
	TRY( mlctx_prep(C) );
	
	ltensor_to_backend(x, t_x);
	ggml_backend_tensor_set(t_t, &t, 0, sizeof(t));
	ltensor_to_backend(cond, t_c);
	if (t_l) ltensor_to_backend(label, t_l);

	TRY( mlctx_compute(C) );

	ltensor_from_backend(dx, out);
	ltensor_from_backend(&emb, t_e);
	vec_resize_zero(lstack, vec_count(tstack));
	vec_for(lstack,i,0) ltensor_from_backend(&lstack[i], tstack[i]);

	// Second half
	mlctx_begin(C, "UNet 2/2");
	
	t_x = mlctx_input_new(C, "x", GGML_TYPE_F32, LT_SHAPE_UNPACK(*dx));
	t_e = mlctx_input_new(C, "e", GGML_TYPE_F32, LT_SHAPE_UNPACK(emb));
	t_c = mlctx_input_new(C, "c", GGML_TYPE_F32, P->n_ctx, 77, 1, 1);
	vec_for(lstack,i,0)
		tstack[i] = mlctx_input_new(C, "skip", GGML_TYPE_F32,
			LT_SHAPE_UNPACK(lstack[i]));

	mlctx_block_begin(C);
	out = mlb_unet__out(C, t_x, t_e, t_c, P, &tstack);
	TRY( mlctx_prep(C) );
	
	ltensor_to_backend(dx, t_x);
	ltensor_to_backend(&emb, t_e);
	ltensor_to_backend(cond, t_c);
	vec_for(lstack,i,0) ltensor_to_backend(&lstack[i], tstack[i]);
	
	TRY( mlctx_compute(C) );
	
	ltensor_from_backend(dx, out);

end:
	vec_for(lstack,i,0) ltensor_free(&lstack[i]);
	vec_free(lstack);
	ltensor_free(&emb);
	vec_free(tstack);
	mlctx_free(C);
	return R;
}

int unet_denoise_run(UnetState* S,
	const LocalTensor* x, const LocalTensor* cond, const LocalTensor* label,
	float sigma, LocalTensor* dx)
{
	int R=1;
	
	ltensor_resize_like(dx, x);
	
	float t = unet_sigma_to_t(S->par, sigma);

	// Scale input
	float c_in = 1 / sqrt(sigma*sigma + 1);
	ltensor_for(*dx,i,0) dx->d[i] = x->d[i] * c_in;

	// Compute
	if (!S->split || S->nfe > 0) S->ctx->c.quiet = true;
	double t_comp = timing_time();
	if (S->split) {
		TRY( unet_compute_split(S->ctx, S->par, dx, cond, label, t, dx) );
	} else {
		TRY( unet_compute(S->ctx, S->par, dx, cond, label, t, dx) );
	}
	t_comp = timing_time() - t_comp;
	//log_debug("dx  %.6e", ltensor_mean(dx));
	S->nfe++;
	log_info("Step %u/%u NFE %d done {%.3fs}",
		S->i_step+1, S->n_step, S->nfe, t_comp);
	
	ltensor_for(*dx,i,0)
		if (!isfinite(dx->d[i])) ERROR_LOG(-1, "NaN found in UNet output");

	// Scale output
	if (S->par->vparam) {
		float c_skip = sigma / (sigma*sigma + 1),
		      c_out = 1 / sqrt(sigma*sigma + 1);
		ltensor_for(*dx,i,0) dx->d[i] = dx->d[i] * c_out + x->d[i] * c_skip;
	}

end:
	S->ctx->c.quiet = false;
	return R;
}
