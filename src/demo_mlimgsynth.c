/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * Demostration of the capabilities of the MLImgSynth library.
 */
#include "mlimgsynth.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define error(...) do { \
	printf("ERROR "); \
	printf(__VA_ARGS__); \
	printf("\n"); \
	exit(1); \
} while (0)

#define log(...) do { \
	printf(__VA_ARGS__); \
	printf("\n"); \
} while (0)
	
void img_save(MLIS_Ctx* ctx, const char* name)
{
	char buffer[128];

	const MLIS_Image *img = mlis_image_get(ctx, 0);
	const char *info = mlis_infotext_get(ctx, 0);

	log("Saving...");

	sprintf(buffer, "%s.ppm", name);
	FILE *f = fopen(buffer, "w");
	fprintf(f, "P6 %u %u 255\n", img->w, img->h);
	fwrite(img->d, 1, img->sz, f);
	fclose(f);

	sprintf(buffer, "%s.txt", name);
	f = fopen(buffer, "w");
	fwrite(info, 1, strlen(info), f);
	fclose(f);
}

void demo_txt2img(MLIS_Ctx* ctx)
{
	log("txt2img");
	mlis_option_set(ctx, MLIS_OPT_PROMPT,
		"a photograph of an astronaut riding a horse in a grassland");

	mlis_generate(ctx);
	
	img_save(ctx, "demo_txt2img");
}

void demo_img2img(MLIS_Ctx* ctx)
{
	log("img2img");
	mlis_option_set(ctx, MLIS_OPT_PROMPT,
		"a photograph of an astronaut riding a horse in a forest");
	mlis_option_set(ctx, MLIS_OPT_F_T_INI, 0.70);  // Strength
	
	// For this example we just use the previously generated image
	const MLIS_Image *img = mlis_image_get(ctx, 0);
	mlis_option_set(ctx, MLIS_OPT_IMAGE, img);

	mlis_generate(ctx);

	img_save(ctx, "demo_img2img");
}

void demo_inpaint(MLIS_Ctx* ctx)
{
	log("inpaint");
	mlis_option_set(ctx, MLIS_OPT_PROMPT, "a pile of gold coins");
	mlis_option_set(ctx, MLIS_OPT_NO_DECODE, 1);

	mlis_generate(ctx);
	
	mlis_option_set(ctx, MLIS_OPT_NO_DECODE, 0);

	// Creates a circular mask for latent space
	MLIS_Tensor *latent = mlis_tensor_get(ctx, MLIS_TENSOR_LATENT);
	MLIS_Tensor *lmask = mlis_tensor_get(ctx, MLIS_TENSOR_LMASK);
	mlis_tensor_resize_like(lmask, latent);
	int r0 = lmask->n[0] / 2;  // Radius
	int r1 = lmask->n[1] / 2;
	mlis_tensor_for(*lmask, i) {
		lmask->d[ip] = ((i0-r0)*(i0-r0) + (i1-r1)*(i1-r1)) > r1*r1;
	}
	
	mlis_option_set(ctx, MLIS_OPT_PROMPT, "a red dragon on a pile of gold coins");
	mlis_option_set(ctx, MLIS_OPT_F_T_INI, 0.70);
	mlis_option_set(ctx, MLIS_OPT_TENSOR_USE_FLAGS,
		MLIS_TUF_LATENT | MLIS_TUF_LMASK);
	
	mlis_generate(ctx);

	img_save(ctx, "demo_inpaint");
}

void error_handler(void*, MLIS_Ctx* ctx, const MLIS_ErrorInfo* ei)
{
	error("mlis error 0x%x: %s", -ei->code, ei->desc);
}

int progress_callback(void*, MLIS_Ctx* ctx, const MLIS_Progress* prg)
{
	double etc = -1;
	if (1 < prg->step) etc = (prg->step_end - prg->step) * prg->step_time;
	log("%s %d/%d nfe=%d {%.3fs} ETC %.0fs",
		mlis_stage_str(prg->stage), prg->step, prg->step_end, prg->nfe,
		prg->step_time, etc);
	return 0;  //continue
}

int main(int argc, char* argv[])
{
	if (argc != 2)
		error("Usage: %s [MODEL FILE PATH]", argv[0]);
	
	log("Initializing...");
	MLIS_Ctx *ctx = mlis_ctx_create();
	mlis_option_set(ctx, MLIS_OPT_ERROR_HANDLER, error_handler, NULL);
	mlis_option_set(ctx, MLIS_OPT_CALLBACK, progress_callback, NULL);
	mlis_option_set(ctx, MLIS_OPT_MODEL, argv[1]);
	
	// If you do not set the following options, default values will be used.
	mlis_option_set(ctx, MLIS_OPT_IMAGE_DIM, 768, 512);
	mlis_option_set(ctx, MLIS_OPT_SEED, 42);
	mlis_option_set(ctx, MLIS_OPT_METHOD, MLIS_METHOD_EULER);
	mlis_option_set(ctx, MLIS_OPT_SCHEDULER, MLIS_SCHED_UNIFORM);
	mlis_option_set(ctx, MLIS_OPT_STEPS, 20);
	// Be sure to use floating point numbers with options that require it.
	mlis_option_set(ctx, MLIS_OPT_CFG_SCALE, 7.0);
	mlis_option_set(ctx, MLIS_OPT_S_ANCESTRAL, 1.0);
	// You can also set options using strings.
	mlis_option_set_str(ctx, "image_dim", "768,512");
	//mlis_option_set(ctx, MLIS_OPT_LORA, lora_path, 1.0);

	// Initialized the backend and load the model header
	// This is not required, but it can be useful to catch errors early.
	mlis_setup(ctx);

	demo_txt2img(ctx);
	demo_img2img(ctx);
	demo_inpaint(ctx);

	log("End");
	mlis_ctx_destroy(&ctx);
	return 0;
}
