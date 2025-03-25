/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * MLImgSynth command line utility.
 * Synthetize images with AI from the command line.
 */
#include "mlimgsynth.h"
#include "localtensor.h"
#include "ccommon/vector.h"
#include "ccommon/stream.h"
#include "ccommon/logging.h"
#include "ccommon/image.h"
#include "ccommon/image_io.h"
#include "ccommon/fsutil.h"
#include <stdlib.h>
#include <math.h>

#define F_MIB  (1.0 / (1024.0*1024.0))
#define F_GIB  (1.0 / (1024.0*1024.0*1024.0))

#define APP_NAME_VERSION "mlimgsynth v" MLIS_VERSION_STR

const char version_string[] = APP_NAME_VERSION "\n";

const char help_string[] =
APP_NAME_VERSION "\n"
"Image synthesis using AI.\n"
"Currently Stable Diffusion 1, 2 and XL are implemented.\n"
"\n"
"Usage: mlimgsynth [COMMAND] [OPTIONS]\n"
"\n"
"Commands:\n"
"  generate             Generate an image.\n"
"  list-backends        List available GGML backends.\n"
"  vae-encode           Encode an image to a latent.\n"
"  vae-decode           Decode a latent to an image.\n"
"  vae-test             Encode and decode an image.\n"
"  clip-encode          Encode a prompt with the CLIP tokenizer and model.\n"
"  check                Checks that all the operations (models) are working.\n"
"\n"
"Generation options:\n"
"  -p --prompt TEXT     Prompt for text conditioning.\n"
"  -n --nprompt TEXT    Negative prompt.\n"
"  -d --image-dim W,H   Image dimensions (width and height).\n"
"                       Default: 512x512 (SD1), 768x768 (SD2), 1024x1024 (SDXL).\n"
"  -i --input PATH      Input image for img2img or inpainting.\n"
"                       An alpha channel will be used as the mask for inpaiting.\n"
"  --imask PATH         Input image mask for inpainting.\n"
"  --ilatent PATH       Input latent tensor.\n"
"  --ilmask PATH        Input latent mask tensor.\n"
"  -o --output PATH     Output image path.\n"
"\n"
"Models and backend:\n"
"  -m --model PATH      Model file.\n"
"  --tae PATH           TAE model file. Enables TAE instead of VAE.\n"
"  --lora PATH:MULT     Apply the LoRA from PATH with multiplier MULT.\n"
"         PATH          The multiplier is optional.\n"
"                       This option can be used multiple times.\n"
"  --lora-dir PATH      Directory to search for LoRA's found in the prompt as:\n"
"                       <lora:NAME:MULT>\n"
"                       where NAME is the file name without extension.\n"
"  -b --backend NAME    Backend for computation (passed to GGML).\n"
"  -t --threads INT     Number of threads to use in the CPU backend.\n"
"  --unet-split BOOL    Split each unet steps to reduce memory usage.\n"
"  --vae-tile INT       Encode and decode images using tiles of NxN pixels.\n"
"                       Reduces memory usage. On doubt, try 512.\n"
//"  --type NAME          Convert the weight to this tensor type.\n"
//"                       Useful to quantize and reduce RAM usage (try q8_0).\n"
"  --dump-model         Dumps model tensors and graphs.\n"
"  --dump-lora          Dumps lora's tensors and graphs.\n"
"\n"
"Sampling:\n"
"  -S --seed INT        RNG seed.\n"
"  -s --steps INT       Denoising steps with UNet.\n"
"  --method NAME        Sampling method (default taylor3).\n"
"                       euler, euler_a, heun, taylor3, dpm++2m, dpm++2s, dpm++2s_a.\n"
"                       The _a variants are just a shortcut for --s-ancestral 1.\n"
"  --scheduler NAME     Sampling scheduler: uniform (default), karras.\n"
"  --s-noise FLOAT      Level of noise injection at each sampling step (try 1).\n"
"  --s-ancestral FLOAT  Ancestral sampling noise level (try 1).\n"
"  --cfg-scale FLOAT    Enables and sets the scale of the classifier-free guidance\n"
"                       (default: 1).\n"
"  --clip-skip INT      Number of CLIP layers to skip.\n"
"                       Default: 1 (SD1), 2 (SD2/XL).\n"
"  --f-t-ini FLOAT      Initial time factor (default 1).\n"
"                       Use it to control the strength in img2img.\n"
"  --f-t-end FLOAT      End time factor (default 0).\n"
"\n"
"Output control:\n"
"  -v --verbose         Verbose: increases information output. Can be repeated.\n"
"  -q --quiet           Output only errors.\n"
"  -s --silent          No output to terminal.\n"
"  --debug              Enables debug output.\n"
"  -h --help            Print this message and exit.\n"
"  -V --version         Print the version and exit.\n"
;

struct arg_parse_short_opt_t {
	char c;
	const char *name;
};

enum {
	ARG_PARSE_END = 10,
	ARG_PARSE_NEXT_USED = 11,
};

int arg_parse(int argc, char* argv[], int npos,
	const struct arg_parse_short_opt_t* sopt,
	int (*callback)(void*, const char*, const char*), void* userdata)
{
	int r, i, j, k, ipos=0;
	char buf[16];

	for (i=1; i<argc; ++i) {
		char * arg = argv[i];
		if (arg[0] == '-' && arg[1] == '-') {
			char *next = (i+1 < argc) ? argv[i+1] : "";
			TRYR( r = callback(userdata, arg+2, next) );
			if (r == ARG_PARSE_END) return 0;
			if (r == ARG_PARSE_NEXT_USED) i++;
		}
		else if (arg[0] == '-') {
			char opt;
			for (j=1; (opt = arg[j]); ++j) {
				char *next = (i+1 < argc) ? argv[i+1] : "";
				for (k=0; sopt[k].c && opt != sopt[k].c; ++k) ;
				if (sopt[k].c) {
					TRYR( r = callback(userdata, sopt[k].name, next) );
					if (r == ARG_PARSE_END) return 0;
					if (r == ARG_PARSE_NEXT_USED) i++;
				} else {
					log_error("Unknown short option '%c'", opt);
					return -1;
				}
			}
		}
		else if (ipos < npos) {
			sprintf(buf, "POS%u", ipos);
			TRYR( r = callback(userdata, buf, arg) );
			if (r == ARG_PARSE_END) return 0;
		}
		else {
			log_error("Excess of positional arguments");
			return -1;
		}
	}

	return 1;
}

const struct arg_parse_short_opt_t short_options[] = {
	{ 'h', "help" },
	{ 'V', "version" },
	{ 'v', "verbose" },
	{ 'q', "quiet" },
	{ 'b', "backend" },
	{ 'm', "model" },
	{ 'p', "prompt" },
	{ 'n', "nprompt" },
	{ 'd', "image-dim" },
	{ 's', "steps" },
	{ 'S', "seed" },
	{ 't', "threads" },
	{ 'i', "input" },
	{ 'o', "output" },
	{0}
};

typedef struct MLIS_CliOptions {
	const char *cmd,
		*path_input_image, *path_input_mask,
		*path_input_latent, *path_input_lmask,
		*path_output_image,
		*path_output_latent;
	
	MLIS_Ctx *mlis_ctx;
} MLIS_CliOptions;

int mlis_cli_opt_set(void* userdata, const char* optname, const char* next_value)
{
	MLIS_CliOptions* opt = userdata;
	MLIS_Ctx* ctx = opt->mlis_ctx;

	log_debug("opt '%s' '%s'", optname, next_value);

#define IF_OPT(NAME)\
	else if (!strcmp(optname, NAME))

	if (!optname) ;
	IF_OPT("help") {
		return ARG_PARSE_END;
	}
	IF_OPT("version") {
		puts(version_string);
		return ARG_PARSE_END;
	}
	IF_OPT("debug") {
		log_level_set(LOG_LVL_DEBUG);
		mlis_option_set(ctx, MLIS_OPT_LOG_LEVEL, MLIS_LOGLVL_DEBUG);
	}
	IF_OPT("verbose") {
		log_level_inc(+LOG_LVL_STEP);
		mlis_option_set(ctx, MLIS_OPT_LOG_LEVEL, MLIS_LOGLVL__INCREASE);
	}
	IF_OPT("quiet") {
		log_level_set(LOG_LVL_ERROR);
		mlis_option_set(ctx, MLIS_OPT_LOG_LEVEL, MLIS_LOGLVL_ERROR);
	}
	IF_OPT("silent") {
		log_level_set(LOG_LVL_NONE);
		mlis_option_set(ctx, MLIS_OPT_LOG_LEVEL, MLIS_LOGLVL_NONE);
	}
	IF_OPT("input") {
		opt->path_input_image = next_value;
		return ARG_PARSE_NEXT_USED;
	}
	IF_OPT("imask") {
		opt->path_input_mask = next_value;
		return ARG_PARSE_NEXT_USED;
	}
	IF_OPT("ilatent") {
		opt->path_input_latent = next_value;
		return ARG_PARSE_NEXT_USED;
	}
	IF_OPT("ilmask") {
		opt->path_input_lmask = next_value;
		return ARG_PARSE_NEXT_USED;
	}
	IF_OPT("output") {
		opt->path_output_image = next_value;
		return ARG_PARSE_NEXT_USED;
	}
	IF_OPT("olatent") {
		opt->path_output_latent = next_value;
		return ARG_PARSE_NEXT_USED;
	}
	IF_OPT("POS0") {
		opt->cmd = next_value;
	}
	//TODO: other paths
	else {
		int r = mlis_option_set_str(ctx, optname, next_value);
		if (r < 0) {
			log_error("failed to set option '%s': %s", optname,
				mlis_errstr_get(ctx));
			return -1;
		}
		return ARG_PARSE_NEXT_USED;
	}

	return 1;
}

int mlis_cli_backends_print(MLIS_CliOptions* opt, MLIS_Ctx* ctx)
{
	Stream out={0};
	TRYR( stream_open_std(&out, STREAM_STD_OUT, 0) );

	const MLIS_BackendInfo *bi;
	for (unsigned idx=0; (bi = mlis_backend_info_get(ctx, idx, 0)); ++idx)
	{
		stream_printf(&out, "%s\n", bi->name);
		for (unsigned idev=0; idev < bi->n_dev; ++idev)
		{
			stream_printf(&out, "\t%s '%s' %.1f/%.1fGiB\n",
				bi->devs[idev].name,
				bi->devs[idev].desc,
				bi->devs[idev].mem_free * F_GIB,
				bi->devs[idev].mem_total * F_GIB );
			}
	}

	stream_close(&out, 0);
	return 1;
}

static inline
Image mlis_image_to_image(const MLIS_Image* img)
{
	return (Image){ .data=img->d, .w=img->w, .h=img->h, .bypp=img->c,
		.pitch = img->w*img->c,
		.format = img->c == 3 ? IMG_FORMAT_RGB :
			img->c == 4 ? IMG_FORMAT_RGBA : IMG_FORMAT_GRAY };
}

static inline
MLIS_Image mlis_image_from_image(const Image* img)
{
	return (MLIS_Image){ .d=img->data, .w=img->w, .h=img->h, .c=img->bypp,
		.sz=img->pitch*img->h};
}

#define CLI_OPEN_PIPE 2

int cli_path_pipe_is(const char* path)
{
	return !strcmp(path, "-");
}

int cli_stream_open(Stream* stm, const char* path, int oflags)
{
	int R=1;

	if (cli_path_pipe_is(path)) {
		bool b_write = (oflags & SOF_CREATE || oflags & SOF_WRITE);
		int std = b_write ? STREAM_STD_OUT : STREAM_STD_IN;
		TRY_LOG( stream_open_std(stm, std, oflags),
			"Cannot open %s", b_write ? "stdout" : "stdin");
		R = CLI_OPEN_PIPE;
	}
	else {
		TRY_LOG( stream_open_file(stm, path, oflags),
			"Cannot open file '%s'", path);
	}

end:
	return R;
}

int cli_image_load(Image* img, const char* path)
{
	int R=1;
	Stream stm={0};
	ImageIO imgio={0};
	
	log_debug("Loading image from '%s'", path);
	TRY( cli_stream_open(&stm, path, SOF_READ) );
	TRY( imgio_open_stream(&imgio, &stm, 0, 0) );
	TRY( imgio_load(&imgio, img) );

end:
	if (R<0) log_error("Cannot load image from '%s'", path);
	imgio_free(&imgio);
	stream_close(&stm, 0);
	return R;
}

int cli_image_save(const Image* img, const char* info_text, const char* path)
{
	int R=1;
	Stream stm={0};
	ImageIO imgio={0};
	DynStr tmps=NULL;
	
	log_debug("Saving image to '%s'", path);

	const ImageCodec* codec;
	if (cli_path_pipe_is(path)) {
		codec = img_codec_by_name("pnm");
	}
	else {
		codec = img_codec_detect_filename(path, IMG_OF_SAVE);
		if (!codec)
			ERROR_LOG(-1, "Cannot find an image codec to save '%s'", path);
	}

	TRY( cli_stream_open(&stm, path, SOF_CREATE) );

	TRY( imgio_open_stream(&imgio, &stm, IMG_OF_SAVE, codec) );

	if (info_text) {
		const char *info_key = "parameters";
		dstr_copyz(tmps, info_key);
		dstr_push(tmps, '\0');
		dstr_appendz(tmps, info_text);
		int r = imgio_value_set(&imgio, IMG_VALUE_METADATA, tmps,
				dstr_count(tmps)+1);
		if (r<0)
			log_warning("Cannot write '%s' in '%s'", info_key, path);
	}

	TRY( imgio_save(&imgio, img) );

end:
	if (R<0) log_error("Cannot save image to '%s'", path );
	dstr_free(tmps);
	imgio_free(&imgio);
	stream_close(&stm, 0);
	return R;
}

int cli_tensor_load(MLIS_Tensor* ten, const char* path)
{
	int R=1;
	Stream stm={0};
	
	log_debug("Loading tensor from '%s'", path);
	TRY( cli_stream_open(&stm, path, SOF_READ) );
	TRY( ltensor_load_stream((LocalTensor*)ten, &stm) );

end:
	if (R<0) log_error("Cannot load tensor from '%s'", path);
	stream_close(&stm, 0);
	return R;
}

int cli_tensor_save(MLIS_Tensor* ten, const char* path)
{
	int R=1;
	Stream stm={0};
	
	log_debug("Saving tensor from '%s'", path);
	TRY( cli_stream_open(&stm, path, SOF_CREATE) );
	TRY( ltensor_save_stream((LocalTensor*)ten, &stm) );

end:
	if (R<0) log_error("Cannot save tensor to '%s'", path);
	stream_close(&stm, 0);
	return R;
}

static
void error_handler(void* user, MLIS_Ctx* ctx, const MLIS_ErrorInfo* ei)
{
	log_error("mlis error 0x%x: %s", -ei->code, ei->desc);
	exit(1);  //TODO: ok?
}

static
int progress_callback(void* user, MLIS_Ctx* ctx, const MLIS_Progress* prg)
{
	MLIS_CliOptions *opt = user;
	const char *path;

	// Print progress
	if (log_line_begin(LOG_LVL_INFO)) {
		log_line_strf("%s %d/%d {%.3fs}",
			mlis_stage_str(prg->stage), prg->step, prg->step_end, prg->step_time);
		if (prg->stage == MLIS_STAGE_DENOISE) {
			log_line_strf(" nfe:%d", prg->nfe);
		}
		if (1 < prg->step && prg->step < prg->step_end) {
			double etc = (prg->step_end - prg->step) * prg->step_time;
			log_line_strf(" etc:%.0fs", etc);
		}
		log_line_end();
	}

	// End of conditinings encoding
	if (prg->stage == MLIS_STAGE_COND_ENCODE && prg->step == prg->step_end) {
		//TODO: save tensors for debuging
	}

	// End of image encode
	if (prg->stage == MLIS_STAGE_IMAGE_ENCODE && prg->step == prg->step_end) {
		//TODO: save latent for debuging
	}
	
	// End of denoising
	if (prg->stage == MLIS_STAGE_DENOISE && prg->step == prg->step_end) {
		// Save encoded output image
		if ((path = opt->path_output_latent)) {
			MLIS_Tensor *latent = mlis_tensor_get(ctx, MLIS_TENSOR_LATENT);
			TRYR( cli_tensor_save(latent, path) );
		}
	}

	// End of image decode
	if (prg->stage == MLIS_STAGE_IMAGE_DECODE && prg->step == prg->step_end) {
	}
	
	return 0;  //continue
}

int mlis_cli_generate(MLIS_CliOptions* opt, MLIS_Ctx* ctx)
{
	int R=1;
	int tuflags=0;
	const char *path;
	Image image={0};

	// Load input image for img2img
	if ((path = opt->path_input_image)) {
		TRY( cli_image_load(&image, path) );
		MLIS_Image img = mlis_image_from_image(&image);
		mlis_option_set(ctx, MLIS_OPT_IMAGE, &img);
		tuflags |= MLIS_TUF_IMAGE;  // Need only if a tensor is set
	}

	// Load input image mask for img2img / inpainting
	if ((path = opt->path_input_mask)) {
		TRY( cli_image_load(&image, path) );
		MLIS_Image img = mlis_image_from_image(&image);
		mlis_option_set(ctx, MLIS_OPT_IMAGE_MASK, &img);
		tuflags |= MLIS_TUF_MASK;  // Need only if a tensor is set
	}

	// Load input latent for img2img
	if ((path = opt->path_input_latent)) {
		MLIS_Tensor *latent = mlis_tensor_get(ctx, MLIS_TENSOR_LATENT);
		TRY( cli_tensor_load(latent, path) );
		tuflags |= MLIS_TUF_LATENT;
	}

	// Load input latent mask for img2img / inpainting
	if ((path = opt->path_input_lmask)) {
		MLIS_Tensor *lmask = mlis_tensor_get(ctx, MLIS_TENSOR_LMASK);
		TRY( cli_tensor_load(lmask, path) );
		tuflags |= MLIS_TUF_LMASK;
	}
	
	if (tuflags)
		mlis_option_set(ctx, MLIS_OPT_TENSOR_USE_FLAGS, tuflags);

	mlis_generate(ctx);
	
	// Save output image
	if ((path = opt->path_output_image)) {
		MLIS_Image *img = mlis_image_get(ctx, 0);
		const char *info = mlis_infotext_get(ctx, 0);
		Image image = mlis_image_to_image(img);
		TRY( cli_image_save(&image, info, path) );
	}

end:
	img_free(&image);
	return R;
}

int mlis_cli_vae_cmd(MLIS_CliOptions* opt, MLIS_Ctx* ctx, bool encode, bool decode)
{
	int R=1;
	Image image={0};
	MLIS_Tensor t_orig_img={0};
	const char *path;
		
	MLIS_Tensor *t_image  = mlis_tensor_get(ctx, MLIS_TENSOR_IMAGE);
	MLIS_Tensor *t_latent = mlis_tensor_get(ctx, MLIS_TENSOR_LATENT);

	if (encode) {
		if (!(path = opt->path_input_image))
			ERROR_LOG(-1, "You must set the input image path.");

		TRY( cli_image_load(&image, path) );
		MLIS_Image tmpimg = mlis_image_from_image(&image);
		mlis_option_set(ctx, MLIS_OPT_IMAGE, &tmpimg);
		assert( t_image->n[0] == image.w && t_image->n[1] == image.h );

		mlis_tensor_copy(&t_orig_img, t_image);

		mlis_image_encode(ctx, t_image, t_latent, 0);
		
		if ((path = opt->path_output_latent)) {
			TRY( cli_tensor_save(t_latent, path) );
		}
	}
	else {
		if (!(path = opt->path_input_latent))
			ERROR_LOG(-1, "You must set the input latent path.");
		
		TRY( cli_tensor_load(t_latent, path) );
	}

	if (decode) {
		mlis_image_decode(ctx, t_latent, t_image, 0);
		
		if ((path = opt->path_output_image)) {
			MLIS_Image *img = mlis_image_get(ctx, 0);
			Image tmpimg = mlis_image_to_image(img);
			TRY( cli_image_save(&tmpimg, NULL, path) );
		}
	}

	if (encode && decode && log_level_check(LOG_LVL_INFO)) {
		// Calculate error
		double mse=0;
		mlis_tensor_for(*t_image, i) {
			double e = (t_orig_img.d[ip] - t_image->d[ip]);
			mse += e*e;
		}
		mse = sqrt(mse / mlis_tensor_count(t_image));
		log_info("Image encode/decode mse: %.3f", mse);
	}

end:
	mlis_tensor_free(&t_orig_img);
	img_free(&image);
	return R;
}

int mlis_cli_clip_cmd(MLIS_CliOptions* opt, MLIS_Ctx* ctx)
{
	int R=1;
	ERROR_LOG(-1, "not implemented");
	
	//mlis_option_get(ctx, MLIS_OPT_PROMPT, &prompt);  //TODO
	//mlis_clip_text_encode(ctx, prompt, &t_embed, &t_feat, 0, 0);

end:
	return R;
}

int mlis_cli_check(MLIS_CliOptions* opt, MLIS_Ctx* ctx)
{
	int R=1;
	ERROR_LOG(-1, "not implemented");
end:
	return R;
}

int main(int argc, char* argv[])
{
	int R=0, r;
	MLIS_Ctx *ctx = mlis_ctx_create();
	MLIS_CliOptions opt={ .mlis_ctx=ctx };

	// Set the auxiliary directory to the one where the binary is located.
	// Can be overriden with the option "aux-dir".
	if (argv[0] && argv[0][0]) {
		const char *tail = path_tail(argv[0]);  // "dir/file" -> "file"
		if (tail > argv[0]+1) {
			DynStr path=NULL;
			dstr_copy(path, tail - argv[0] - 1, argv[0]);
			mlis_option_set(ctx, MLIS_OPT_AUX_DIR, path);
			dstr_free(path);
		}
	}

	TRY( r = arg_parse(argc, argv, 1, short_options, mlis_cli_opt_set, &opt) );
	if (!r) {
		puts(help_string);
		return 0;
	}

	mlis_option_set(ctx, MLIS_OPT_ERROR_HANDLER, error_handler, (void*)&opt);
	mlis_option_set(ctx, MLIS_OPT_CALLBACK, progress_callback, (void*)&opt);

	// Load image io codecs
#ifdef USE_LIB_PNG
	extern const ImageCodec img_codec_png;
	img_codec_register(&img_codec_png);
#endif
#ifdef USE_LIB_JPEG
	extern const ImageCodec img_codec_jpeg;
	img_codec_register(&img_codec_jpeg);
#endif
	extern const ImageCodec img_codec_pnm;
	img_codec_register(&img_codec_pnm);

	// Run command
#define IF_CMD(NAME) \
	else if (!strcmp(opt.cmd, NAME))

	if (!opt.cmd) {
		puts("No command. Use -h for help.");
	}
	IF_CMD("list-backends") {
		mlis_cli_backends_print(&opt, ctx);
	}
	IF_CMD("generate") {
		TRY( mlis_cli_generate(&opt, ctx) );
	}
	IF_CMD("vae-encode") {
		TRY( mlis_cli_vae_cmd(&opt, ctx, true, false) );
	}
	IF_CMD("vae-decode") {
		TRY( mlis_cli_vae_cmd(&opt, ctx, false, true) );
	}
	IF_CMD("vae-test") {
		TRY( mlis_cli_vae_cmd(&opt, ctx, true, true) );
	}
	IF_CMD("clip-encode") {
		TRY( mlis_cli_clip_cmd(&opt, ctx) );
	}
	IF_CMD("check") {
		TRY( mlis_cli_check(&opt, ctx) );
	}
	else {
		ERROR_LOG(-1, "Unknown command '%s'", opt.cmd);
	}

end:
	if (R<0) log_error("error exit: %x", -R);
	mlis_ctx_destroy(&ctx);
	return -R;
}
