/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * mlimgsynth main code.
 */
#include "ccommon/timing.h"
#include "ccommon/logging.h"
#include "ccommon/stream.h"
#include "ccommon/fsutil.h"
#include "ccommon/rng_philox.h"
#include "ccompute/tensorstore.h"

#include "localtensor.h"
#include "mlblock.h"
#include "tae.h"
#include "vae.h"
#include "clip.h"
#include "unet.h"
#include "lora.h"
#include "sampling.h"
#include "tensor_name_conv.h"
#include "util.h"

#define IDS_IMPLEMENTATION
#include "ids.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml_extend.h"

#include <math.h>

#define APP_NAME_VERSION "mlimgsynth v0.3.0"

#define debug_pause() do { \
	puts("Press ENTER to continue"); \
	getchar(); \
} while (0)

#define log_vec(LVL,DESC,VEC,VAR,I0,...) \
if (log_level_check(LVL)) { \
	log_line_begin(LVL); \
	log_line_str(DESC ":"); \
	vec_for(VEC,VAR,I0) log_line_strf(" " __VA_ARGS__); \
	log_line_end(); \
}

#define log_debug_vec(...)  log_vec(LOG_LVL_DEBUG, __VA_ARGS__)

const char version_string[] = APP_NAME_VERSION "\n";

const char help_string[] =
APP_NAME_VERSION "\n"
"Image synthesis using machine learning.\n"
"Currently Stable Diffusion 1, 2 and XL are implemented.\n"
"\n"
"Usage: mlimgsynth [COMMAND] [OPTIONS]\n"
"\n"
"Commands:\n"
"  list-backends        List available GGML backends.\n"
"  vae-encode           Encode an image to a latent.\n"
"  vae-decode           Decode a latent to an image.\n"
"  vae-test             Encode and decode an image.\n"
"  clip-encode          Encode a prompt with the CLIP tokenizer and model.\n"
"  generate             Generate an image.\n"
"  check                Checks that all the operations (models) are working.\n"
"\n"
"Options:\n"
"  -p TEXT              Prompt for text conditioning.\n"
"  -n TEXT              Negative prompt for text unconditioning.\n"
"  -b NAME              Backend for computation.\n"
"  -B TEXT              Backend-specific parameters.\n"
"  -t INT               Number of threads to use in the CPU backend.\n"
"  -m PATH              Model file.\n"
"  -i PATH              Input image or latent.\n"
"  -2 PATH              Second input.\n"
"  -o PATH              Output path.\n"
"  -s --steps INT       Denoising steps with UNet.\n"
"  -W INT               Image width. Default: 512 (SD1), 768 (SD2), 1024 (SDXL).\n"
"  -H INT               Image height. Default: width.\n"
"  -S --seed INT        RNG seed.\n"
"  --in-mask PATH       Input image mask for inpainting.\n"
"\n"
"  --method NAME        Sampling method (default taylor3).\n"
"                       euler, euler_a, heun, taylor3, dpm++2m, dpm++2s, dpm++2s_a.\n"
"                       The _a variants are just a shortcut for --s-ancestral 1.\n"
"  --sched NAME         Sampling scheduler: uniform (default), karras.\n"
"  --s-noise FLOAT      Level of noise injection at each sampling step (try 1).\n"
"  --s-ancestral FLOAT  Ancestral sampling noise level (try 1).\n"
"  --cfg-scale FLOAT    Enables and sets the scale of the classifier-free guidance\n"
"                       (default: 1).\n"
"  --clip-skip INT      Number of CLIP layers to skip.\n"
"                       Default: 1 (SD1), 2 (SD2/XL).\n"
"  --f-t-ini FLOAT      Initial time factor (default 1).\n"
"                       Use it to control the strength in img2img.\n"
"  --f-t-end FLOAT      End time factor (default 0).\n"
"  --tae PATH           Enables TAE and sets path to tensors.\n"
"  --unet-split         Split each unet compute step compute in two parts to reduce memory usage.\n"
"  --vae-tile INT       Encode and decode images using tiles of NxN pixels.\n"
"                       Reduces memory usage. On doubt, try 512."
//"  --type NAME          Convert the weight to this tensor type.\n"
//"                       Useful to quantize and reduce RAM usage (try q8_0).\n"
"  --lora PATH:MULT     Apply the LoRA from PATH with multiplier MULT.\n"
"         PATH          The multiplier is optional.\n"
"                       This option can be used multiple time.\n"
"  --lora-dir PATH      Directory to search for LoRA's found in the prompt as:\n"
"                       <lora:NAME:MULT>\n"
"                       where NAME is the file name without extension.\n"
"  --dump               Dumps models tensors and graphs.\n"
"\n"
"  -q                   Quiet: reduces information output.\n"
"  -v                   Verbose: increases information output.\n"
"  -d                   Enables debug output.\n"
"  -h                   Print this message and exit.\n"
"  --version            Print the version and exit.\n"
;

typedef struct {
	MLCtx ctx;
	TensorStore tstore;
	Stream stm_model, stm_tae;
	ClipTokenizer tokr;
	DenoiseSampler sampler;

	const SdTaeParams *tae_p;
	const VaeParams *vae_p;
	const ClipParams *clip_p, *clip2_p;
	const UnetParams *unet_p;

	DynStr path_bin, tmps_path, prompt;

	struct {
		DynStr path;
		float mult;
	} *loras;  //vector

	struct {
		const char *backend, *beparams,
			*path_model, *path_in, *path_in2, *path_out, *path_tae,
			*path_inmask, *path_lora_dir,
			*prompt, *nprompt;
		int cmd, n_thread, width, height, seed, clip_skip, vae_tile;
		float cfg_scale;
		unsigned dump_info:1, use_tae:1, use_cfg:1, unet_split:1;
	} c;
} MLImgSynthApp;

int mlis_args_lora_add(MLImgSynthApp* S, const char* arg)
{
	unsigned i = vec_count(S->loras);
	vec_append_zero(S->loras, 1);
	dstr_copyz(S->loras[i].path, arg);
	S->loras[i].mult = 1;

	// Find the last ':'
	unsigned l = dstr_count(S->loras[i].path);
	char *b=S->loras[i].path, *c=b+l-1;
	while (c>=b && *c != ':') c--;
	if (*c == ':' && c > b+1) {  // windows path have ':' at 2nd position
		S->loras[i].mult = atof(c+1);
		dstr_resize(S->loras[i].path, c-b);
	}

	return 1;
}

int mlis_lora_path_find(MLImgSynthApp* S, StrSlice name, DynStr *out)
{
	//TODO: more sophisticated file search
	dstr_copyz(*out, S->c.path_lora_dir);
	if (dstr_count(*out) > 0) {
		char c = (*out)[dstr_count(*out)-1];
		if (c != '/' && c != '\\')
			dstr_push(*out, '/');
	}
	dstr_append(*out, name.s, name.b);
	dstr_appendz(*out, ".safetensors");  //TODO: support other
	return 1;
}

int mlis_prompt_cfg_parse(MLImgSynthApp* S, StrSlice ss)
{
	int R=1, count=0;
	if (strsl_prefix_trim(&ss, strsl_static("lora:")))
	{
		const char *sep=ss.b, *end=ss.b+ss.s;
		while (sep < end && *sep != ':') sep++;

		unsigned i = vec_count(S->loras);
		vec_append_zero(S->loras, 1);

		TRY( mlis_lora_path_find(S, strsl_fromr(ss.b, sep), &S->loras[i].path) );
		
		S->loras[i].mult = 1;
		if (*sep == ':') {  //optional
			char *tail=NULL;
			S->loras[i].mult = strtof(sep+1, &tail);
			if (tail != end) ERROR_LOG(-1, "wrong format");
		}
		
		count++;
	}
	else ERROR_LOG(-1, "unknown prompt option");

	if (count > 0) {
		log_debug("cfg elements in prompt: %d", count);
		S->c.prompt = S->prompt;
	}

end:
	if (R<0) log_error("prompt option '%.*s': %x", (int)ss.s, ss.b, -R);
	return R;
}

int mlis_prompt_cfg_extract(MLImgSynthApp* S)
{
	if (!S->c.prompt) return 0;
	int R=1;
	unsigned n = strlen(S->c.prompt);
	dstr_realloc(S->prompt, n);  //reserve memory
	for (const char *cur=S->c.prompt, *end=cur+n; cur<end; ++cur) {
		if (*cur == '<') {
			const char *e=cur+1;
			while (e < end && *e != '>') ++e;
			if (*e != '>') ERROR_LOG(-1, "prompt: '<' not matched with '>'");
			TRY( mlis_prompt_cfg_parse(S, strsl_fromr(cur+1, e)) );
			cur = e;
		}
		else dstr_push(S->prompt, *cur);
	}
end:
	return R;
}

int mlis_args_load(MLImgSynthApp* S, int argc, char* argv[])
{
	int R=1;
	bool print_help=false;

	if (argc <= 1) RETURN(0);

	// Defaults
	S->ctx.c.wtype = GGML_TYPE_F16;

	//TODO: validate input ranges

	int i, j;
	for (i=1; i<argc; ++i) {
		char * arg = argv[i];
		if (arg[0] == '-' && arg[1] == '-') {
			char * next = (i+1 < argc) ? argv[i+1] : "";
			if      (!strcmp(arg+2, "method"))
			{
				if (!strcmp(next, "euler_a")) {
					S->sampler.c.method = ID_euler;
					S->sampler.c.s_ancestral = 1;
				}
				else if (!strcmp(next, "dpm++2s_a")) {
					S->sampler.c.method = ID_dpmpp2s;
					S->sampler.c.s_ancestral = 1;
				}
				else {
					S->sampler.c.method = id_fromz(next);
				}
				i++;
			}
			else if (!strcmp(arg+2, "sched" )) {
				S->sampler.c.sched = id_fromz(next); i++;
			}
			else if (!strcmp(arg+2, "s-ancestral" )) {
				S->sampler.c.s_ancestral = atof(next); i++;
			}
			else if (!strcmp(arg+2, "s-noise" )) {
				S->sampler.c.s_noise = atof(next); i++;
			}
			else if (!strcmp(arg+2, "steps" )) {
				S->sampler.c.n_step = atoi(next); i++;
			}
			else if (!strcmp(arg+2, "seed" )) {
				g_rng.seed = strtoull(next, NULL, 10); i++;
			}
			else if (!strcmp(arg+2, "in-mask" )) {
				S->c.path_inmask = next; i++;
			}
			else if (!strcmp(arg+2, "cfg-scale" )) {
				S->c.cfg_scale = atof(next); i++;
				S->c.use_cfg = S->c.cfg_scale > 1;
			}
			else if (!strcmp(arg+2, "clip-skip" )) {
				S->c.clip_skip = atoi(next); i++;
			}
			else if (!strcmp(arg+2, "f-t-ini" )) {
				S->sampler.c.f_t_ini = atof(next); i++;
			}
			else if (!strcmp(arg+2, "f-t-end" )) {
				S->sampler.c.f_t_end = atof(next); i++;
			}
			else if (!strcmp(arg+2, "tae" )) {
				S->c.path_tae = next; i++;
				S->c.use_tae = S->c.path_tae && S->c.path_tae[0];
			}
			else if (!strcmp(arg+2, "unet-split" )) {
				S->c.unet_split = true;
			}
			else if (!strcmp(arg+2, "vae-tile" )) {
				S->c.vae_tile = atoi(next);  i++;
			}
			//TODO: using other types is not working well right now
			/*else if (!strcmp(arg+2, "type" )) {
				i++;
				int dt = tstore_dtype_fromz(next);
				if (!(dt > 0))
					ERROR_LOG(-1, "unknown tensor type '%s'", next);				
				S->ctx.c.wtype = tstore_dtype_attr(dt)->ggml;
			}*/
			else if (!strcmp(arg+2, "lora" )) {
				TRY( mlis_args_lora_add(S, next) );
				i++;
			}
			else if (!strcmp(arg+2, "lora-dir" )) {
				S->c.path_lora_dir = next;
				i++;
			}
			else if (!strcmp(arg+2, "dump" )) {
				S->c.dump_info = true;
			}
			else if (!strcmp(arg+2, "version" )) {
				puts(APP_NAME_VERSION "\n");
				RETURN(0);
			}
			else if (!strcmp(arg+2, "help" )) {
				print_help = true;
				RETURN(0);
			}
			else {
				ERROR_LOG(-1, "Unknown option '%s'", arg);
			}
		}
		else
		if (arg[0] == '-') {
			char opt;
			for (j=1; (opt = arg[j]); ++j) {
				char * next = (i+1 < argc) ? argv[i+1] : "";
				switch (opt) {
				case 'p':  S->c.prompt = next; i++; break;
				case 'n':  S->c.nprompt = next; i++; break;
				case 'b':  S->c.backend = next; i++; break;
				case 'B':  S->c.beparams = next; i++; break;
				case 't':  S->c.n_thread = atoi(next); i++; break;
				case 's':  S->sampler.c.n_step = atoi(next); i++; break;
				case 'W':  S->c.width = atoi(next); i++; break;
				case 'H':  S->c.height = atoi(next); i++; break;
				case 'S':  g_rng.seed = strtoull(next, NULL, 10); i++; break;
				case 'm':  S->c.path_model = next; i++; break;
				case 'i':  S->c.path_in = next; i++; break;
				case '2':  S->c.path_in2 = next; i++; break;
				case 'o':  S->c.path_out = next; i++; break;
				case 'q':  log_level_inc(-LOG_LVL_STEP); break;
				case 'v':  log_level_inc(+LOG_LVL_STEP); break;
				case 'd':  log_level_set(LOG_LVL_DEBUG); break;
				case 'h':
					print_help = true;
					RETURN(0);
				default:
					ERROR_LOG(-1, "Unknown option '-%c'", opt);
				}
			}
		}
		else if (!S->c.cmd) {
			S->c.cmd = id_fromz(arg);
		}
		else {
			ERROR_LOG(-1, "Excess of arguments: %s", arg);
		}
	}

	IFNPOSSET(S->c.cfg_scale, 1);
	
	// Save the path of the direction where the binary is located.
	// May be used later to look for related files.
	if (argv[0] && argv[0][0]) {
		const char *tail = path_tail(argv[0]);  // "dir/file" -> "file"
		if (tail > argv[0]+1) {
			dstr_copy(S->path_bin, tail - argv[0] - 1, argv[0]);
			log_debug("bin path: %s", S->path_bin);
			assert( file_exists(S->path_bin) );
		}
	}

	// RNG seed
	if (!g_rng.seed) g_rng.seed = timing_timeofday();
	log_info("Seed: %zu", g_rng.seed);

	// Extract configuration option from the prompt (e.g. loras)
	TRY( mlis_prompt_cfg_extract(S) );

end:
	if (print_help) puts(help_string);
	return R;
}

void mlis_free(MLImgSynthApp* S)
{
	dnsamp_free(&S->sampler);
	dstr_free(S->tmps_path);
	dstr_free(S->path_bin);
	mlctx_free(&S->ctx);
	clip_tokr_free(&S->tokr);
	stream_close(&S->stm_tae, 0);
	tstore_free(&S->tstore);
	stream_close(&S->stm_model, 0);
	if (S->ctx.backend)
		ggml_backend_free(S->ctx.backend);
	if (S->ctx.backend2)
		ggml_backend_free(S->ctx.backend2);

	dstr_free(S->prompt);
	vec_for(S->loras,i,0)
		dstr_free(S->loras[i].path);
	vec_free(S->loras);
}

int mlis_file_find(MLImgSynthApp* S, const char *name, const char **path)
{
	int R=1;
	dstr_printf(S->tmps_path, "%s", name);
	if (file_exists(S->tmps_path)) goto end;
	if (!dstr_empty(S->path_bin)) {
		dstr_printf(S->tmps_path, "%s/%s", S->path_bin, name);
		if (file_exists(S->tmps_path)) goto end;
	}
	dstr_printf(S->tmps_path, "/usr/share/mlimgsynth/%s", name);
	if (file_exists(S->tmps_path)) goto end;
	dstr_printf(S->tmps_path, "/usr/local/share/mlimgsynth/%s", name);
	if (file_exists(S->tmps_path)) goto end;
	//TODO: other paths? system specific? environment variable?
	ERROR_LOG(-1, "file '%s' could not be found", name);
end:
	if (R==1) *path = S->tmps_path;
	return R;
}

static
int open_clip_attn_conv(TensorStore* ts, const TSTensorEntry *e, const char* name)
{
	TSTensorEntry new={0};
	DynStr tmps = dstr_stack(128);
	StrSlice ss = strsl_fromz(name);

	const char *type;
	if (strsl_suffixz_trim(&ss, "in_proj_bias")) type = "bias";
	else if (strsl_suffixz_trim(&ss, "in_proj_weight")) type = "weight";
	else return 0;

	unsigned idim = e->shape[1] == 1 ? 0 : 1;

	if (!(e->shape[idim] % 3 == 0)) {
		log_error("invalid open_clip tensor '%s'", name);
		return -1;
	}

	new = *e;
	new.shape[idim] /= 3;
	new.size /= 3;
	
	dstr_copy(tmps, ss.s, ss.b);
	dstr_appendz(tmps, "q_proj.");
	dstr_appendz(tmps, type);
	tstore_tensor_add(ts, tmps, &new);
	new.offset += new.size;
	
	dstr_copy(tmps, ss.s, ss.b);
	dstr_appendz(tmps, "k_proj.");
	dstr_appendz(tmps, type);
	tstore_tensor_add(ts, tmps, &new);
	new.offset += new.size;
	
	dstr_copy(tmps, ss.s, ss.b);
	dstr_appendz(tmps, "v_proj.");
	dstr_appendz(tmps, type);
	tstore_tensor_add(ts, tmps, &new);

	return 1;
}

static
int tensor_callback_main(void* user, TensorStore* ts, TSTensorEntry* te,
	DynStr* pname)
{
	int r;
	DynStr newname = dstr_stack(128);

	// Rename tensors to uniform names
	TRYR( r = tnconv_sd(strsl_fromd(*pname), &newname) );
	if (r == 0) {  //unused
		log_debug2("unused tensor '%s'", *pname);
		return 0;
	}

	if (r == TNCONV_R_QKV_PROJ) {
		// Convert from openclip attention projection in one tensor
		// to three tensors.
		TRYR( open_clip_attn_conv(ts, te, newname) );
		return 0;  //do not save the original tensor
	}

	dstr_copyd(*pname, newname);
	return 1;
}

static
int tensor_callback_prefix_add(void* user, TensorStore* ts, TSTensorEntry* te,
	DynStr* pname)
{
	const char *prefix = user;
	if (prefix)
		dstr_insertz(*pname, 0, prefix);
	return 1;
}

static
int tensor_callback_lora(void* user, TensorStore* ts, TSTensorEntry* te,
	DynStr* pname)
{
	int r;
	DynStr newname = dstr_stack(128);
	StrSlice ss = strsl_fromd(*pname);
	
	// Check and remove prefix
	if (!strsl_prefix_trim(&ss, strsl_static("lora_"))) return 0;

	// Rename tensors to uniform names
	TRYR( r = tnconv_sd(ss, &newname) );
	if (r == 0) {
		if (strsl_endswith(ss, strsl_static(".lora_down.weight"))) {
			log_error("unmatched lora tensor: %s", *pname);
			return -1;
		} else {
			log_debug2("unused lora tensor: %s", *pname);
			return 0;
		}
	}

	dstr_copyd(*pname, newname);
	return 1;
}

int mlis_lora_load_apply(MLImgSynthApp* S, const char* path, float mult)
{
	int R=1;
	Stream stm={0};
	TensorStore ts={ .ss=&g_ss };

	log_debug("lora apply: '%s' %g", path, mult);

	TRY_LOG( stream_open_file(&stm, path, SOF_READ | SOF_MMAP),
		"could not open '%s'", path);

	TSCallback cb = { tensor_callback_lora };
	TRY( tstore_read(&ts, &stm, NULL, &cb) );
	
	if (S->c.dump_info)
		TRY( tstore_info_dump_path(&ts, "dump-tensors-lora.txt") );

	TRY( lora_apply(&S->tstore, &ts, mult, &S->ctx) );

end:
	if (R<0) log_error("lora apply '%s': %x", path, -R);
	return R;
}

static
void ggml__backend_set_n_threads(ggml_backend_t backend, int n_threads)
{
	ggml_backend_dev_t dev = ggml_backend_get_device(backend);
	ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
	ggml_backend_set_n_threads_t func = ggml_backend_reg_get_proc_address(reg,
		"ggml_backend_set_n_threads");
	if (func)
		func(backend, n_threads);
}

int mlis_backend_init(MLImgSynthApp* S)
{
	int R=1;

	if (S->c.backend && S->c.backend[0])
		S->ctx.backend = ggml_backend_init_by_name(S->c.backend, S->c.beparams);
	else 
		S->ctx.backend = ggml_backend_init_best();
	
	if (!S->ctx.backend) ERROR_LOG(-1, "ggml backend init");
	log_info("Backend: %s", ggml_backend_name(S->ctx.backend));

	if (S->c.n_thread)
		ggml__backend_set_n_threads(S->ctx.backend, S->c.n_thread);

#if USE_GGML_SCHED  //old code
	{
		log_debug("Fallback backend CPU");
		S->ctx.backend2 = ggml_backend_cpu_init();
		if (!S->ctx.backend2) ERROR_LOG(-1, "ggml fallback backend CPU init");
		
		if (S->c.n_thread)
			ggml_backend_set_n_threads_t(S->ctx.backend2, S->c.n_thread);
	}
#endif

end:
	return R;
}

int mlis_model_load(MLImgSynthApp* S)
{
	int R=1;

	double t = timing_time();
	if (S->c.path_model) {
		log_debug("Loading model header from '%s'", S->c.path_model);
		TRY_LOG( stream_open_file(&S->stm_model, S->c.path_model,
			SOF_READ | SOF_MMAP),
			"could not open '%s'", S->c.path_model);
		log_debug("model stream class: %s", S->stm_model.cls->name);
		TSCallback cb = { tensor_callback_main };
		TRY( tstore_read(&S->tstore, &S->stm_model, NULL, &cb) );
	}

	// TAE model load
	if (S->c.path_tae) {
		log_debug("Loading model header from '%s'", S->c.path_tae);
		TRY_LOG( stream_open_file(&S->stm_tae, S->c.path_tae,
			SOF_READ | SOF_MMAP),
			"could not open '%s'", S->c.path_tae);
		TSCallback cb = { tensor_callback_prefix_add, "tae." };
		TRY( tstore_read(&S->tstore, &S->stm_tae, NULL, &cb) );
	}

	t = timing_time() - t;
	log_info2("Model header loaded {%.3fs}", t);
		
	if (S->c.dump_info)
		TRY( tstore_info_dump_path(&S->tstore, "dump-tensors.txt") );

end:
	return R;
}

int mlis_model_ident(MLImgSynthApp* S)
{
	int R=1;
	const char *model_type=NULL;
	const TSTensorEntry *te=NULL;

	if ((te = tstore_tensor_get(&S->tstore,
		"unet.in.1.1.transf.0.attn2.k_proj.weight")))
	{
		if (te->shape[0] == 768) {
			model_type = "Stable Diffusion 1.x";
			S->tae_p = &g_sdtae_sd1;
			S->vae_p = &g_vae_sd1;
			S->clip_p = &g_clip_vit_l_14;
			S->unet_p = &g_unet_sd1;
			IFNPOSSET(S->c.width, 512);
			IFNPOSSET(S->c.height, S->c.width);
			IFNPOSSET(S->c.clip_skip, 1);
		}
		else if (te->shape[0] == 1024) {
			model_type = "Stable Diffusion 2.x";
			S->tae_p = &g_sdtae_sd1;
			S->vae_p = &g_vae_sd1;
			S->clip_p = &g_clip_vit_h_14;
			S->unet_p = &g_unet_sd2;
			IFNPOSSET(S->c.width, 768);
			IFNPOSSET(S->c.height, S->c.width);
			IFNPOSSET(S->c.clip_skip, 2);
		}
	}
	else if ((te = tstore_tensor_get(&S->tstore,
		"unet.in.4.1.transf.0.attn2.k_proj.weight")))
	{
		if (te->shape[0] == 2048) {
			model_type = "Stable Diffusion XL";
			S->tae_p = &g_sdtae_sd1;
			S->vae_p = &g_vae_sdxl;
			S->clip_p = &g_clip_vit_l_14;
			S->clip2_p = &g_clip_vit_bigg_14;
			S->unet_p = &g_unet_sdxl;
			IFNPOSSET(S->c.width, 1024);
			IFNPOSSET(S->c.height, S->c.width);
			IFNPOSSET(S->c.clip_skip, 2);
		}
	}

	if (!model_type)
		ERROR_LOG(-1, "Unknown model type");
	
	log_info("Model type: %s", model_type);
	
end:
	return R;
}

int mlis_ml_init(MLImgSynthApp* S)
{
	int R=1;
	double t;
	assert(!S->ctx.backend);
	
	S->ctx.tstore = &S->tstore;
	S->ctx.ss = &g_ss;
	S->tstore.ss = &g_ss;

	// Backend init
	TRY( mlis_backend_init(S) );

	// Model parameters header load
	TRY( mlis_model_load(S) );

	// Identify model type
	TRY( mlis_model_ident(S) );
	
	// Load loras
	if (vec_count(S->loras)) {
		t = timing_time();
		vec_for(S->loras,i,0) {
			TRY( mlis_lora_load_apply(S, S->loras[i].path, S->loras[i].mult) );
		}
		t = timing_time() - t;
		log_info("LoRA's applied: %u {%.3fs}", vec_count(S->loras), t);
	}
	
end:
	return R;
}

int mlis_img_encode(MLImgSynthApp* S, const LocalTensor* img, LocalTensor* latent)
{
	int R=1;
	if (S->c.use_tae) {
		S->ctx.tprefix = "tae";
		TRY( sdtae_encode(&S->ctx, S->tae_p, img, latent) );
	} else {
		S->ctx.tprefix = "vae";
		TRY( sdvae_encode(&S->ctx, S->vae_p, img, latent, S->c.vae_tile) );
	}
	TRY_LOG( ltensor_finite_check(img), "NaN found in encoded latent");
end:
	return R;
}

int mlis_img_decode(MLImgSynthApp* S, const LocalTensor* latent, LocalTensor* img)
{
	int R=1;
	if (S->c.use_tae) {
		S->ctx.tprefix = "tae";
		TRY( sdtae_decode(&S->ctx, S->tae_p, latent, img) );
	} else {
		S->ctx.tprefix = "vae";
		TRY( sdvae_decode(&S->ctx, S->vae_p, latent, img, S->c.vae_tile) );
	}
	TRY_LOG( ltensor_finite_check(img), "NaN found in decoded image");
end:
	return R;
}

int mlis_vae_cmd(MLImgSynthApp* S, bool encode, bool decode)
{
	int R=1;
	Image img_in={0}, img_out={0};
	LocalTensor latent={0};

	TRY( mlis_ml_init(S) );

	if (encode) {
		// Load image
		if (!S->c.path_in) ERROR_LOG(-1, "Input image not set");
		log_debug("Loading image from '%s'", S->c.path_in);
		TRY_LOG( img_load_file(&img_in, S->c.path_in),
			"could not load '%s'", S->c.path_in);

		// Check image
		log_info("Input image: %ux%ux%u", img_in.w, img_in.h, img_in.bypp);
		if (img_in.format != IMG_FORMAT_RGB)
			ERROR_LOG(-1, "image must be RBG");
		if (!(img_in.w%64==0 && img_in.h%64==0))
			ERROR_LOG(-1, "image dimentions must be multiples of 64");

		// Convert to tensor
		ltensor_from_image(&latent, &img_in);

		// Encode
		TRY( mlis_img_encode(S, &latent, &latent) );
		const char *path_latent =
			(decode || !S->c.path_out) ? "latent.tensor" : S->c.path_out;
		TRY( ltensor_save_path(&latent, path_latent) );
	}
	else {
		// Load latent
		if (!S->c.path_in) ERROR_LOG(-1, "Input latent not set");
		TRY_LOG( ltensor_load_path(&latent, S->c.path_in),
			"could not load '%s'", S->c.path_in);
		log_debug3_ltensor(&latent, "latent");
	}

	// Sample latent distribution if needed
	if (latent.s[2] == S->vae_p->d_embed*2)
	{
		log_debug("latent sampling");
		sdvae_latent_sample(&latent, &latent, S->vae_p);
	}
	else if (latent.s[2] != S->vae_p->d_embed)
	{
		ERROR_LOG(-1, "invalid latent shape: " LT_SHAPE_FMT,
			LT_SHAPE_UNPACK(latent));
	}

	if (decode) {
		// Decode
		TRY( mlis_img_decode(S, &latent, &latent) );
		ltensor_to_image(&latent, &img_out);
		IFFALSESET(S->c.path_out, "output.png");
		log_debug("Writing image to '%s'", S->c.path_out);
		TRY( img_save_file(&img_out, S->c.path_out) );
	}
	
	if (encode && decode) {
		// Compare output with input
		assert( img_in.w == img_out.w && img_in.h == img_out.h &&
			img_out.bypp == 3);
		double err=0;
		for (unsigned y=0; y<img_in.h; ++y) {
			for (unsigned x=0; x<img_in.w; ++x) {
				for (unsigned c=0; c<3; ++c) {
					int d = (int) IMG_INDEX3(img_out,x,y,c)
						- IMG_INDEX3(img_in,x,y,c);
					err += d*d;
				}
			}
		}
		err = sqrt(err / (img_in.w * img_in.h * 3));
		log_info("MSE: %.3f", err);
	}

end:
	if (R<0) log_error("vae cmd");
	img_free(&img_out);
	ltensor_free(&latent);
	img_free(&img_in);
	return R;
}

int mlis_clip_encode(MLImgSynthApp* S, const char* prompt, LocalTensor* embed,
	LocalTensor* feat, const ClipParams* clip_p, const char* tprefix, bool norm)
{
	int R=1;
	int32_t *tokens=NULL;

	// Load vocabulary
	if (!clip_good(&S->tokr)) {
		const char *path_vocab;
		TRY( mlis_file_find(S, "clip-vocab-merges.txt", &path_vocab) );
		log_debug("Loading vocabulary from '%s'", path_vocab);
		TRY( clip_tokr_vocab_load(&S->tokr, path_vocab) );
	}	
	unsigned nvocab = strsto_count(&S->tokr.vocab);
	if (nvocab != clip_p->n_vocab)
		ERROR_LOG(-1, "wrong vocabulary size: %u (read) != %u (expected)",
			nvocab, clip_p->n_vocab);

	// Tokenize the prompt
	IFFALSESET(prompt, "");
	TRY( clip_tokr_tokenize(&S->tokr, prompt, &tokens) );
	log_debug_vec("Tokens", tokens, i, 0, "%u %s",
		tokens[i], clip_tokr_word_from_token(&S->tokr, tokens[i]) );
	log_info("Prompt: %u tokens", vec_count(tokens));

	S->ctx.tprefix = tprefix;
	TRY( clip_text_encode(&S->ctx, clip_p,
		tokens, embed, feat, S->c.clip_skip, norm) );

end:
	vec_free(tokens);
	return R;
}

int mlis_clip_cmd(MLImgSynthApp* S)
{
	int R=1;
	LocalTensor embed={0}, feat={0}, feat2={0};
	
	TRY( mlis_ml_init(S) );

	bool has_tproj = tstore_tensor_get(&S->tstore, "clip.text.text_proj");

	TRY( mlis_clip_encode(S, S->c.prompt, &embed,
		has_tproj ? &feat : NULL, S->clip_p, "clip", true) );
	const char *path_out = S->c.path_out;
	IFFALSESET(path_out, "text_embed.tensor");
	ltensor_save_path(&embed, path_out);
	ltensor_img_redblue_path(&embed, "text_embed.png");
	
	if (feat.d) {
		ltensor_save_path(&feat, "text_feat.tensor");	
		
		if (S->c.path_in2) {
			ltensor_load_path(&feat2, S->c.path_in2);
			TRY( ltensor_shape_check_log(&feat2, "feat2", feat.s[0], 1, 1, 1) );

			//TODO: move similarity calculation to a function
			unsigned n=ltensor_nelements(&feat2);
			double p=0, n1=0, n2=0;
			for (unsigned i=0; i<n; ++i) {
				n1 += feat.d[i] * feat.d[i];
				n2 += feat2.d[i] * feat2.d[i];
				p += feat.d[i] * feat2.d[i];
			}
			p /= sqrt(n1*n2);
			//TODO: softmax? logit_scale?
			
			log_info("Features similarity: %.6f", p);
		}
	}

end:
	ltensor_free(&feat2);
	ltensor_free(&feat);
	ltensor_free(&embed);
	return R;
}

// Fill sinusoidal timestep embedding (from CompVis)
size_t sd_timestep_embedding(unsigned nsteps, float* steps, unsigned dim,
	float max_period, float* out)
{
	assert(dim%2==0);
	unsigned half = dim/2;
	for (unsigned i=0; i<half; ++i) {
		float freq = exp(-log(max_period)*i/half);
		for (unsigned s=0; s<nsteps; ++s) {
			out[s*dim+i     ] = cos(steps[s] * freq);
			out[s*dim+i+half] = sin(steps[s] * freq);
		}
	}
	return nsteps * dim;
}

int mlis_sd_cond_get(MLImgSynthApp* S, const char* prompt,
	LocalTensor* cond, LocalTensor* label)
{
	int R=1;
	LocalTensor tmpt={0};

	bool norm = S->unet_p->clip_norm;

	TRY( mlis_clip_encode(S, prompt, cond, NULL, S->clip_p, "clip", norm) );

	if (S->clip2_p) {
		assert(label);
		TRY( mlis_clip_encode(S, prompt, &tmpt, NULL,
			S->clip2_p, "clip2", norm) );

		// Concatenate both text embeddings
		assert( cond->s[1] == tmpt.s[1] &&
		        cond->s[2] == 1 && tmpt.s[2] == 1 &&
			    cond->s[3] == 1 && tmpt.s[3] == 1 );	

		unsigned n_tok = tmpt.s[1],
		         n_emb1 = cond->s[0],
				 n_emb2 = tmpt.s[0],
		         n_emb = n_emb1 + n_emb2;

		ltensor_resize(cond, n_emb, n_tok, 1, 1);
		for (unsigned i1=n_tok-1; (int)i1>=0; --i1) {
			ARRAY_COPY(cond->d+n_emb*i1+n_emb1, tmpt.d+n_emb2*i1, n_emb2);
			ARRAY_COPY(cond->d+n_emb*i1, cond->d+n_emb1*i1, n_emb1);
		}
		
		//TODO: no need to reprocess from scratch...
		TRY( mlis_clip_encode(S, prompt, NULL, label,
			S->clip2_p, "clip2", true) );

		// Complete label embedding
		assert( label->s[0]==n_emb2 &&
			label->s[1]==1 && label->s[2]==1 && label->s[3]==1 );
		ltensor_resize(label, S->unet_p->ch_adm_in, 1, 1, 1);
		float *ld = label->d + n_emb2;
		unsigned w = S->c.width, h = S->c.height;
		// Original size
		ld += sd_timestep_embedding(2, (float[]){h,w}, 256, 10000, ld);
		// Crop top,left
		ld += sd_timestep_embedding(2, (float[]){0,0}, 256, 10000, ld);
		// Target size
		ld += sd_timestep_embedding(2, (float[]){h,w}, 256, 10000, ld);
		assert(ld == label->d + label->s[0]);
	}

end:
	ltensor_free(&tmpt);
	return R;
}

struct dxdt_args {
	MLImgSynthApp *app;
	UnetState *ctx;
	LocalTensor *cond, *label, *uncond, *unlabel, *tmpt;
};

int mlis_denoise_dxdt(Solver* sol, float t, const LocalTensor* x,
	LocalTensor* dx)
{
	if (!(t >= 0)) return 0;
	struct dxdt_args *A = sol->user;
	
	A->ctx->i_step = A->app->sampler.i_step;  //just to show the progress
	A->ctx->n_step = A->app->sampler.n_step;
	
	TRYR( unet_denoise_run(A->ctx, x, A->cond, A->label, t, dx) );
	
	if (A->app->c.use_cfg) {
		TRYR( unet_denoise_run(A->ctx, x, A->uncond, A->unlabel, t, A->tmpt) );
		float f = A->app->c.cfg_scale;
		ltensor_for(*dx,i,0) dx->d[i] = dx->d[i]*f + A->tmpt->d[i]*(1-f);
	}
	
	return 1;
}

int mlis_gen_img_save(MLImgSynthApp* S, Image* img, int nfe)
{
	int R=1;
	DynStr infotxt=NULL;

	// Make info text
	// Imitates stable-diffusion-webui create_infotext
	if (S->c.prompt)
		dstr_printfa(infotxt, "%s\n", S->c.prompt);
	//TODO: input latent or image filename?
	dstr_printfa(infotxt, "Seed: %"PRIu64, g_rng.seed);
	dstr_printfa(infotxt, ", Sampler: %s", id_str(S->sampler.c.method));
	dstr_printfa(infotxt, ", Schedule type: %s", id_str(S->sampler.c.sched));
	if (S->sampler.c.s_ancestral > 0)
		dstr_printfa(infotxt, ", Ancestral: %g", S->sampler.c.s_ancestral);
	if (S->sampler.c.s_noise > 0)
		dstr_printfa(infotxt, ", SNoise: %g", S->sampler.c.s_noise);
	if (S->c.use_cfg)
		dstr_printfa(infotxt, ", CFG scale: %g", S->c.cfg_scale);
	if (S->c.path_in) {
		dstr_printfa(infotxt, ", Mode: %s, t_ini: %g",
			S->sampler.c.lmask ? "inpaint" : "img2img", S->sampler.c.f_t_ini);
	}
	dstr_printfa(infotxt, ", Steps: %u", S->sampler.n_step);
	dstr_printfa(infotxt, ", NFE: %u", nfe);
	dstr_printfa(infotxt, ", Size: %ux%u", img->w, img->h);
	dstr_printfa(infotxt, ", Clip skip: %d", S->c.clip_skip);
	{
		const char *b = path_tail(S->c.path_model),
		           *e = path_ext(b);
		if (*e) e--;  // .
		dstr_appendz(infotxt, ", Model: ");
		dstr_append(infotxt, e-b, b);
	}
	if (S->c.use_tae)
		dstr_printfa(infotxt, ", VAE: tae");
	dstr_printfa(infotxt, ", Version: %s", APP_NAME_VERSION);
	
	// Save image with comment
	IFFALSESET(S->c.path_out, "output.png");
	log_debug("Writing image to '%s'", S->c.path_out);
	TRY( img_save_file_info(img, S->c.path_out, "parameters", infotxt) );

end:
	dstr_free(infotxt);
	return R;
}

int mlis_generate(MLImgSynthApp* S)
{
	int R=1;
	const char *path;
	UnetState ctx={0};
	Image img={0};
	LocalTensor latent={0}, lmask={0}, tmpt={0},
	            cond={0}, label={0},
				uncond={0}, unlabel={0};
	
	double tm = timing_time();

	unet_params_init();  //global
	
	TRY( mlis_ml_init(S) );
	int vae_f = S->vae_p->f_down;

	// Latent load
	if ((path = S->c.path_in)) {
		const char *ext = path_ext(path);
		if (!strcmp(ext, "tensor")) {  //latent
			TRY( ltensor_load_path(&latent, path) );
		}
		else {  //image
			log_debug("Loading image from '%s'", path);
			TRY_LOG( img_load_file(&img, path),
				"Could not load image '%s'", path);

			if (img.format == IMG_FORMAT_RGBA) {
				log_info("In-painting from alpha channel");
				ltensor_from_image_alpha(&latent, &lmask, &img);
				ltensor_downsize(&lmask, vae_f, vae_f, 1, 1);
			} else if (img.format == IMG_FORMAT_RGB)
				ltensor_from_image(&latent, &img);
			else
				ERROR_LOG(-1, "invalid image format: should be RGB or RGBA");

			TRY( mlis_img_encode(S, &latent, &latent) );
		}
		
		// Sample if needed
		if (latent.s[2] == S->unet_p->n_ch_in*2)
			sdvae_latent_sample(&latent, &latent, S->vae_p);

		TRY( ltensor_shape_check_log(&latent, "input latent",
			0, 0, S->unet_p->n_ch_in, 1) );
		
		log_debug3_ltensor(&latent, "input latent");

		S->c.width  = latent.s[0] * vae_f;
		S->c.height = latent.s[1] * vae_f;
	}
	else {
		log_debug("Empty initial latent");
		int f = S->vae_p->f_down,
		    w = S->c.width / f,
			h = S->c.height / f;
		ltensor_resize(&latent, w, h, S->unet_p->n_ch_in, 1);
		memset(latent.d, 0, ltensor_nbytes(&latent));
	}
	log_info("Output size: %ux%u", S->c.width, S->c.height);

	// Mask for inpainting
	if ((path = S->c.path_inmask)) {
		const char *ext = path_ext(path);
		if (!strcmp(ext, "tensor")) {  //latent
			TRY( ltensor_load_path(&lmask, path) );
		}
		else {  //image
			log_debug("Loading mask from '%s'", path);
			TRY_LOG( img_load_file(&img, path),
				"Could not load image '%s'", path);
			if (img.format == IMG_FORMAT_GRAY)
				ltensor_from_image(&lmask, &img);
			else
				ERROR_LOG(-1, "invalid mask format: should be grayscale");

			ltensor_downsize(&lmask, vae_f, vae_f, 1, 1);
		}
		TRY( ltensor_shape_check_log(&lmask, "input latent mask",
			latent.s[0], latent.s[1], 1, 1) );
		
		log_info("In-painting from mask");
	}

	// Conditioning load
	if ((path = S->c.path_in2)) {
		TRY( ltensor_load_path(&cond, path) );
		TRY( ltensor_shape_check_log(&cond, "input conditioning",
			S->unet_p->n_ctx, 0, 1, 1) );
	}
	else if (S->c.prompt)
		TRY( mlis_sd_cond_get(S, S->c.prompt, &cond, &label) );
	else
		ERROR_LOG(-1, "no conditioning (option -2) or prompt (option -p) set");
	
	log_debug3_ltensor(&cond, "cond");
	//ltensor_save_path(&cond, "cond.tensor");
	log_debug3_ltensor(&label, "label");

	// Negative conditioning
	if (S->c.use_cfg) {
		TRY( mlis_sd_cond_get(S, S->c.nprompt, &uncond, &unlabel) );
		if (S->unet_p->uncond_empty_zero && !(S->c.nprompt && S->c.nprompt[0]))
			ltensor_for(uncond,i,0) uncond.d[i] = 0;
	}
	else if (S->c.nprompt && S->c.nprompt[0])
		log_warning("negative prompt provided but CFG is not enabled");	
	
	log_debug3_ltensor(&uncond, "uncond");
	log_debug3_ltensor(&unlabel, "unlabel");

	// Sampling initialization
	S->sampler.unet_p = S->unet_p;
	S->sampler.nfe_per_dxdt = S->c.use_cfg ? 2 : 1;
	S->sampler.c.lmask = ltensor_good(&lmask) ? &lmask : NULL;

	struct dxdt_args A = { .app=S, .ctx=&ctx, .tmpt=&tmpt,
		.cond=&cond, .uncond=&uncond, .label=&label, .unlabel=&unlabel };
	S->sampler.solver.dxdt = mlis_denoise_dxdt;
	S->sampler.solver.user = &A;
	
	TRY( dnsamp_init(&S->sampler) );
	
	// Prepare computation
	S->ctx.tprefix = "unet";
	TRY( unet_denoise_init(&ctx, &S->ctx, S->unet_p, latent.s[0], latent.s[1],
		S->c.unet_split) );
	
	log_info("Generating "
		"(solver: %s, sched: %s, ancestral: %g, snoise: %g, cfg-s: %g, steps: %d"
		", nfe/s: %d)",
		id_str(S->sampler.c.method), id_str(S->sampler.c.sched),
		S->sampler.c.s_ancestral, S->sampler.c.s_noise, S->c.cfg_scale,
		S->sampler.n_step, S->sampler.nfe_per_step);

	// Denoising / generation / sampling
	TRY( dnsamp_sample(&S->sampler, &latent) );
	
	// Save latent
	const char *path_latent = "latent-out.tensor";  //TODO: option
	ltensor_save_path(&latent, path_latent);
	
	mlctx_free(&S->ctx);  //free memory

	// Decode  //TODO: option
	TRY( mlis_img_decode(S, &latent, &latent) );
	ltensor_to_image(&latent, &img);

	// Save
	TRY( mlis_gen_img_save(S, &img, ctx.nfe) );

	tm = timing_time() - tm;
	log_info("Generation done {%.3fs}", tm);

end:
	img_free(&img);
	ltensor_free(&tmpt);
	ltensor_free(&unlabel);
	ltensor_free(&label);
	ltensor_free(&uncond);
	ltensor_free(&cond);
	ltensor_free(&lmask);
	ltensor_free(&latent);
	mlctx_free(&S->ctx);
	return R;
}

/* Checks all the operations with deterministic inputs and prints the
 * resulting tensor sums. Useful to easily check if anything broke down
 * during development. The tests are independent of each other.
 */
int mlis_check(MLImgSynthApp* S)
{
	int R=1;
	LocalTensor lt={0}, lt2={0}, lt3={0}, lt4={0};
	Stream out={0};

	TRY( mlis_ml_init(S) );

	TRY( stream_open_std(&out, STREAM_STD_OUT, 0) );

	// Check text to embedding encode
	const char *prompt = "a photograph of an astronaut riding a horse";

	{
		bool has_tproj = tstore_tensor_get(&S->tstore, "clip.text.text_proj");
		TRY( mlis_clip_encode(S, prompt, &lt,
			has_tproj ? &lt2 : NULL, S->clip_p, "clip", true) );
		
		float embed_sum = ltensor_sum(&lt);
		stream_printf(&out, "CHECK clip embed: %.6g\n", embed_sum);
		ltensor_shape_check_log(&lt, "clip embed",
			S->clip_p->d_embed, S->clip_p->n_token, 1, 1);
		
		if (has_tproj) {
			float feat_sum = ltensor_sum(&lt2);
			stream_printf(&out, "CHECK clip feat: %.6g\n", feat_sum);
			ltensor_shape_check_log(&lt2, "clip feat",
				S->clip_p->d_embed, 1, 1, 1);
		}
	}

	if (S->clip2_p)
	{
		bool has_tproj = tstore_tensor_get(&S->tstore, "clip2.text.text_proj");
		TRY( mlis_clip_encode(S, prompt, &lt,
			has_tproj ? &lt2 : NULL, S->clip2_p, "clip2", true) );

		float embed_sum = ltensor_sum(&lt);
		stream_printf(&out, "CHECK clip2 embed: %.6g\n", embed_sum);
		ltensor_shape_check_log(&lt, "clip2 embed",
			S->clip2_p->d_embed, S->clip_p->n_token, 1, 1);

		if (has_tproj) {
			float feat_sum = ltensor_sum(&lt2);
			stream_printf(&out, "CHECK clip2 feat: %.6g\n", feat_sum);
			ltensor_shape_check_log(&lt2, "clip2 feat",
				S->clip2_p->d_embed, 1, 1, 1);
		}
	}

	// Check image to latent encode
	{
		ltensor_resize(&lt, 64, 64, 3, 1);
		for(unsigned n=ltensor_nelements(&lt), i=0; i<n; ++i)
			lt.d[i] = (float)i/(n-1);

		TRY( mlis_img_encode(S, &lt, &lt2) );
		float latent_sum = ltensor_sum(&lt2);
		stream_printf(&out, "CHECK vae latent: %.6g\n", latent_sum);

		if (lt2.s[2] == S->vae_p->ch_z*2) {
			sdvae_latent_mean(&lt2, &lt2, S->vae_p);
			float latent_sum = ltensor_sum(&lt2);
			stream_printf(&out, "CHECK vae latent mean: %.6g\n", latent_sum);
		}

		ltensor_shape_check_log(&lt2, "vae latent", lt.s[0]/S->vae_p->f_down,
			lt.s[1]/S->vae_p->f_down, S->vae_p->ch_z, 1);
	}

	// Check latent to image decode
	{
		ltensor_resize(&lt, 8, 8, 4, 1);
		for(unsigned n=ltensor_nelements(&lt), i=0; i<n; ++i)
			lt.d[i] = (float)i/(n-1);

		TRY( mlis_img_decode(S, &lt, &lt2) );
		float image_sum = ltensor_sum(&lt2);
		stream_printf(&out, "CHECK vae image: %.6g\n", image_sum);
		ltensor_shape_check_log(&lt2, "vae image", lt.s[0]*S->vae_p->f_down,
			lt.s[1]*S->vae_p->f_down, S->vae_p->ch_x, 1);
	}

	// Check UNet denoise
	{
		ltensor_resize(&lt, 24, 48, 4, 1);  //x
		for(unsigned n=ltensor_nelements(&lt), i=0; i<n; ++i)
			lt.d[i] = (float)i/(n-1) *2 - 1;
		
		ltensor_resize(&lt2, S->unet_p->n_ctx, 77, 1, 1);  //cond
		for(unsigned n=ltensor_nelements(&lt2), i=0; i<n; ++i)
			lt2.d[i] = (float)i/(n-1) *2 - 1;
		
		if (S->unet_p->ch_adm_in) {
			ltensor_resize(&lt3, S->unet_p->ch_adm_in, 1, 1, 1);  //label
			for(unsigned n=ltensor_nelements(&lt3), i=0; i<n; ++i)
				lt3.d[i] = (float)i/(n-1) *2 - 1;
		}

		float t = S->unet_p->n_step_train-1;
	
		UnetState unet={0};
		S->ctx.tprefix = "unet";
		TRY( unet_denoise_init(&unet, &S->ctx, S->unet_p, lt.s[0], lt.s[1],
			S->c.unet_split) );

		TRY( unet_denoise_run(&unet, &lt, &lt2, &lt3, t, &lt4) );
		float denoise_sum = ltensor_sum(&lt4);
		stream_printf(&out, "CHECK unet denoise: %.6g\n", denoise_sum);
		ltensor_shape_check_log(&lt4, "unet denoise", LT_SHAPE_UNPACK(lt));
	}

end:
	stream_close(&out, 0);
	ltensor_free(&lt4);
	ltensor_free(&lt3);
	ltensor_free(&lt2);
	ltensor_free(&lt);
	return R;
}

int mlis_backends_print()
{
	Stream out={0};

	TRYR( stream_open_std(&out, STREAM_STD_OUT, 0) );

	// List backends
	size_t nb = ggml_backend_reg_count();
	for (size_t ib=0; ib<nb; ++ib) {
		ggml_backend_reg_t br = ggml_backend_reg_get(ib);
		stream_printf(&out, "%s\n", ggml_backend_reg_name(br));
		
		// List devices
		size_t nd = ggml_backend_reg_dev_count(br);
		for (size_t id=0; id<nd; ++id) {
			ggml_backend_dev_t bd = ggml_backend_reg_dev_get(br, id);
			stream_printf(&out, "\t%s %s\n",
				ggml_backend_dev_name(bd),
				ggml_backend_dev_description(bd) );
		}
	}

	stream_close(&out, 0);
	return 1;
}

int main(int argc, char* argv[])
{
	int R=0, r;
	MLImgSynthApp app={0};

	ids_init();

	TRY( r = mlis_args_load(&app, argc, argv) );
	if (!r) return 0;

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

	if (!app.c.cmd) ;
	else if (app.c.cmd == ID_list_backends) {
		mlis_backends_print();
	}
	else if (app.c.cmd == ID_check) {
		TRY( mlis_check(&app) );
	}
	else if (app.c.cmd == ID_vae_encode) {
		TRY( mlis_vae_cmd(&app, true, false) );
	}
	else if (app.c.cmd == ID_vae_decode) {
		TRY( mlis_vae_cmd(&app, false, true) );
	}
	else if (app.c.cmd == ID_vae_test) {
		TRY( mlis_vae_cmd(&app, true, true) );
	}
	else if (app.c.cmd == ID_clip_encode) {
		TRY( mlis_clip_cmd(&app) );
	}
	else if (app.c.cmd == ID_generate) {
		TRY( mlis_generate(&app) );
	}
	else {
		ERROR_LOG(-1, "Unknown command '%s'", id_str(app.c.cmd));
	}

end:
	if (R<0) log_error("error exit: %x", -R);
	mlis_free(&app);
	return -R;
}
