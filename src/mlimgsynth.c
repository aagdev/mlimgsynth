/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * mlimgsynth main code.
 */
#include "ccommon/timing.h"
#include "ccommon/logging.h"
#include "ccommon/stream.h"
#include "ccommon/fsutil.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml_extend.h"

#include <math.h>

#include "ids.h"
#include "tensorstore.h"
#include "safetensors.h"
#include "localtensor.h"
#include "rng_philox.h"
#include "tae.h"
#include "vae.h"
#include "clip.h"
#include "unet.h"
#include "solvers.h"
#include "util.h"

#define APP_NAME_VERSION "mlimgsynth v0.2"

#define debug_pause() do { \
	puts("Press ENTER to continue"); \
	getchar(); \
} while (0)

//#define MLIS_DEBUG_LTENSOR_STATS
#ifdef MLIS_DEBUG_LTENSOR_STATS
#define debug_ltensor_stats(T, D) do { \
	if (ltensor_good(T)) { \
		float mn, mx = ltensor_minmax(T, &mn); \
		float avg = ltensor_mean(T); \
		unsigned n = ltensor_nelements(T); \
		log_debug("%s n:%u min:%.6g avg:%.6g max:%.6g", (D), n, mn, avg, mx); \
	} \
} while (0)
#else
#define debug_ltensor_stats(T, D)
#endif

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
	"  list-backends      List available GGML backends.\n"
	"  vae-encode         Encode an image to a latent.\n"
	"  vae-decode         Decode a latent to an image.\n"
	"  vae-test           Encode and decode an image.\n"
	"  clip-encode        Encode a prompt with the CLIP tokenizer and model.\n"
	"  generate           Generate an image.\n"
	"  check              Checks that all the operations (models) are working.\n"
	"\n"
	"Options:\n"
	"  -p TEXT            Prompt for text conditioning.\n"
	"  -n TEXT            Negative prompt for text unconditioning.\n"
	"  -b NAME            Select a backend for computation.\n"
	"  -t INT             Number of threads to use in the CPU backend.\n"
	"  -m PATH            Model file.\n"
	"  -i PATH            Input image or latent.\n"
	"  -2 PATH            Second input.\n"
	"  -o PATH            Output path.\n"
	"  -s --steps INT     Denoising steps with UNet.\n"
	"  -W INT             Image width. Default: 512 (SD1), 768 (SD2), 1024 (SDXL).\n"
	"  -H INT             Image height. Default: width.\n"
	"  -S --seed INT      RNG seed.\n"
	"\n"
	"  --method NAME      Sampling method: euler, heun, taylor3 (default).\n"
	"  --sched NAME       Sampling scheduler: uniform (default), karras.\n"
	"  --snoise FLOAT     Level of noise injection at each sampling step.\n"
	"  --cfg-scale FLOAT  Enables and sets the scale of the classifier-free guidance\n"
	"                     (default: 1).\n"
	"  --clip-skip INT    Number of CLIP layers to skip.\n"
	"                     Default: 1 (SD1), 2 (SD2/XL).\n"
	"  --f-t-ini FLOAT    Initial time factor (default 1).\n"
	"                     Use it to control the strength in img2img.\n"
	"  --f-t-end FLOAT    End time factor (default 0).\n"
	"  --tae PATH         Enables TAE and sets path to tensors.\n"
	"  --unet-split       Split each unet compute step compute in two parts to reduce memory usage.\n"
	"  --dump             Dumps models tensors and graphs.\n"
	"\n"
	"  -q                 Quiet: reduces information output.\n"
	"  -v                 Verbose: increases information output.\n"
	"  -d                 Enables debug output.\n"
	"  -h                 Print this message and exit.\n"
	"  --version          Print the version and exit.\n"
	;

typedef struct {
	MLCtx ctx;
	TensorStore tstore;
	Stream stm_model, stm_tae;
	ClipTokenizer tokr;

	const SdTaeParams *tae_p;
	const VaeParams *vae_p;
	const ClipParams *clip_p, *clip2_p;
	const UnetParams *unet_p;

	DynStr path_bin, tmps_path;

	struct {
		const char *path_model, *path_in, *path_in2, *path_out, *path_tae,
			*backend, *prompt, *nprompt;
		int cmd, n_thread, n_step, width, height, seed, method, sched, clip_skip;
		float s_noise, f_t_ini, f_t_end, cfg_scale;
		unsigned dump_info:1, use_tae:1, use_cfg:1, unet_split:1;
	} c;
} MLImgSynthApp;

int mlis_args_load(MLImgSynthApp* S, int argc, char* argv[])
{
	if (argc <= 1) {
		puts(help_string);
		return 1;
	}

	//TODO: validate input ranges

	int i, j;
	for (i=1; i<argc; ++i) {
		char * arg = argv[i];
		if (arg[0] == '-' && arg[1] == '-') {
			char * next = (i+1 < argc) ? argv[i+1] : "";
			if      (!strcmp(arg+2, "method")) {
				S->c.method = id_fromz(next); i++;
			}
			else if (!strcmp(arg+2, "sched" )) {
				S->c.sched = id_fromz(next); i++;
			}
			else if (!strcmp(arg+2, "snoise" )) {
				S->c.s_noise = atof(next); i++;
			}
			else if (!strcmp(arg+2, "steps" )) {
				S->c.n_step = atoi(next); i++;
			}
			else if (!strcmp(arg+2, "seed" )) {
				g_rng.seed = strtoull(next, NULL, 10); i++;
			}
			else if (!strcmp(arg+2, "cfg-scale" )) {
				S->c.cfg_scale = atof(next); i++;
				S->c.use_cfg = S->c.cfg_scale > 1;
			}
			else if (!strcmp(arg+2, "clip-skip" )) {
				S->c.clip_skip = atoi(next); i++;
			}
			else if (!strcmp(arg+2, "f-t-ini" )) {
				S->c.f_t_ini = atof(next); i++;
			}
			else if (!strcmp(arg+2, "f-t-end" )) {
				S->c.f_t_end = atof(next); i++;
			}
			else if (!strcmp(arg+2, "tae" )) {
				S->c.path_tae = next; i++;
				S->c.use_tae = S->c.path_tae && S->c.path_tae[0];
			}
			else if (!strcmp(arg+2, "unet-split" )) {
				S->c.unet_split = true;
			}
			else if (!strcmp(arg+2, "dump" )) {
				S->c.dump_info = true;
			}
			else if (!strcmp(arg+2, "version" )) {
				puts(version_string);
				return 1;
			}
			else {
				log_error("Unknown option '%s'", arg);
				return 1;
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
				case 't':  S->c.n_thread = atoi(next); i++; break;
				case 's':  S->c.n_step = atoi(next); i++; break;
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
					puts(help_string);
					return 1;
				default:
					log_error("Unknown option '-%c'", opt);
					return 1;
				}
			}
		}
		else if (!S->c.cmd) {
			S->c.cmd = id_fromz(arg);
		}
		else {
			log_error("Excess of arguments");
			return 1;
		}
	}
	
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

	return 0;
}

void mlis_free(MLImgSynthApp* S)
{
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

typedef struct TSNameConvEntry {
	const char *prefix, *replace;
	const struct TSNameConvEntry *sub;  //sub rules
	char skip_until;
} TSNameConvEntry;

#define TSNCSub  (const TSNameConvEntry[])

const TSNameConvEntry g_tnconv_oclip[] = {  //clip.text.
	{"transformer.resblocks.", "encoder.layers.", .skip_until='.',
	.sub=TSNCSub{
		{"attn.out_proj.", "self_attn.out_proj."},
		{"ln_1.", "layer_norm1."},
		{"ln_2.", "layer_norm2."},
		{"mlp.c_fc.", "mlp.fc1."},
		{"mlp.c_proj.", "mlp.fc2."},
		{}, //end mark
	}},
	{"positional_embedding", "embeddings.position_embedding.weight"},
	{"token_embedding.", "embeddings.token_embedding."},
	{"ln_final.", "final_layer_norm."},
	{}, //end mark
};

const TSNameConvEntry g_tensor_name_conv[] = {
	//SD1
	{"cond_stage_model.", "clip.", .sub=TSNCSub{
		{"transformer.text_model.", "text."},
		//SD2
		{"model.", "text.", .sub=g_tnconv_oclip},
		{}, //end mark
	}},
	{"first_stage_model.", "vae."},
	{"model.diffusion_model.", "unet."},

	//SDXL
	{"conditioner.embedders.0.", "clip.", .sub=TSNCSub{
		{"transformer.text_model.", "text."},
		{}, //end mark
	}},
	{"conditioner.embedders.1.model.", "clip2.text.", .sub=g_tnconv_oclip},
	{}, //end mark
};

#define TNC_match(PREFIX, REPLACE) \
	dstr_prefix_replace(*name, (PREFIX), (REPLACE), &pos)

/*static
int tname_conv(void*, DynStr* name)
{
	unsigned pos=0;
	if (TNC_match("cond_stage_model.", "clip.")) {
		if (TNC_match("transformer.text_model.", "text.")) ;
		//SD2
		else if (TNC_match("model.", "text."))
			TRY( tname_conv_oclip(*name, &pos) );
	}
	else if () {
	{"first_stage_model.", "vae."},
	{"model.diffusion_model.", "unet.", .sub=TSNCSub{
		{"input_blocks.", .skip_until='.', .sub=TSNCSub{
			{.skip_until='.', .sub=TSNCSub{
				{"transformer_blocks.", .skip_until='.', .sub=TSNCSub{
					{"attn", .skip_until='.', .sub=TSNCSub{
						{"to_q.", "q_proj."},
						{"to_k.", "k_proj."},
						{"to_v.", "v_proj."},
						{"to_out.", "out_proj."},
						{}, //end mark
					}},
					{}, //end mark
				}},
				{}, //end mark
			}},
			{}, //end mark
		}},
		{}, //end mark
	}},

	//SDXL
	{"conditioner.embedders.0.", "clip.", .sub=TSNCSub{
		{"transformer.text_model.", "text."},
		{}, //end mark
	}},
	{"conditioner.embedders.1.model.", "clip2.text.", .sub=g_tnconv_oclip},
	{}, //end mark
	}
}*/

int dstr_replace(DynStr* s, const char* p, const char* r)
{
	if (!p || !p[0]) return 0;
	unsigned slen = dstr_count(*s),
	         plen = strlen(p),
	         rlen = r ? strlen(r) : 0;
	int nrep=0;
	for (unsigned is=0, ip=0; is<slen; ++is) {
		if ((*s)[is] == p[ip]) {
			ip++;
			if (ip == plen) {
				ip=0;
				is++;
				if (plen > rlen) {
					is -= plen-rlen;
					dstr_remove(*s, is, plen-rlen);
					slen -= plen-rlen;
				}
				else if (plen < rlen) {
					dstr_insert(*s, is, rlen-plen, NULL);
					is += rlen-plen;
					slen += rlen-plen;
				}
				if (rlen > 0)
					memcpy(*s+is-rlen, r, rlen);
				nrep++;
				is--;
			}
		}
		else ip=0;  //TODO: p with repetitions (i.g. abb in aabbcc)
	}
	return nrep;
}

static
int tname_conv(void* p, DynStr* name)
{
	const TSNameConvEntry *conv=g_tensor_name_conv, *conv_next;
	unsigned icur=0;
	while (conv) {
		conv_next = NULL;
		unsigned i, p;
		const char *nm = *name + icur;
		for (i=0; conv[i].prefix; ++i) {
			const TSNameConvEntry *e = &conv[i];
			bool match = true;
			if (e->prefix) {
				for (p=0; e->prefix[p] == nm[p] && e->prefix[p]; ++p);
				match = (e->prefix[p] == 0);
			}
			if (match) {
				if (e->replace) {
					dstr_remove(*name, icur, p);
					dstr_insertz(*name, icur, e->replace);
					p = strlen(e->replace);
				}
				icur += p;
				if (e->skip_until) {
					nm = *name;
					while (nm[icur] && nm[icur] != e->skip_until) icur++;
					if (nm[icur]) icur++;
				}
				conv_next = e->sub;
				break;
			}
		}
		conv = conv_next;
	}

	//TODO: restrict better
	//TODO: multi-replace op
	//unet attn_mhead
	if (dstr_replace(name, ".to_q.", ".q_proj.")) ;
	else if (dstr_replace(name, ".to_k.", ".k_proj.")) ;
	else if (dstr_replace(name, ".to_v.", ".v_proj.")) ;
	else if (dstr_replace(name, ".to_out.0.", ".out_proj.")) ;
	//unet resnet
	else if (dstr_replace(name, ".in_layers.0.", ".norm1.")) ;
	else if (dstr_replace(name, ".in_layers.2.", ".conv1.")) ;
	else if (dstr_replace(name, ".out_layers.0.", ".norm2.")) ;
	else if (dstr_replace(name, ".out_layers.3.", ".conv2.")) ;
	else if (dstr_replace(name, ".out_layers.3.", ".conv2.")) ;
	else if (dstr_replace(name, ".emb_layers.1.", ".emb_proj.")) ;
	else if (dstr_replace(name, ".skip_connection.", ".skip_conv.")) ;
	//vae resnet
	else if (dstr_replace(name, ".nin_shortcut.", ".skip_conv.")) ;

	return 0;
}

static
int open_clip_attn_conv(TensorStore* ts)
{
	int R=1;
	DynStr tmps=NULL;
	TSTensorEntry new={0};

	vec_for(ts->tensors,i,0) {
		const TSTensorEntry *e = &ts->tensors[i];
		const char *name = id_str(e->key);
		unsigned nlen = strlen(name);
		if (nlen >= 20 && !memcmp(name, "clip", 4) &&
			(name[4] == '.' || (name[4] == '2' && name[5] == '.')) )
		{
			if (!memcmp(name+nlen-18, ".attn.in_proj_bias", 18))
			{
				if (!(e->shape_n==1 && e->shape[0] % 3 == 0))
					ERROR_LOG(-1, "invalid open_clip tensor '%s'", name);
				new = *e;
				new.shape[0] /= 3;
				new.size /= 3;
				nlen -= 18;
				
				dstr_copy(tmps, nlen, name);
				dstr_appendz(tmps, ".self_attn.q_proj.bias");
				tstore_tensor_add(ts, &tmps, &new);
				new.offset += new.size;
				
				dstr_copy(tmps, nlen, name);
				dstr_appendz(tmps, ".self_attn.k_proj.bias");
				tstore_tensor_add(ts, &tmps, &new);
				new.offset += new.size;
				
				dstr_copy(tmps, nlen, name);
				dstr_appendz(tmps, ".self_attn.v_proj.bias");
				tstore_tensor_add(ts, &tmps, &new);
			}
			else if (!memcmp(name+nlen-20, ".attn.in_proj_weight", 20))
			{
				if (!(e->shape_n==2 && e->shape[1] % 3 == 0))
					ERROR_LOG(-1, "invalid open_clip tensor '%s'", name);
				new = *e;
				new.shape[1] /= 3;
				new.size /= 3;
				nlen -= 20;
				
				dstr_copy(tmps, nlen, name);
				dstr_appendz(tmps, ".self_attn.q_proj.weight");
				tstore_tensor_add(ts, &tmps, &new);
				new.offset += new.size;
				
				dstr_copy(tmps, nlen, name);
				dstr_appendz(tmps, ".self_attn.k_proj.weight");
				tstore_tensor_add(ts, &tmps, &new);
				new.offset += new.size;
				
				dstr_copy(tmps, nlen, name);
				dstr_appendz(tmps, ".self_attn.v_proj.weight");
				tstore_tensor_add(ts, &tmps, &new);
			}
		}
	}

end:
	dstr_free(tmps);
	return R;
}

int mlis_ml_init(MLImgSynthApp* S)
{
	int R=1;
	assert(!S->ctx.backend);
	
	S->ctx.c.wtype = GGML_TYPE_F16;
	S->ctx.tstore = &S->tstore;

	// Backend init
	if (S->c.backend && S->c.backend[0])
		S->ctx.backend = ggml_backend_reg_init_backend_from_str(S->c.backend);	
	else 
		S->ctx.backend = ggml_backend_cpu_init();
	if (!S->ctx.backend) ERROR_LOG(-1, "ggml backend init");
	log_info("Backend: %s", ggml_backend_name(S->ctx.backend));

	if (ggml_backend_is_cpu(S->ctx.backend))
	{
		if (S->c.n_thread)
			ggml_backend_cpu_set_n_threads(S->ctx.backend, S->c.n_thread);
	}
#if USE_GGML_SCHED
	else
	{
		S->ctx.backend2 = ggml_backend_cpu_init();
		if (!S->ctx.backend2) ERROR_LOG(-1, "ggml fallback backend CPU init");
		log_debug("Fallback backend CPU");
		if (S->c.n_thread)
			ggml_backend_cpu_set_n_threads(S->ctx.backend2, S->c.n_thread);
	}
#endif

	// Model parameters header load
	S->tstore.cb_add = tname_conv;
	if (S->c.path_model) {
		log_debug("Loading model header from '%s'", S->c.path_model);
		TRY_LOG( stream_open_file(&S->stm_model, S->c.path_model,
			SOF_READ | SOF_MMAP),
			"could not open '%s'", S->c.path_model);
		log_debug("model stream class: %s", S->stm_model.cls->name);
		TRY( safet_load_head(&S->tstore, &S->stm_model, NULL) );
	}

	// TAE model load
	if (S->c.path_tae) {
		log_debug("Loading model header from '%s'", S->c.path_tae);
		TRY_LOG( stream_open_file(&S->stm_tae, S->c.path_tae,
			SOF_READ | SOF_MMAP),
			"could not open '%s'", S->c.path_tae);
		TRY( safet_load_head(&S->tstore, &S->stm_tae, "tae.") );
	}
	
	TRY( open_clip_attn_conv(&S->tstore) );  //TODO: name_conv callback?
		
	if (S->c.dump_info)
		TRY( tstore_info_dump_path(&S->tstore, "dump-tensors.txt") );

	// Identify model type
	const char *model_type=NULL;
	const TSTensorEntry *te=NULL;
	if ((te = tstore_tensor_get(&S->tstore,
		"unet.input_blocks.1.1.transformer_blocks.0.attn2.k_proj.weight")))
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
		"unet.input_blocks.4.1.transformer_blocks.0.attn2.k_proj.weight")))
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

	if (model_type)
		log_info("Model type: %s", model_type);
	else
		ERROR_LOG(-1, "Unknown model type");
	
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
		TRY( sdvae_encode(&S->ctx, S->vae_p, img, latent) );
	}
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
		TRY( sdvae_decode(&S->ctx, S->vae_p, latent, img) );
	}
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
		debug_ltensor_stats(&latent, "latent");
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

	bool has_tproj = tstore_tensor_get(&S->tstore, "clip.text.text_projection");

	TRY( mlis_ml_init(S) );

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
	
	TRYR( unet_denoise_run(A->ctx, x, A->cond, A->label, t, dx) );
	
	if (A->app->c.use_cfg) {
		TRYR( unet_denoise_run(A->ctx, x, A->uncond, A->unlabel, t, A->tmpt) );
		float f = A->app->c.cfg_scale;
		ltensor_for(*dx,i,0) dx->d[i] = dx->d[i]*f + A->tmpt->d[i]*(1-f);
	}
	
	return 1;
}

int mlis_generate(MLImgSynthApp* S)
{
	int R=1;
	UnetState ctx={0};
	Image img={0};
	DynStr infotxt=NULL;
	LocalTensor latent={0}, noise={0}, 
	            cond={0}, label={0},
				uncond={0}, unlabel={0};
	
	unet_params_init();  //global
	
	TRY( mlis_ml_init(S) );

	// Latent load
	if (S->c.path_in) {
		const char *ext = path_ext(S->c.path_in);
		if (!strcmp(ext, "tensor")) {  //latent
			TRY( ltensor_load_path(&latent, S->c.path_in) );
		}
		else {  //image
			log_debug("Loading image from '%s'", S->c.path_in);
			img_load_file(&img, S->c.path_in);
			ltensor_from_image(&latent, &img);
			TRY( mlis_img_encode(S, &latent, &latent) );
		}
		
		// Sample if needed
		if (latent.s[2] == S->unet_p->n_ch_in*2)
			sdvae_latent_sample(&latent, &latent, S->vae_p);

		TRY( ltensor_shape_check_log(&latent, "input latent",
			0, 0, S->unet_p->n_ch_in, 1) );
		
		debug_ltensor_stats(&latent, "input latent");

		S->c.width  = latent.s[0] * S->vae_p->f_down;
		S->c.height = latent.s[1] * S->vae_p->f_down;
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

	// Conditioning load
	if (S->c.path_in2){
		TRY( ltensor_load_path(&cond, S->c.path_in2) );
		TRY( ltensor_shape_check_log(&cond, "input conditioning",
			S->unet_p->n_ctx, 0, 1, 1) );
	}
	else if (S->c.prompt)
		TRY( mlis_sd_cond_get(S, S->c.prompt, &cond, &label) );
	else
		ERROR_LOG(-1, "no conditioning (option -2) or prompt (option -p) set");
	
	debug_ltensor_stats(&cond, "cond");
	//ltensor_save_path(&cond, "cond.tensor");
	debug_ltensor_stats(&label, "label");

	// Negative conditioning
	if (S->c.use_cfg) {
		TRY( mlis_sd_cond_get(S, S->c.nprompt, &uncond, &unlabel) );
		if (S->unet_p->uncond_empty_zero && !(S->c.nprompt && S->c.nprompt[0]))
			ltensor_for(uncond,i,0) uncond.d[i] = 0;
	}
	else if (S->c.nprompt && S->c.nprompt[0])
		log_warning("negative prompt provided but CFG is not enabled");	
	
	debug_ltensor_stats(&uncond, "uncond");
	debug_ltensor_stats(&unlabel, "unlabel");

	// Sampling method
	Solver sol={0};
	IFFALSESET(S->c.method, ID_taylor3);
	for (unsigned i=0; g_solvers[i]; ++i)
		if (S->c.method == g_solvers[i]->name) {
			sol.C = g_solvers[i];
		}
	if (!sol.C) ERROR_LOG(-1, "Invalid method '%s'", id_str(S->c.method));

	struct dxdt_args A = { .app=S, .ctx=&ctx, .tmpt=&noise,
		.cond=&cond, .uncond=&uncond, .label=&label, .unlabel=&unlabel };
	sol.dxdt = mlis_denoise_dxdt;
	sol.user = &A;

	// Scheduling
	// Compute times and sigmas for inference
	int n_step = S->c.n_step;
	if (n_step < 1) n_step = 20;
	if (sol.C->n_fe > 1)
		n_step = (n_step + sol.C->n_fe-1) / sol.C->n_fe;
	
	int n_nfe_s = sol.C->n_fe;
	if (S->c.use_cfg) n_nfe_s *= 2;
	
	float *sigmas=NULL;
	vec_resize(sigmas, n_step+1);
	sigmas[n_step] = 0;

	IFNPOSSET(S->c.f_t_ini, 1);
	float t_ini = (S->unet_p->n_step_train - 1) * S->c.f_t_ini;
	float t_end = (S->unet_p->n_step_train - 1) * S->c.f_t_end;

	IFFALSESET(S->c.sched, ID_uniform);
	switch (S->c.sched) {
	case ID_uniform: {
		float b = t_ini,
		      f = n_step>1 ? (t_end-t_ini)/(n_step-1) : 0;
		for (unsigned i=0; i<n_step; ++i)
			sigmas[i] = unet_t_to_sigma(S->unet_p, b+i*f);
	} break;
	case ID_karras: {
		// Uses the model's min and max sigma instead of 0.1 and 10.
		float smin = unet_t_to_sigma(S->unet_p, t_end),
		      smax = unet_t_to_sigma(S->unet_p, t_ini),
			  p=7,
		      sminp = pow(smin, 1/p),
		      smaxp = pow(smax, 1/p),
			  b = smaxp,
			  f = n_step>1 ? (sminp - smaxp) / (n_step-1) : 0;
		for (unsigned i=0; i<n_step; ++i)
			sigmas[i] = pow(b+i*f, p);
	} break;
	default:
		ERROR_LOG(-1, "Unknown scheduler '%s'", id_str(S->c.sched));
	}

	//log_debug_vec("Times" , times , i, 0, "%.6g", times[i]);
	log_debug_vec("Sigmas", sigmas, i, 0, "%.6g", sigmas[i]);

	// Add noise to initial latent
	ltensor_resize_like(&noise, &latent);
	rng_randn(ltensor_nelements(&noise), noise.d);
	ltensor_for(latent,i,0) latent.d[i] += noise.d[i] * sigmas[0];
	debug_ltensor_stats(&latent, "latent+noise");

	if (!(0 <= S->c.s_noise && S->c.s_noise < 1))
		ERROR_LOG(-1, "snoise out of range");
	
	// Prepare computation
	S->ctx.tprefix = "unet";
	TRY( unet_denoise_init(&ctx, &S->ctx, S->unet_p, latent.s[0], latent.s[1],
		S->c.unet_split) );
	ctx.n_step = n_step;
	
	log_info("Generating "
		"(solver: %s, sched: %s, snoise: %g, steps: %d, nfe/s: %d)",
		id_str(S->c.method), id_str(S->c.sched), S->c.s_noise, n_step, n_nfe_s);
	sol.t = sigmas[0];  //initial t
	for (int s=0; s<n_step; ++s)
	{
		if (S->c.s_noise > 0 && s > 0) {
			// Stochastic sampling can help to add detail lost during sampling.
			// See Karras2022 Algo2
			rng_randn(ltensor_nelements(&noise), noise.d);
			float f = S->c.s_noise,
			      s_curr  = sol.t,
			      s_hat   = sqrt(s_curr*s_curr + f*f);
			log_debug("sigma_hat_%d=%g f=%g", s, s_hat, f);
			ltensor_for(latent,i,0) latent.d[i] += noise.d[i] * f;
			sol.t = s_hat;
		}
		ctx.i_step = s;
		TRY( solver_step(&sol, sigmas[s+1], &latent) );
		debug_ltensor_stats(&latent, "step latent");
	}

	// Save latent
	const char *path_latent = "latent-out.tensor";  //TODO: option
	ltensor_save_path(&latent, path_latent);
	
	mlctx_free(&S->ctx);  //free memory

	// Decode  //TODO: option
	TRY( mlis_img_decode(S, &latent, &latent) );
	ltensor_to_image(&latent, &img);

	// Make info text
	// Imitates stable-diffusion-webui create_infotext
	if (S->c.prompt)
		dstr_printfa(infotxt, "%s\n", S->c.prompt);
	//TODO: input latent or image filename?
	dstr_printfa(infotxt, "Seed: %"PRIu64, g_rng.seed);
	dstr_printfa(infotxt, ", Sampler: %s", id_str(S->c.method));
	dstr_printfa(infotxt, ", Schedule type: %s", id_str(S->c.sched));
	if (S->c.s_noise > 0)
		dstr_printfa(infotxt, ", SNoise: %s", S->c.s_noise);
	if (S->c.use_cfg)
		dstr_printfa(infotxt, ", CFG scale: %g", S->c.cfg_scale);
	dstr_printfa(infotxt, ", Steps: %u", n_step);
	dstr_printfa(infotxt, ", NFE: %u", ctx.nfe);
	dstr_printfa(infotxt, ", Size: %ux%u", img.w, img.h);
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
	TRY( img_save_file_info(&img, S->c.path_out, "parameters", infotxt) );

end:
	dstr_free(infotxt);
	img_free(&img);
	ltensor_free(&noise);
	ltensor_free(&uncond);
	ltensor_free(&cond);
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
		bool has_tproj = tstore_tensor_get(&S->tstore, "clip.text.text_projection");
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
		bool has_tproj = tstore_tensor_get(&S->tstore, "clip2.text.text_projection");
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

int main(int argc, char* argv[])
{
	int R=0;
	MLImgSynthApp app={0};

	ids_init();

	if (mlis_args_load(&app, argc, argv)) return 11;

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
		size_t n = ggml_backend_reg_get_count();
		for (size_t i=0; i<n; ++i)
			printf("%s\n", ggml_backend_reg_get_name(i));
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
	if (R<0) log_error("error exit: %d", R);
	mlis_free(&app);
	return -R;
}
