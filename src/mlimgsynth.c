/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 */
#define MLIS_IMPLEMENTATION 1
typedef struct LocalTensor MLIS_Tensor;
#include "mlimgsynth.h"

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

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml_extend.h"

#include <math.h>

#define ERROR_HANDLE_BEGIN \
	int R=1; \
	dstr_resize(S->errstr, 0);

#define ERROR_HANDLE_END(FUNC_NAME) \
	if (R<0 && dstr_empty(S->errstr)) \
		dstr_printf(S->errstr, "error 0x%x in %s", -R, FUNC_NAME); \
	if (R<0 && S->errh) mlis_error_handle(S, R); \
	return R;

#undef ERROR_LOG
#define ERROR_LOG(CODE, ...) do { \
	dstr_printf(S->errstr, __VA_ARGS__); \
	log_error(S->errstr); \
	RETURN(CODE); \
} while(0)

#define log_vec(LVL,DESC,VEC,VAR,I0,...) \
if (log_level_check(LVL)) { \
	log_line_begin(LVL); \
	log_line_str(DESC ":"); \
	vec_for(VEC,VAR,I0) log_line_strf(" " __VA_ARGS__); \
	log_line_end(); \
}

#define log_debug_vec(...)  log_vec(LOG_LVL_DEBUG, __VA_ARGS__)

//TODO: try to remove this in the future...
bool global_init_done = false;

/* Tensor interface wrappers
 */

void mlis_tensor_free(MLIS_Tensor* S) {
	ltensor_free(S);
}

size_t mlis_tensor_count(const MLIS_Tensor* S) {
	return ltensor_nelements(S);
}

void mlis_tensor_resize(MLIS_Tensor* S, int n0, int n1, int n2, int n3) {
	ltensor_resize(S, n0, n1, n2, n3);
}

void mlis_tensor_resize_like(MLIS_Tensor* S, const MLIS_Tensor* src) {
	ltensor_resize_like(S, src);
}

void mlis_tensor_copy(MLIS_Tensor* S, const MLIS_Tensor* src) {
	ltensor_copy(S, src);
}

float mlis_tensor_similarity(const MLIS_Tensor* t1, const MLIS_Tensor* t2)
{
	size_t n1 = ltensor_nelements(t1);
	size_t n2 = ltensor_nelements(t2);
	if (n1 != n2) return NAN;
	double v11=0, v22=0, v12=0;
	for (size_t i=0; i<n1; ++i) v11 += t1->d[i] * t1->d[i];
	for (size_t i=0; i<n2; ++i) v22 += t2->d[i] * t2->d[i];
	for (size_t i=0; i<n1; ++i) v12 += t1->d[i] * t2->d[i];
	return v12 / sqrt(v11 * v22);
}

static
void mlis_image_resize(MLIS_Image* I, unsigned w, unsigned h, unsigned c)
{
	I->w = w;
	I->h = h;
	I->c = c;
	I->sz = (size_t)I->w * I->h * I->c;
	if (!(I->flags & LT_F_OWNMEM))
		I->d = NULL;
	I->d = alloc_realloc(g_allocator, I->d, I->sz);
	I->flags |= LT_F_OWNMEM;
}

static
void mlis_tensor_to_image(const MLIS_Tensor* T, MLIS_Image* I, int idx)
{
	int n0=T->n[0], n1=T->n[1], n2=T->n[2];
	mlis_image_resize(I, n0, n1, n2);
	
	const float *td = T->d + n0*n1*n2 * idx;
	uint8_t *id = I->d;
	for (int y=0; y<n1; ++y) {
		for (int x=0; x<n0; ++x) {
			for (int c=0; c<n2; ++c) {
				float v = td[n0*n1*c +n0*y +x] * 255;
				ccCLAMP(v, 0, 255);
				id[n0*n2*y +n2*x +c] = v;
			}
		}
	}
}

static
int mlis_tensor_from_image(MLIS_Tensor* T, const MLIS_Image* I)
{
	int n0=I->w, n1=I->h, n2=I->c;
	if (!(n0 * n1 * n2 > 0 && I->d != NULL)) return MLIS_E_IMAGE;
	mlis_tensor_resize(T, n0, n1, n2, 1);  //TODO: n_batch?
	
	float *td = T->d;// + n0*n1*n2 * idx;
	const uint8_t *id = I->d;
	float f = 1 / 255.0;
	for (int y=0; y<n1; ++y) {
		for (int x=0; x<n0; ++x) {
			for (int c=0; c<n2; ++c) {
				float v = (float)id[n0*n2*y +n2*x +c] * f;
				td[n0*n1*c +n0*y +x] = v;
			}
		}
	}

	return 1;
}

/* Enums
 */

// String comparison for ID's. It's case insensitive and allows '-' for '_'.
static
int strsl_cmpz_id(const StrSlice ss, const char* strz)
{
	for (const char *cur=ss.b, *end=cur+ss.s; cur<end; ++cur, ++strz) {
		int c = *cur;
		if (!*strz) return c ? c : 1;
		if ('A' <= c && c <= 'Z') c -= 'A';
		else if (c == '-') c = '_';
		else if (c == '+') c = 'p';  // Just for dpm++
		int d = c - *strz;
		if (d) return d;
	}
	return *strz;
}

#define IMPL_ENUM_FUNC(NAME, TYPE, DEFAULT) \
const char * mlis_##NAME##_str(TYPE x) { \
	if (!(0 <= x && x < COUNTOF(g_##NAME##_str))) return "???"; \
	return g_##NAME##_str[x]; \
} \
TYPE mlis_##NAME##_froms(const StrSlice ss) { \
	for (int x=0; x<COUNTOF(g_##NAME##_str); ++x) \
		if (!strsl_cmpz_id(ss, g_##NAME##_str[x])) \
			return x; \
	return (TYPE) DEFAULT; \
} \
TYPE mlis_##NAME##_fromz(const char* str) { \
	return mlis_##NAME##_froms(strsl_fromz(str)); \
}

#define IMPL_ENUM_FUNC_KV(NAME, TYPE, DEFAULT) \
const char * mlis_##NAME##_str(TYPE id) { \
	for (int i=0; i<COUNTOF(g_##NAME##_kv); ++i) \
		if ((int)id == g_##NAME##_kv[i].v) \
			return g_##NAME##_kv[i].k; \
	return "???"; \
} \
TYPE mlis_##NAME##_froms(const StrSlice ss) { \
	for (int i=0; i<COUNTOF(g_##NAME##_kv); ++i) \
		if (!strsl_cmpz_id(ss, g_##NAME##_kv[i].k)) \
			return g_##NAME##_kv[i].v; \
	return (TYPE) DEFAULT; \
} \
TYPE mlis_##NAME##_fromz(const char* str) { \
	return mlis_##NAME##_froms(strsl_fromz(str)); \
}

const char * g_stage_str[] = {
	"idle",
	"cond_encode",
	"image_encode",
	"image_decode",
	"denoise",
};

const char * g_method_str[] = {
	"none",
	"euler",
	"heun",
	"taylor3",
	"dpmpp2m",
	"dpmpp2s",
};

const char * g_sched_str[] = {
	"none",
	"uniform",
	"karras",
};

struct { const char *k; int v; } g_loglvl_kv[] = {
	{ "none", MLIS_LOGLVL_NONE },
	{ "error", MLIS_LOGLVL_ERROR },
	{ "warning", MLIS_LOGLVL_WARNING },
	{ "info", MLIS_LOGLVL_INFO },
	{ "verbose", MLIS_LOGLVL_VERBOSE },
	{ "debug", MLIS_LOGLVL_DEBUG },
	{ "max", MLIS_LOGLVL_MAX },
};

const char * g_option_str[MLIS_OPT__LAST+1] = {
	"none",
	"backend",
	"model",
	"tae",
	"lora_dir",
	"lora",
	"lora_clear",
	"prompt",
	"nprompt",
	"image_dim",
	"batch_size",
	"clip_skip",
	"cfg_scale",
	"method",
	"scheduler",
	"steps",
	"f_t_ini",
	"f_t_end",
	"s_noise",
	"s_ancestral",
	"image",
	"image_mask",
	"no_decode",
	"tensor_use_flags",
	"seed",
	"vae_tile",
	"unet_split",
	"threads",
	"dump_flags",
	"aux_dir",
	"callback",
	"error_handler",
	"log_level",
};

IMPL_ENUM_FUNC(stage, MLIS_Stage, -1)
IMPL_ENUM_FUNC(method, MLIS_Method, -1)
IMPL_ENUM_FUNC(sched, MLIS_Scheduler, -1)
IMPL_ENUM_FUNC(option, MLIS_Option, -1)
IMPL_ENUM_FUNC_KV(loglvl, MLIS_LogLvl, -1)

/* Internal state
 */
typedef struct MLIS_Ctx {
	MLCtx ctx;
	TensorStore tstore;
	Stream stm_model, stm_tae;
	ClipTokenizer tokr;
	DenoiseSampler sampler;  // Sampler options inside
	StringStore ss;

	const SdTaeParams *tae_p;
	const VaeParams *vae_p;
	const ClipParams *clip_p, *clip2_p;
	const UnetParams *unet_p;

	DynStr path_bin;

	// Active loras
	struct MLIS_LoraCfg {
		DynStr path;
		float mult;
		int flags;
	} *loras;  //vector

	// After a function returns an error (<0), this will have its description.
	// Don't overwrite.
	char *errstr;  //dynstr
	
	// Last generated image parameters textual description to be incluided in the
	// metadata. Don't overwrite.
	char *infotext;  //dynstr
	
	// Tensor used and produced during generation.
	// Managed internally, there is no need to explicitly create or free them.
	MLIS_Tensor image, mask,
				latent, lmask,
				cond, label,
				ncond, nlabel,
				t_tmp[4];
	
	// Image for mlis_image_get
	MLIS_Image imgex;
	
	// Backend info structure for mlis_backend_info_get
	MLIS_BackendInfo backend_info;

	// User callback called after each step in the process.
	// If it returns non-zero, the generation is cancelled.
	// Use it to show the progress and to allow cancelation.
	// Could be also used to implement advanced features modifying the tensors.
	MLIS_Callback callback;
	void *callback_ud;  // User-reserved data to be used by the callback

	// User-specified error handler
	MLIS_ErrorHandler errh;
	void *errh_ud;

	// Current progress in the image generation.
	// Use it to show the progress from the callback, or to check the state
	// of the session.
	MLIS_Progress prg;

	// Ready flags: keeps track of initializations
	int rflags;

	// Configuration, fill it before generation.
	// Most of them can be left in zero and default values will be used.
	struct MLIS_Config {
		DynStr backend,        // Backend name, passed to GGML.
		       be_params,      // Backend parameters, passed to GGML.
		       path_model,     // Path to model file. Mandatory.
		       path_tae,       // Path to TAE model.
		       path_lora_dir,  // Path to the directory with LoRA files.
			   path_aux;       // Path to auxiliary file (e.g. clip vocabulary)
		       //path_vae,
		       //path_clip,
		
		// Textual conditioning for the image generation.
		// Can also configure LoRA's putting anywhere "<lora:NAME:MULT>".
		DynStr prompt;
		
		// Negative prompt. Used only if cfg_scale > 1.
		DynStr nprompt;
	
		int			width,      // Image width in pixels
					height,     // Image height in pixels
					clip_skip,
					vae_tile,   // Reduces memory usage, try with 512
					n_batch,    // Number of images to generate simultaneously
					n_thread;

		float		cfg_scale;
		int			flags;  //MLIS_CF_*

		int dump_flags;

		// Tensor use flags
		int tuflags;  //MLIS_TUF_*
		
		// Sampler options
		//MLIS_Method		method;
		//MLIS_Scheduler	sched;
		//int				n_step;   // Number of steps
		//float			f_t_ini,  // 1 for txt2img, <1 for img2img
		//				f_t_end,  // Should be zero
		//				s_noise,
		//				s_ancestral;  // 1 for ancestral methods
	} c;

	// Used to validated the context
	int signature;
#define CTX_SIGNATURE 0xb5e884d7
} MLIS_Ctx;

enum MLIS_ConfigFlag {
	// Split unet model in two parts to reduce the memory usage.
	MLIS_CF_UNET_SPLIT		= 1,
	// Use by default the TAE instead of the VAE.
	// Faster and reduced memory usage, but lower quality.
	MLIS_CF_USE_TAE			= 2,
	// Do not decode the latent image after generation
	MLIS_CF_NO_DECODE		= 4,
	//MLIS_CF_PROMPT_NO_PROC
};

enum MLIS_DumpFlag {
	MLIS_DUMP_MODEL		= 1,
	MLIS_DUMP_LORA		= 2,
	MLIS_DUMP_GRAPH		= 4,
};

enum MLIS_ReadyFlag {
	MLIS_READY_BACKEND		= 1,
	MLIS_READY_MODEL		= 2,
	MLIS_READY_LORAS		= 4,
	MLIS_READY_RNG			= 8,
};

enum MLIS_LoraFlag {
	MLIS_LF_PROMPT = 1,
};

MLIS_Ctx* mlis_ctx_create_i(int version)
{
	if (!global_init_done) {
		unet_params_init();
	
		g_rng.seed = timing_timeofday() * 1000;  //TODO: local rng

		g_logger.prefix = "[MLIS] ";
#ifdef NDEBUG
		log_level_set(0);  // No log output by default
#endif

		global_init_done = true;
	}
	
	MLIS_Ctx *S = alloc_alloc(g_allocator, sizeof(MLIS_Ctx));  //zero'ed
	S->signature = CTX_SIGNATURE;
	S->ctx.tstore = &S->tstore;
	S->ctx.ss = &S->ss;
	S->tstore.ss = &S->ss;

	// Default options
	S->ctx.c.wtype = GGML_TYPE_F16;
	S->c.cfg_scale = 7;  //TODO: is it possible to detect a model-optimal value?
	
	return S;
}

static
void mlis_cfg_loras_free(MLIS_Ctx* S)
{
	vec_for(S->loras,i,0)
		dstr_free(S->loras[i].path);

	vec_free(S->loras);

	S->rflags &= ~MLIS_READY_LORAS;
}

static
void mlis_free(MLIS_Ctx* S)
{
	//TODO: use a local allocator and free all at once?
	dstr_free(S->errstr);
	dstr_free(S->infotext);
	dstr_free(S->c.backend);
	dstr_free(S->c.be_params);
	dstr_free(S->c.path_model);
	dstr_free(S->c.path_tae);
	dstr_free(S->c.path_lora_dir);
	dstr_free(S->c.prompt);
	dstr_free(S->c.nprompt);
	dstr_free(S->path_bin);

	vec_free(S->backend_info.devs);

	if (S->imgex.flags & LT_F_OWNMEM)
		alloc_free(g_allocator, S->imgex.d);

	ltensor_free(&S->image);
	ltensor_free(&S->mask);
	ltensor_free(&S->latent);
	ltensor_free(&S->lmask);
	ltensor_free(&S->cond);
	ltensor_free(&S->label);
	ltensor_free(&S->ncond);
	ltensor_free(&S->nlabel);

	dnsamp_free(&S->sampler);
	mlctx_free(&S->ctx);
	clip_tokr_free(&S->tokr);
	stream_close(&S->stm_tae, 0);
	tstore_free(&S->tstore);
	stream_close(&S->stm_model, 0);
	strsto_free(&S->ss);
	if (S->ctx.backend)
		ggml_backend_free(S->ctx.backend);

	mlis_cfg_loras_free(S);
}

void mlis_ctx_destroy(MLIS_Ctx** pS)
{
	if (!*pS) return;
	if ((*pS)->signature != CTX_SIGNATURE) {
		log_error("mlis ctx invalid signature");
		return;
	}
	mlis_free(*pS);
	alloc_free(g_allocator, *pS);
	*pS = NULL;
}

const char* mlis_errstr_get(const MLIS_Ctx* S)
{
	return S->errstr;
}

MLIS_Tensor* mlis_tensor_get(MLIS_Ctx* S, MLIS_TensorId id)
{
	switch (id) {
	case MLIS_TENSOR_IMAGE	: return &S->image;
	case MLIS_TENSOR_MASK	: return &S->mask;
	case MLIS_TENSOR_LATENT	: return &S->latent;
	case MLIS_TENSOR_LMASK	: return &S->lmask;
	case MLIS_TENSOR_COND	: return &S->cond;
	case MLIS_TENSOR_LABEL	: return &S->label;
	case MLIS_TENSOR_NCOND	: return &S->ncond;
	case MLIS_TENSOR_NLABEL	: return &S->nlabel;
	default:
		if (id >= MLIS_TENSOR_TMP) {
			unsigned idx = (unsigned)id - MLIS_TENSOR_TMP;
			if (idx < COUNTOF(S->t_tmp))
				return &S->t_tmp[idx];
		}
		return NULL;
	}
}

const MLIS_BackendInfo* mlis_backend_info_get(MLIS_Ctx* ctx, unsigned idx,
	int flags)
{
	unsigned n_backend = ggml_backend_reg_count();
	if (!(idx < n_backend)) return NULL;
	
	ggml_backend_reg_t br = ggml_backend_reg_get(idx);
	if (br == NULL) return NULL;

	MLIS_BackendInfo *bi = &ctx->backend_info;
	bi->name = ggml_backend_reg_name(br);
	bi->n_dev = ggml_backend_reg_dev_count(br);
	vec_resize(bi->devs, bi->n_dev);

	vec_for(bi->devs, idev, 0) {
		ggml_backend_dev_t bd = ggml_backend_reg_dev_get(br, idev);
		bi->devs[idev].name = ggml_backend_dev_name(bd);
		bi->devs[idev].desc = ggml_backend_dev_description(bd);
		size_t mfree=0, mtotal=0;
		ggml_backend_dev_memory(bd, &mfree, &mtotal);
		bi->devs[idev].mem_free = mfree;
		bi->devs[idev].mem_total = mtotal;
		//TODO: check capabilities?
	}

	return bi;
}

static
void mlis_prompt_clear(MLIS_Ctx* S)
{
	dstr_resize(S->c.prompt, 0);
	dstr_resize(S->c.nprompt, 0);
	S->sampler.c.f_t_ini = 1;
	S->sampler.c.f_t_end = 0;
	S->c.tuflags = 0;
}

static
void mlis_progress_reset(MLIS_Ctx* S)
{
	S->prg = (MLIS_Progress){
		.time = timing_time(),
	};
}

static
int mlis_callback(MLIS_Ctx* S, MLIS_Stage stage, int step, int step_end)
{
	S->prg.stage = stage;
	S->prg.step = step;
	S->prg.step_end = step_end;
	S->prg.step_time = timing_tic(&S->prg.time);
	return (S->callback) ? S->callback(S->callback_ud, S, &S->prg) : 0;
}

static
void mlis_error_handle(MLIS_Ctx* S, int code)
{
	if (dstr_empty(S->errstr))
		dstr_printf(S->errstr, "mlis error 0x%x", -code);

	MLIS_ErrorInfo ei = {
		.code = code,
		.desc = S->errstr,
	};
	S->errh(S->errh_ud, S, &ei);
}

static
int mlis_lora_path_find(MLIS_Ctx* S, const StrSlice name, DynStr *out)
{
	int R=1;

	// 1. Check if name is a valid path
	dstr_copy(*out, strsl_len(name), strsl_begin(name));
	if (file_exists(*out))
		RETURN(1);

	// 2. Make path
	dstr_copyd(*out, S->c.path_lora_dir);

	if (dstr_count(*out) > 0) {  // Add path separator at end if needed
		char c = (*out)[dstr_count(*out)-1];
		if (c != '/' && c != '\\')
			dstr_push(*out, '/');
	}
	
	dstr_append(*out, name.s, name.b);
	dstr_appendz(*out, ".safetensors");  //TODO: support other

	if (file_exists(*out))
		RETURN(1);
	
	// 3. Not found
	ERROR_LOG(MLIS_E_FILE_NOT_FOUND, "lora model file not found '%s'", *out);
	//TODO: more sophisticated file search? multiple dirs? subdirs?

end:
	return R;
}

static
int mlis_cfg_lora_add(MLIS_Ctx* S, const StrSlice path, float mult, int flags)
{
	int R=1;

	vec_append_zero(S->loras, 1);
	struct MLIS_LoraCfg *item = &vec_last(S->loras, 0);

	TRY( mlis_lora_path_find(S, path, &item->path) );
	item->mult = mult;
	item->flags = flags;
		
	S->rflags &= ~MLIS_READY_LORAS;

end:
	return R;
}

static
int mlis_cfg_prompt_option_parse(MLIS_Ctx* S, StrSlice ss)
{
	int R=1;
	DynStr tmps=NULL;

	if (strsl_prefix_trim(&ss, strsl_static("lora:")))
	{
		const char *sep=ss.b, *end=ss.b+ss.s;
		while (sep < end && *sep != ':') sep++;  // Find multiplier option
		
		float mult=1;
		if (*sep == ':') {  // Optional multiplier
			char *tail=NULL;
			mult = strtof(sep+1, &tail);
			if (tail != end)
				ERROR_LOG(MLIS_E_OPT_VALUE, "invalid lora multiplier");
		}
		
		TRY( mlis_cfg_lora_add(S, strsl_fromr(ss.b, sep), mult, MLIS_LF_PROMPT) );
	}
	else {
		ERROR_LOG(MLIS_E_PROMPT_OPT, "unknown prompt option '%.*s'",
			(int)ss.s, ss.b);
	}

end:
	dstr_free(tmps);
	return R;
}

static
void mlis_cfg_loras_prompt_remove(MLIS_Ctx* S)
{
	// Remove previous prompt lora's
	//TODO: optimize for the case multiple generations with the same loras
	for (unsigned i=0; i<vec_count(S->loras); ++i) {
		if (S->loras[i].flags & MLIS_LF_PROMPT) {
			vec_remove(S->loras, i, 1);
			i--;
			S->rflags &= ~MLIS_READY_LORAS;
		}
	}
}

static
int mlis_cfg_prompt_set(MLIS_Ctx* S, const StrSlice ss)
{
	int R=1;
	unsigned n_opt=0;

	mlis_cfg_loras_prompt_remove(S);
	
	dstr_realloc(S->c.prompt, strsl_len(ss));  //reserve memory
	dstr_resize(S->c.prompt, 0);

	// Prompt parsing
	for (const char *cur=strsl_begin(ss), *end=strsl_end(ss); cur<end; ++cur) {
		if (*cur == '<') {
			const char *e=cur+1;
			while (e < end && *e != '>') ++e;
			if (*e != '>')
				ERROR_LOG(MLIS_E_PROMPT_OPT, "prompt: '<' not matched with '>'");
			TRY( mlis_cfg_prompt_option_parse(S, strsl_fromr(cur+1, e)) );
			cur = e;
			n_opt++;
		}
		else dstr_push(S->c.prompt, *cur);
	}

	if (n_opt > 0)
		log_debug("Prompt options: %u", n_opt);

end:
	return R;
}

static
int mlis_file_find(MLIS_Ctx* S, const char* name, DynStr* out)
{
	int R=1;

	dstr_printf(*out, "%s", name);
	if (file_exists(*out)) goto end;

	if (!dstr_empty(S->c.path_aux)) {
		dstr_printf(*out, "%s/%s", S->c.path_aux, name);
		if (file_exists(*out)) goto end;
	}
	
	if (!dstr_empty(S->path_bin)) {
		dstr_printf(*out, "%s/%s", S->path_bin, name);
		if (file_exists(*out)) goto end;
	}
	
	dstr_printf(*out, "/usr/share/mlimgsynth/%s", name);
	if (file_exists(*out)) goto end;
	
	dstr_printf(*out, "/usr/local/share/mlimgsynth/%s", name);
	if (file_exists(*out)) goto end;
	
	//TODO: fs_dir_get, windows
	//TODO: environmental variable
	ERROR_LOG(MLIS_E_FILE_NOT_FOUND, "file '%s' could not be found", name);

end:
	return R;
}

int mlis_option_set(MLIS_Ctx* S, MLIS_Option id, ...)
{
	ERROR_HANDLE_BEGIN
	va_list ap;
	va_start(ap, id);

#define OPTION(NAME) \
	else if (id == MLIS_OPT_##NAME)

// Macros to get and check arguments
#define ARG_C(VAR, TYPE) \
	TYPE VAR = va_arg(ap, TYPE);
#define ARG_STR(VAR, MIN, MAX) \
	StrSlice VAR = strsl_fromz( va_arg(ap, const char*) ); \
	if (!(MIN <= VAR.s && VAR.s <= MAX)) goto error_value;
#define ARG_STR_NO_PARSE  ARG_STR
#define ARG_INT(VAR, MIN, MAX, DEF) \
	int VAR = va_arg(ap, int); \
	if (!(MIN <= VAR && VAR <= MAX)) goto error_value;
#define ARG_FLOAT(VAR, MIN, MAX, DEF) \
	float VAR = va_arg(ap, double); \
	if (!(MIN <= VAR && VAR <= MAX)) goto error_value;
#define ARG_BOOL(VAR) \
	bool VAR = !!va_arg(ap, int);
#define ARG_UINT64(VAR) \
	uint64_t VAR = va_arg(ap, uint64_t);
#define ARG_ENUM(VAR, FROMZ) \
	int VAR = va_arg(ap, int);
#define ARG_FLAGS(VAR) \
	int VAR = va_arg(ap, int);

	if (0) ;
#include "mlimgsynth_options_set.c.h"
	else
		ERROR_LOG(MLIS_E_UNK_OPT, "unknown option %u", id);

#undef ARG_FLAGS
#undef ARG_ENUM
#undef ARG_UINT64
#undef ARG_BOOL
#undef ARG_FLOAT
#undef ARG_INT
#undef ARG_STR_NO_PARSE
#undef ARG_STR
#undef ARG_C
	
end:
	va_end(ap);
	ERROR_HANDLE_END("mlis_option_set")

error_value:
	ERROR_LOG(MLIS_E_OPT_VALUE, "invalid argument for option '%s'",
		mlis_option_str(id));
}

static
StrSlice value_str_next(const char** pcur)
{
	StrSlice ss;
	const char *cur=*pcur;
	if (*cur == ',') cur++;
	if (*cur == '"') {  // Quoted argument
		cur++;
		ss.b = cur;
		while (*cur && *cur != '"') cur++;
		ss.s = cur - ss.b;
		if (*cur == '"') cur++;
	} else {
		ss.b = cur;
		while (*cur && *cur != ',') cur++;
		ss.s = cur - ss.b;
	}
	*pcur = cur;
	return ss;
}

static
int parse_bool(const StrSlice ss, int *out)
{
	if (!strsl_cmpz(ss, "true")) { *out=1; return 1; }
	if (!strsl_cmpz(ss, "false")) { *out=0; return 1; }
	if (!strsl_cmpz(ss, "1")) { *out=1; return 1; }
	if (!strsl_cmpz(ss, "0")) { *out=0; return 1; }
	return -1;
}

int mlis_option_set_str(MLIS_Ctx* S, const char* name, const char* value)
{
	ERROR_HANDLE_BEGIN
	StrSlice arg={0};
	const char *vcur = value ? value : "";
	char *tail;

	int id = mlis_option_fromz(name);
	if (id <= 0)
		ERROR_LOG(MLIS_E_UNK_OPT, "unknown option '%s'", name);

// Macros to get, parse and check arguments
//TODO: move the parse and error checking to functions
#define ARG_IS_STR 1
#define ARG_C(VAR, TYPE) \
	TYPE VAR; \
	ERROR_LOG(MLIS_E_OPT_VALUE, \
		"option '%s' cannot be set with a string value", \
		mlis_option_str(id));
#define ARG_STR(VAR, MIN, MAX) \
	StrSlice VAR = arg = value_str_next(&vcur); \
	if (!(MIN <= VAR.s && VAR.s <= MAX)) goto error_value;
#define ARG_STR_NO_PARSE(VAR, MIN, MAX) \
	StrSlice VAR = arg = strsl_fromz(vcur); \
	vcur = strsl_end(arg); \
	if (!(MIN <= VAR.s && VAR.s <= MAX)) goto error_value;
#define ARG_INT(VAR, MIN, MAX, DEF) \
	arg = value_str_next(&vcur); \
	tail = (char*) strsl_end(arg); \
	int VAR = arg.s ? strtol(arg.b, &tail, 10) : DEF; \
	if (tail != strsl_end(arg)) goto error_value; \
	if (!(MIN <= VAR && VAR <= MAX)) goto error_value;
#define ARG_FLOAT(VAR, MIN, MAX, DEF) \
	arg = value_str_next(&vcur); \
	tail = (char*) strsl_end(arg); \
	float VAR = arg.s ? strtof(arg.b, &tail) : DEF; \
	if (tail != strsl_end(arg)) goto error_value; \
	if (!(MIN <= VAR && VAR <= MAX)) goto error_value;
#define ARG_UINT64(VAR) \
	arg = value_str_next(&vcur); \
	uint64_t VAR = strtoll(arg.b, &tail, 10); \
	if (tail != strsl_end(arg)) goto error_value;
#define ARG_BOOL(VAR) \
	arg = value_str_next(&vcur); \
	int VAR; \
	if (parse_bool(arg, &VAR) < 0) goto error_value;
#define ARG_ENUM(VAR, FROMZ) \
	arg = value_str_next(&vcur); \
	int VAR = FROMZ(arg); \
	if (VAR < 0) goto error_value;
#define ARG_FLAGS(VAR) \
	arg = value_str_next(&vcur); \
	int VAR = strtol(arg.b, &tail, 10); \
	if (tail != strsl_end(arg)) goto error_value;
	
	if (0) ;
#include "mlimgsynth_options_set.c.h"
	else
		ERROR_LOG(-1, "invalid string option %u '%s'", id, name);

#undef ARG_FLAGS
#undef ARG_ENUM
#undef ARG_UINT64
#undef ARG_BOOL
#undef ARG_FLOAT
#undef ARG_INT
#undef ARG_STR_NO_PARSE
#undef ARG_STR
#undef ARG_C
#undef ARG_IS_STR

done:
end:
	ERROR_HANDLE_END("mlis_option_set_str")

error_value:
	ERROR_LOG(MLIS_E_OPT_VALUE, "invalid argument '%.*s' for option '%s'",
		(int)arg.s, arg.b, mlis_option_str(id));
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

static
int mlis_lora_load_apply(MLIS_Ctx* S, const char* path, float mult)
{
	int R=1;
	Stream stm={0};
	TensorStore ts={ .ss=&S->ss };

	log_debug("lora apply: '%s' %g", path, mult);

	TRY_LOG( stream_open_file(&stm, path, SOF_READ | SOF_MMAP),
		"could not open '%s'", path);

	TSCallback cb = { tensor_callback_lora };
	TRY( tstore_read(&ts, &stm, NULL, &cb) );
	
	if (S->c.dump_flags & MLIS_DUMP_LORA)
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

static
int mlis_backend_init(MLIS_Ctx* S)
{
	int R=1;
	
	if (S->ctx.backend) {
		ggml_backend_free(S->ctx.backend);
		S->ctx.backend = NULL;
	}

	if (!dstr_empty(S->c.backend))
		S->ctx.backend =
			ggml_backend_init_by_name(S->c.backend, S->c.be_params);
	else 
		S->ctx.backend = ggml_backend_init_best();
	
	if (!S->ctx.backend) ERROR_LOG(-1, "ggml backend init");
	log_info("Backend: %s", ggml_backend_name(S->ctx.backend));

	if (S->c.n_thread > 0)
		ggml__backend_set_n_threads(S->ctx.backend, S->c.n_thread);

#if USE_GGML_SCHED  //old code
	if (!S->ctx.backend2) {
		log_debug("Fallback backend CPU");
		S->ctx.backend2 = ggml_backend_cpu_init();
		if (!S->ctx.backend2) ERROR_LOG(-1, "ggml fallback backend CPU init");
		
		if (S->c.n_thread > 0)
			ggml_backend_set_n_threads_t(S->ctx.backend2, S->c.n_thread);
	}
#endif

end:
	return R;
}

static
int mlis_model_load(MLIS_Ctx* S)
{
	int R=1;
	
	if (!(S->c.path_model))  //TODO: allow to set the model by parts
		ERROR_LOG(MLIS_E_UNKNOWN, "No model file set");

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
		
	if (S->c.dump_flags & MLIS_DUMP_MODEL)
		TRY( tstore_info_dump_path(&S->tstore, "dump-tensors-model.txt") );

end:
	return R;
}

static
int mlis_model_identify(MLIS_Ctx* S)
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
		ERROR_LOG(-1, "unknown model type");
	
	log_info("Model type: %s", model_type);
	//TODO: save model_type and allow to retrieve it
	
end:
	return R;
}

int mlis_setup(MLIS_Ctx* S)
{
	if (S->signature != CTX_SIGNATURE) {
		log_error("mlis ctx invalid signature");
		return -1;
	}
	
	ERROR_HANDLE_BEGIN
		
	if (!(S->rflags & MLIS_READY_RNG)) {
		log_info("Seed: %" PRIu64, g_rng.seed);
		S->rflags |= MLIS_READY_RNG;
	}

	if (!(S->rflags & MLIS_READY_BACKEND)) {
		// Backend init
		TRY( mlis_backend_init(S) );
		S->rflags |= MLIS_READY_BACKEND;
	}

	if (!(S->rflags & MLIS_READY_MODEL)) {
		// Model parameters header load
		TRY( mlis_model_load(S) );

		// Identify model type
		TRY( mlis_model_identify(S) );
		
		S->rflags |= MLIS_READY_MODEL;
	}
	
	if (!(S->rflags & MLIS_READY_LORAS)) {
		// Clear cache'd tensors that could have previous loras applied
		tstore_cache_clear(&S->tstore);

		// Load loras
		if (vec_count(S->loras)) {
			double t = timing_time();
			vec_for(S->loras,i,0) {
				TRY( mlis_lora_load_apply(S, S->loras[i].path, S->loras[i].mult) );
			}
			t = timing_time() - t;
			log_info("LoRA's applied: %u {%.3fs}", vec_count(S->loras), t);
		}
		
		S->rflags |= MLIS_READY_LORAS;
	}

	S->ctx.c.dump = !!(S->c.dump_flags & MLIS_DUMP_GRAPH);

end:
	ERROR_HANDLE_END("mlis_setup")
}

int mlis_image_encode(MLIS_Ctx* S, const LocalTensor* image, LocalTensor* latent,
	int flags)
{
	ERROR_HANDLE_BEGIN

	TRY( mlis_setup(S) );
	
	if (S->c.flags & MLIS_CF_USE_TAE) {
		S->ctx.tprefix = "tae";
		TRY( sdtae_encode(&S->ctx, S->tae_p, image, latent) );
	} else {
		S->ctx.tprefix = "vae";
		TRY( sdvae_encode(&S->ctx, S->vae_p, image, latent, S->c.vae_tile) );
		
		// Sample if needed
		if (latent->n[2] == S->vae_p->d_embed*2)
			sdvae_latent_sample(latent, latent, S->vae_p);
	}

	if (ltensor_finite_check(latent) < 0 )
		ERROR_LOG(MLIS_E_NAN, "NaN found in encoded latent");
	
	//TODO: call for each tile?
	TRYR( mlis_callback(S, MLIS_STAGE_IMAGE_ENCODE, 1, 1) );

end:
	ERROR_HANDLE_END("mlis_image_encode")
}

int mlis_image_decode(MLIS_Ctx* S, const LocalTensor* latent, LocalTensor* image,
	int flags)
{
	ERROR_HANDLE_BEGIN

	TRY( mlis_setup(S) );
	
	if (S->c.flags & MLIS_CF_USE_TAE) {
		S->ctx.tprefix = "tae";
		TRY( sdtae_decode(&S->ctx, S->tae_p, latent, image) );
	} else {
		S->ctx.tprefix = "vae";
		TRY( sdvae_decode(&S->ctx, S->vae_p, latent, image, S->c.vae_tile) );
	}

	if (ltensor_finite_check(image) < 0 )
		ERROR_LOG(MLIS_E_NAN, "NaN found in encoded latent");
	
	image->flags |= LT_F_READY;
	
	//TODO: call for each tile?
	TRYR( mlis_callback(S, MLIS_STAGE_IMAGE_DECODE, 1, 1) );

end:
	ERROR_HANDLE_END("mlis_image_decode")
}

int mlis_mask_encode(MLIS_Ctx* S, const MLIS_Tensor* mask, MLIS_Tensor* lmask,
	int flags)
{
	int vae_f = S->vae_p->f_down;
	ltensor_downsize(lmask, mask, vae_f, vae_f, 1, 1);
	return 1;
}

int mlis_clip_text_encode(MLIS_Ctx* S, const char* prompt,
	LocalTensor* embed, LocalTensor* feat, int model_idx, int flags)
{
	ERROR_HANDLE_BEGIN
	DynStr tmps=NULL;
	int32_t *tokens=NULL;

	TRY( mlis_setup(S) );

	// Select model
	const ClipParams* clip_p=NULL;
	const char* tprefix=NULL;
	switch (model_idx) {
	case 0:
		clip_p = S->clip_p;
		tprefix = "clip";
		break;
	case 1:
		clip_p = S->clip2_p;
		tprefix = "clip2";
		break;
	}
	if (!clip_p)
		ERROR_LOG(MLIS_E_UNKNOWN, "invalid clip model no. %d", model_idx);

	// Load vocabulary
	if (!clip_good(&S->tokr)) {
		TRY( mlis_file_find(S, "clip-vocab-merges.txt", &tmps) );
		log_debug("Loading vocabulary from '%s'", tmps);
		TRY( clip_tokr_vocab_load(&S->tokr, tmps) );
	}	
	unsigned n_vocab = strsto_count(&S->tokr.vocab);
	if (n_vocab != clip_p->n_vocab)
		ERROR_LOG(-1, "wrong vocabulary size: %u (read) != %u (expected)",
			n_vocab, clip_p->n_vocab);

	// Tokenize the prompt
	IFFALSESET(prompt, "");
	TRY( clip_tokr_tokenize(&S->tokr, prompt, &tokens) );
	log_debug_vec("Tokens", tokens, i, 0, "%u %s",
		tokens[i], clip_tokr_word_from_token(&S->tokr, tokens[i]) );
	log_info("Prompt: %u tokens", vec_count(tokens));

	// Encode
	bool b_norm = !(flags & MLIS_CTEF_NO_NORM);
	S->ctx.tprefix = tprefix;
	TRY( clip_text_encode(&S->ctx, clip_p,
		tokens, embed, feat, S->c.clip_skip, b_norm) );

end:
	vec_free(tokens);
	dstr_free(tmps);
	ERROR_HANDLE_END("mlis_clip_text_encode")
}

// Fill sinusoidal timestep embedding (from CompVis)
static
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

int mlis_text_cond_encode(MLIS_Ctx* S, const char* prompt,
	LocalTensor* cond, LocalTensor* label, int flags)
{
	ERROR_HANDLE_BEGIN
	LocalTensor tmpt={0};

	int cte_flags = 0;
	if (!S->unet_p->clip_norm) cte_flags |= MLIS_CTEF_NO_NORM;

	TRY( mlis_clip_text_encode(S, prompt, cond, NULL, 0, cte_flags) );

	if (S->unet_p->cond_label) {
		TRY( mlis_clip_text_encode(S, prompt, &tmpt, NULL, 1, cte_flags) );

		// Concatenate both text embeddings
		assert( cond->n[1] == tmpt.n[1] &&
		        cond->n[2] == 1 && tmpt.n[2] == 1 &&
			    cond->n[3] == 1 && tmpt.n[3] == 1 );	

		unsigned n_tok = tmpt.n[1],
		         n_emb1 = cond->n[0],
				 n_emb2 = tmpt.n[0],
		         n_emb = n_emb1 + n_emb2;

		ltensor_resize(cond, n_emb, n_tok, 1, 1);
		for (unsigned i1=n_tok-1; (int)i1>=0; --i1) {
			ARRAY_COPY(cond->d+n_emb*i1+n_emb1, tmpt.d+n_emb2*i1, n_emb2);
			ARRAY_COPY(cond->d+n_emb*i1, cond->d+n_emb1*i1, n_emb1);
		}
		
		//TODO: no need to reprocess from scratch...
		TRY( mlis_clip_text_encode(S, prompt, NULL, label, 1, 0) );

		// Complete label embedding
		assert( label->n[0]==n_emb2 &&
			label->n[1]==1 && label->n[2]==1 && label->n[3]==1 );
		ltensor_resize(label, S->unet_p->ch_adm_in, 1, 1, 1);
		float *ld = label->d + n_emb2;
		unsigned w = S->c.width, h = S->c.height;
		// Original size
		ld += sd_timestep_embedding(2, (float[]){h,w}, 256, 10000, ld);
		// Crop top,left
		ld += sd_timestep_embedding(2, (float[]){0,0}, 256, 10000, ld);
		// Target size
		ld += sd_timestep_embedding(2, (float[]){h,w}, 256, 10000, ld);
		assert(ld == label->d + label->n[0]);
	}

end:
	ltensor_free(&tmpt);
	ERROR_HANDLE_END("mlis_text_cond_encode")
}

struct dxdt_args {
	MLIS_Ctx *S;
	UnetState *unet;
	LocalTensor *cond, *label, *uncond, *unlabel, *tmpt;
};

static
int mlis_denoise_dxdt(Solver* sol, float t, const LocalTensor* x,
	LocalTensor* dx)
{
	if (!(t >= 0)) return 0;
	struct dxdt_args *A = sol->user;
	
	TRYR( unet_denoise_run(A->unet, x, A->cond, A->label, t, dx) );
	
	float f = A->S->c.cfg_scale;
	if (f > 1) {
		TRYR( unet_denoise_run(A->unet, x, A->uncond, A->unlabel, t, A->tmpt) );
		ltensor_for(*dx,i,0) dx->d[i] = dx->d[i]*f + A->tmpt->d[i]*(1-f);
	}
	
	return 1;
}

/* Updates information text.
 * Usually saved along with generated images.
 */
static
void mlis_infotext_update(MLIS_Ctx* S, unsigned w, unsigned h)
{
	DynStr *out = &S->infotext;

	dstr_resize(*out, 0);

	// Imitates stable-diffusion-webui create_infotext
	dstr_printfa(*out, "%s\n", S->c.prompt);
	if (S->c.nprompt)
		dstr_printfa(*out, "Negative prompt: %s\n", S->c.nprompt);
	dstr_printfa(*out, "Seed: %"PRIu64, g_rng.seed);
	dstr_printfa(*out, ", Sampler: %s", mlis_method_str(S->sampler.c.method));
	dstr_printfa(*out, ", Schedule type: %s", mlis_sched_str(S->sampler.c.sched));
	if (S->sampler.c.s_ancestral > 0)
		dstr_printfa(*out, ", Ancestral: %g", S->sampler.c.s_ancestral);
	if (S->sampler.c.s_noise > 0)
		dstr_printfa(*out, ", SNoise: %g", S->sampler.c.s_noise);
	if (S->c.cfg_scale > 1)
		dstr_printfa(*out, ", CFG scale: %g", S->c.cfg_scale);
	if (S->sampler.c.f_t_ini < 1) {
		dstr_printfa(*out, ", Mode: %s, f_t_ini: %g",
			S->sampler.c.lmask ? "inpaint" : "img2img", S->sampler.c.f_t_ini);
	}
	dstr_printfa(*out, ", Steps: %u", S->sampler.n_step);
	dstr_printfa(*out, ", NFE: %u", S->prg.nfe);
	dstr_printfa(*out, ", Size: %ux%u", w, h);
	dstr_printfa(*out, ", Clip skip: %d", S->c.clip_skip);
	{
		const char *b = path_tail(S->c.path_model),
		           *e = path_ext(b);
		if (*e) e--;  // .
		dstr_appendz(*out, ", Model: ");
		dstr_append(*out, e-b, b);
	}
	if (S->c.flags & MLIS_CF_USE_TAE)
		dstr_printfa(*out, ", VAE: tae");
	dstr_printfa(*out, ", Version: MLImgSynth v%s", MLIS_VERSION_STR);
}

int mlis_generate(MLIS_Ctx* S)
{
	ERROR_HANDLE_BEGIN
	UnetState unet={0};
	LocalTensor tmpt={0};

	if (S->c.n_batch > 1)
		ERROR_LOG(MLIS_E_UNKNOWN, "Batch size > 1 not supported yet.");

	TRY( mlis_setup(S) );
	
	mlis_progress_reset(S);
	double t_start = S->prg.time;
	
	int vae_f = S->vae_p->f_down,
		w = S->c.width  / vae_f,
		h = S->c.height / vae_f;

	// Encode initial image (img2img)
	if (S->c.tuflags & MLIS_TUF_IMAGE)
	{
		TRY( mlis_image_encode(S, &S->image, &S->latent, 0) );
		S->c.tuflags |= MLIS_TUF_LATENT;
	}

	// Initial latent
	if (S->c.tuflags & MLIS_TUF_LATENT)
	{
		w = S->latent.n[0];
		h = S->latent.n[1];
		log_debug3_ltensor(&S->latent, "input latent");
	}
	else
	{
		log_debug("Empty initial latent");
		ltensor_resize(&S->latent, w, h, S->unet_p->n_ch_in, 1);
		memset(S->latent.d, 0, ltensor_nbytes(&S->latent));
	}
	int w_img = w * vae_f, h_img = h * vae_f;
	log_info("Output size: %ux%u", w_img, h_img);

	// Image mask -> latent mask
	if (S->c.tuflags & MLIS_TUF_MASK)
	{
		TRY( mlis_mask_encode(S, &S->mask, &S->lmask, 0) );
		S->c.tuflags |= MLIS_TUF_LMASK;
	}

	// Latent mask for inpainting
	if (S->c.tuflags & MLIS_TUF_LMASK)
	{
		log_debug3_ltensor(&S->lmask, "latent mask");
		log_info("In-painting with mask");
	}

	// Conditioning
	if (!(S->c.tuflags & MLIS_TUF_CONDITIONING))
	{
		// Text prompt
		TRY( mlis_text_cond_encode(S, S->c.prompt, &S->cond, &S->label, 0) );

		// Negative text prompt
		if (S->c.cfg_scale > 1)
		{
			TRY( mlis_text_cond_encode(S, S->c.nprompt, &S->ncond, &S->nlabel, 0) );

			//TODO: move to unet
			if (S->unet_p->uncond_empty_zero && !(S->c.nprompt && S->c.nprompt[0]))
				ltensor_for(S->ncond,i,0) S->ncond.d[i] = 0;
		}
		
		TRY( mlis_callback(S, MLIS_STAGE_COND_ENCODE, 1, 1) );
	}
	
	S->image.flags &= ~LT_F_READY;
	
	log_debug3_ltensor(&S->cond, "cond");
	log_debug3_ltensor(&S->label, "label");
	if (S->c.cfg_scale > 1) {
		log_debug3_ltensor(&S->ncond, "uncond");
		log_debug3_ltensor(&S->nlabel, "unlabel");
	}

	// Sampling initialization
	S->sampler.unet_p = S->unet_p;
	S->sampler.nfe_per_dxdt = (S->c.cfg_scale > 1) ? 2 : 1;
	S->sampler.c.lmask = ltensor_good(&S->lmask) ? &S->lmask : NULL;

	struct dxdt_args A = { .S=S, .unet=&unet, .tmpt=&tmpt,
		.cond=&S->cond, .uncond=&S->ncond,
		.label=&S->label, .unlabel=&S->nlabel };
	S->sampler.solver.dxdt = mlis_denoise_dxdt;
	S->sampler.solver.user = &A;
	
	TRY( dnsamp_init(&S->sampler) );
	
	// Prepare computation
	S->ctx.tprefix = "unet";
	TRY( unet_denoise_init(&unet, &S->ctx, S->unet_p, w, h,
		S->c.flags & MLIS_CF_UNET_SPLIT) );
	
	log_info("Generating "
		"(solver: %s, sched: %s, ancestral: %g, snoise: %g, cfg-s: %g, steps: %d"
		", nfe/s: %d)",
		mlis_method_str(S->sampler.c.method), mlis_sched_str(S->sampler.c.sched),
		S->sampler.c.s_ancestral, S->sampler.c.s_noise, S->c.cfg_scale,
		S->sampler.n_step, S->sampler.nfe_per_step);


	// Denoising / generation / sampling
	int r;
	while ((r = dnsamp_step(&S->sampler, &S->latent)) > 0) {
		S->prg.nfe = unet.nfe;
		TRY( mlis_callback(S, MLIS_STAGE_DENOISE, S->sampler.i_step,
			S->sampler.n_step) );
	}
	TRY(r);
	
	// Free memory
	mlctx_free(&S->ctx);  

	// Decode
	if (!(S->c.flags & MLIS_CF_NO_DECODE))
	{
		TRY( mlis_image_decode(S, &S->latent, &S->image, 0) );
	}

	//
	mlis_infotext_update(S, w_img, h_img);

	//
	mlis_prompt_clear(S);

	log_info("Generation done {%.3fs}", timing_time() - t_start);

end:
	ltensor_free(&tmpt);
	mlctx_free(&S->ctx);
	ERROR_HANDLE_END("mlis_generate")
}

MLIS_Image* mlis_image_get(MLIS_Ctx* S, int idx)
{
	if (idx != 0) {
		dstr_copyz(S->errstr, "only image idx=0 supported");
		mlis_error_handle(S, MLIS_E_UNKNOWN);
		return NULL;
	}

	if (!(S->image.flags & LT_F_READY)) {
		dstr_copyz(S->errstr, "image not ready");
		mlis_error_handle(S, MLIS_E_UNKNOWN);
		return NULL;
	}

	mlis_tensor_to_image(&S->image, &S->imgex, idx);
	
	return &S->imgex;
}

const char* mlis_infotext_get(MLIS_Ctx* S, int idx)
{
	if (idx != 0) {
		dstr_copyz(S->errstr, "only image idx=0 supported");
		mlis_error_handle(S, MLIS_E_UNKNOWN);
		return NULL;
	}

	return S->infotext;
}
