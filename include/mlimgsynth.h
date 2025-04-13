/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * MLImgSynth library header.
 * Synthetize images using pre-trained machine learning models.
 * Currently supports Stable Diffusion (SD) 1.x, 2.x, and XL.
 * Destilled variants (Lightning, Hyper, Turbo) also work.
 * Uses GGML library for performant execution in CPU and GPU. 
 * Currently, this library is not thread safe. Use it from one thread at a time.
 */
#ifndef MLIMGSYNTH_H
#define MLIMGSYNTH_H

/* Example:
int callback(void* user, MLIS_Ctx* ctx, const MLIS_Progress* pgr) {
	log("%s %d/%d %.3fs",
		mlis_state_str(pgr->state), pgr->step, pgr->step_end, pgr->step_time);
	return 0;
}

void img_save_pnm(const char* path, unsigned width, unsigned height,
	const void* data)
{
	FILE *f = fopen(path, "w");
	fprintf(f, "P6 %u %u 255\n", width, height);
	fwrite(data, 1, width*height*3, f);
	fclose(f);
}

void gen() {
	MLIS_Ctx *mlis = mlis_ctx_create();
	
	mlis_option_set(mlis, MLIS_OPT_MODEL,
		"sd_v1.5-pruned-emaonly-fp16.safetensors");
	
	mlis_option_set(mlis, MLIS_OPT_PROMPT,
		"a photograph of an astronaut riding a horse");
	
	mlis_option_set(mlis, MLIS_OPT_CALLBACK, callback, NULL);

	if (mlis_generate(mlis) < 0)
		error("mlis: error generating: %s", mlis_errstr_get(mlis));
	
	const MLIS_Image *img = mlis_image_get(mlis, 0);
	img_save_pnm("output.pnm", img->w, img->h, img->d);

	log("Infotext:\n%s", mlis_infotext_get(mlis, 0));

	mlis_ctx_destroy(&mlis);
}
*/

#include <stddef.h>
#include <stdint.h>

//TODO: support other compilers? clang, msvc
#if defined(MLIS_IMPLEMENTATION) && defined(__GNUC__)
#pragma GCC visibility push(default)
#endif

#define MLIS_VERSION  0x000401
#define MLIS_VERSION_STR  "0.4.1"

/* Enumerations */

/* Error codes.
 */
typedef enum MLIS_ErrorCode {
	MLIS_E_UNKNOWN			= -1,
	MLIS_E_VERSION			= -2,  //no used
	MLIS_E_UNK_OPT			= -3,
	MLIS_E_OPT_VALUE		= -4,
	MLIS_E_PROMPT_PARSE		= -5,
	MLIS_E_FILE_NOT_FOUND	= -6,
	MLIS_E_NAN				= -7,
	MLIS_E_IMAGE			= -8,
} MLIS_ErrorCode;

/* Context states.
 */
typedef enum MLIS_Stage {
	MLIS_STAGE_IDLE			= 0,
	MLIS_STAGE_COND_ENCODE	= 1,
	MLIS_STAGE_IMAGE_ENCODE	= 2,
	MLIS_STAGE_IMAGE_DECODE	= 3,
	MLIS_STAGE_DENOISE		= 4,
} MLIS_Stage;

/* Methods to solve the diffusion differential equation.
 */
typedef enum MLIS_Method {
	MLIS_METHOD_NONE		= 0,
	MLIS_METHOD_EULER		= 1,
	MLIS_METHOD_HEUN		= 2,
	MLIS_METHOD_TAYLOR3		= 3,
	MLIS_METHOD_DPMPP2M		= 4,
	MLIS_METHOD_DPMPP2S		= 5,
	MLIS_METHOD__LAST		= 5,
} MLIS_Method;

/* Schedulers (use to generate the sequence of time steps).
 */
typedef enum MLIS_Scheduler {
	MLIS_SCHED_NONE			= 0,
	MLIS_SCHED_UNIFORM		= 1,
	MLIS_SCHED_KARRAS		= 2,
	MLIS_SCHED__LAST		= 2,
} MLIS_Scheduler;

/* Logging levels.
 */
typedef enum MLIS_LogLvl {
	MLIS_LOGLVL_NONE	= 0,
	MLIS_LOGLVL_ERROR	= 10,
	MLIS_LOGLVL_WARNING	= 20,
	MLIS_LOGLVL_INFO	= 30,
	MLIS_LOGLVL_VERBOSE	= 40,
	MLIS_LOGLVL_DEBUG	= 50,
	MLIS_LOGLVL_MAX		= 255,
	MLIS_LOGLVL__INCREASE = 0x100 | 10,  // Relative increase with option LOG_LVL
	MLIS_LOGLVL__DECREASE = 0x200 | 10,  // Relative decrease with option LOG_LVL
} MLIS_LogLvl;

/* Internal tensors available.
 */
typedef enum MLIS_TensorId {
	MLIS_TENSOR_IMAGE		= 1,
	MLIS_TENSOR_MASK		= 2,
	MLIS_TENSOR_LATENT		= 3,  // Latent-space image (encoded image)
	MLIS_TENSOR_LMASK		= 4,  // Latent mask
	MLIS_TENSOR_COND		= 5,  // Conditioning
	MLIS_TENSOR_LABEL		= 6,  // Label (SDXL)
	MLIS_TENSOR_NCOND		= 7,  // Negative conditioning
	MLIS_TENSOR_NLABEL		= 8,
	// Tensors starting from the following index are reserved for user use.
	// You don't need to free.
	MLIS_TENSOR_TMP			= 0x100,  
} MLIS_TensorId;

/* Tensor use flags.
 */
typedef enum MLIS_TensorUseFlag {
	MLIS_TUF_IMAGE			= 1,
	MLIS_TUF_MASK			= 2,
	MLIS_TUF_LATENT			= 4,
	MLIS_TUF_LMASK			= 8,
	MLIS_TUF_CONDITIONING	= 16,  //all conditioning (cond, label, ncond, ...)
} MLIS_TensorUseFlag;

/* Image synthesis model types (e.g. Stable Diffusion).
 */
typedef enum MLIS_ModelType {
	MLIS_MODEL_TYPE_NONE	= 0,
	MLIS_MODEL_TYPE_SD1		= 1,
	MLIS_MODEL_TYPE_SD2		= 2,
	MLIS_MODEL_TYPE_SDXL	= 3,
	MLIS_MODEL_TYPE__LAST	= 3,
} MLIS_ModelType;

/* Individual sub-models used in each stage internally.
 */
typedef enum MLIS_SubModel {
	MLIS_SUBMODEL_NONE			= 0,
	MLIS_SUBMODEL_UNET			= 1,
	MLIS_SUBMODEL_VAE			= 2,
	MLIS_SUBMODEL_TAE			= 3,
	MLIS_SUBMODEL_CLIP			= 4,
	MLIS_SUBMODEL_CLIP2		= 5,
} MLIS_SubModel;

/* Options.
 * Set with mlis_option_set.
 */
typedef enum MLIS_Option {
	MLIS_OPT_NONE = 0,

	// GGML backend.
	// Arg: name (str)
	// Arg: params (str, null)
	MLIS_OPT_BACKEND = 1,

	// Model path.
	// Arg: path (str)
	MLIS_OPT_MODEL = 2,

	// TAE model path and enable its use.
	// Arg: path (str)
	MLIS_OPT_TAE = 3,

	// Lora's models directory.
	// Arg: path (str)
	MLIS_OPT_LORA_DIR = 4,

	// Add a lora.
	// Arg: name or path (str)
	// Arg: multiplier (float)
	MLIS_OPT_LORA = 5,

	// Remove all loras.
	MLIS_OPT_LORA_CLEAR = 6,

	// Prompt, text conditioning.
	// Can also configure LoRA's putting anywhere "<lora:NAME:MULT>".
	// This option is cleared after generation.
	// Arg: text (str)
	MLIS_OPT_PROMPT = 7,

	// Negative prompt. Used only if cfg_scale > 1.
	// This option is cleared after generation.
	// Arg: text (str)
	MLIS_OPT_NPROMPT = 8,

	// Image dimensions.
	// Arg: width (int)
	// Arg: height (int)
	MLIS_OPT_IMAGE_DIM = 9,

	// Batch size (number of images to generate simultaneously).
	// Increases memory usage, but should be faster to generating individually.
	// Not implemented yet.
	// Arg: (int)
	MLIS_OPT_BATCH_SIZE = 10,

	// Clip skip (> 0).
	// Arg: (int)
	MLIS_OPT_CLIP_SKIP = 11,

	// CFG (context-free guidanse) scale. Disabled if < 1.
	// Arg: (double)
	MLIS_OPT_CFG_SCALE = 12,

	// Method used during generation (differential equations solver).
	// Arg: (MLIS_Method)
	MLIS_OPT_METHOD = 13,

	// Scheduler used to determine the time steps for generation.
	// Arg: (MLIS_Scheduler)
	MLIS_OPT_SCHEDULER = 14,

	// Number of steps (iterations) to use for generation.
	// Arg: (int)
	MLIS_OPT_STEPS = 15,

	// Relative initial time (0-1). 1 for txt2img, < 1 for img2img.
	// This option is cleared after generation.
	// Arg: (double)
	MLIS_OPT_F_T_INI = 16,

	// Relative final time (0-1). Leave it at 0 unless you know what you are doing.
	// This option is cleared after generation.
	// Arg: (double)
	MLIS_OPT_F_T_END = 17,

	// Amount of noise to add after each step.
	// Arg: (double)
	MLIS_OPT_S_NOISE = 18,

	// Ancestral methods noise level. Usually 0 or 1.
	// Arg: (double)
	MLIS_OPT_S_ANCESTRAL = 19,

	// Set the initial image for img2img.
	// If it has an alpha channel, it will be used as the inpainting mask.
	// Arg: (const MLIS_Image*)
	MLIS_OPT_IMAGE = 20,

	// Set the image mask for inpainting.
	// Only the first channel will be used.
	// Arg: (const MLIS_Image*)
	MLIS_OPT_IMAGE_MASK = 21,

	// Do not decode the generated image.
	// You can access the encoded image with mlis_tensor_get.
	// Arg: true or false (int)
	MLIS_OPT_NO_DECODE = 22,

	// These flags indicate which internal tensors should be kept for the next
	// generation. Otherwise, the tensor are ignored.
	// You can access and modify the tensors with mlis_tensor_get.
	// One use of this is to do img2img in the latent space.
	// This option is cleared after generation.
	// Arg: MLIS_TUF_* (int)
	MLIS_OPT_TENSOR_USE_FLAGS = 23,

	// RNG (random number generator) seed.
	// Arg: (uint64_t)
	MLIS_OPT_SEED = 24,

	// VAE encode/decode tile size. Reduces backend memory usage. Try 512.
	// Arg: (int)
	MLIS_OPT_VAE_TILE = 25,

	// Split unet model in two parts to reduce the memory usage.
	// Arg: true or false (int)
	MLIS_OPT_UNET_SPLIT = 26,

	// Number CPU threads used by the backend.
	// Arg: (int)
	MLIS_OPT_THREADS = 27,

	// (debug) Control dumping of tensors and graphs information to files.
	// Arg: (int)
	MLIS_OPT_DUMP_FLAGS = 28,
	
	// Directory path to search for auxiliary files (e.g. CLIP vocabulary).
	// Arg: path (str)
	MLIS_OPT_AUX_DIR = 29,

	// Progress callback.
	// This functions is called regularly during generation to allow the library
	// user to show the progress and to cancelate the process.
	// Arg: MLIS_Callback
	// Arg: user_data (void*)
	MLIS_OPT_CALLBACK = 30,

	// Error handler.
	// The supplied function will be called in case of error.
	// This may be called from any of the functions that take MLIS_Ctx as an
	// argument, including mlis_option_set.
	// Arg: MLIS_ErrorHandler
	// Arg: user_data (void*)
	MLIS_OPT_ERROR_HANDLER = 31,

	// Logging level.
	// Internal messages with less or equal level will be printed.
	// Currently prints to stderr, in the future a callback may be added.
	// Arg: (MLIS_LogLvl)
	MLIS_OPT_LOG_LEVEL = 32,

	// Model type.
	// You may use this options to force a model type without detection, or to
	// retrieve the detected type after mlis_setup is called.
	// Arg: (MLIS_ModelType)
	MLIS_OPT_MODEL_TYPE = 33,

	// Weight data type. Uses GGML types (0: f32, 1: f16, 8: q8_0).
	// With mlis_option_set_str, names can be used.
	// Arg: ggml_type (int)
	MLIS_OPT_WEIGHT_TYPE = 34,

	// Do not parse the prompt for attention emphasis and loras.
	// Arg: true or false (int)
	MLIS_OPT_NO_PROMPT_PARSE = 35,
	
	MLIS_OPT__LAST = 35,
} MLIS_Option;

/* Structures */

/* Opaque context type used during the image generation.
 */
typedef struct MLIS_Ctx MLIS_Ctx;

/* Image.
 */
typedef struct MLIS_Image {
	uint8_t *d;  // Data, one byte per channel per pixel
	size_t sz;   // Data size in bytes = w*h*c
	unsigned w,  // Width
	         h,  // Height
			 c;  // Channels: 1 for mask, 3 for RGB, 4 for RGBA
	int flags;
} MLIS_Image;

/* Progress information.
 */
typedef struct MLIS_Progress {
	MLIS_Stage stage;
	int step,		// Last finished step of the current stage.
	    step_end,	// Last step. If step == step_end, then it is done.
		nfe;		// Neural function evaluations, number of calls to unet
	double step_time;	// Time in seconds since the last step.
	double time;  // Current time in seconds. Unknown reference.
} MLIS_Progress;

/* Error information.
 */
typedef struct MLIS_ErrorInfo {
	MLIS_ErrorCode code;
	const char *desc;
} MLIS_ErrorInfo;

/* Backend information.
 */
typedef struct MLIS_BackendInfo {
	const char *name;  // Backend name to be used with the BACKEND option.
	unsigned n_dev;    // Number of devices (e.g. GPU's)
	struct MLIS_BackendDeviceInfo {
		const char *name,  // Short name
		           *desc;  // Long description
		size_t mem_free,   // Available memory in bytes
		       mem_total;  // Total device memory in bytes
	} *devs;
} MLIS_BackendInfo;

/* Minimal tensor type used to pass tensors backs and forth.
 */
#ifndef MLIS_IMPLEMENTATION
typedef struct MLIS_Tensor {
	float	*d;		// Pointer to data, contiguous.
	int		n[4];	// Shape. From inner to outer, like in GGML.
	int		flags;
} MLIS_Tensor;
#endif

/* Progress callback
 */
typedef int (*MLIS_Callback)(void*, MLIS_Ctx*, const MLIS_Progress*);

/* Error handler (callback)
 */
typedef void (*MLIS_ErrorHandler)(void*, MLIS_Ctx*, const MLIS_ErrorInfo*);

/* Functions */

/* Create a new context.
 */
#define mlis_ctx_create()  mlis_ctx_create_i(MLIS_VERSION)
MLIS_Ctx* mlis_ctx_create_i(int version);

/* Destroy a context, freeing all associated resources.
 */
void mlis_ctx_destroy(MLIS_Ctx** pctx);

/* Get textual description of the last error.
 */
const char* mlis_errstr_get(const MLIS_Ctx* ctx);

/* Set an option.
 * The number of arguments and their types depend on the option.
 * Returns 1 on success, 0 if ignored, and < 0 on error.
 */
int mlis_option_set(MLIS_Ctx* ctx, MLIS_Option id, ...);

/* Set an option with string.
 * Useful to allow configuration from the end user.
 * Lower and upper case names are accepted.
 * Hyphens instead of underscores are accepted.
 * Example: MLIS_OPT_CFG_SCALE = "CFG_SCALE" = "cfg_scale" = "cfg-scale".
 * In <value>, separate multiple arguments with ';'.
 * Returns 1 on success, 0 if ignored, and < 0 on error.
 */
int mlis_option_set_str(MLIS_Ctx* ctx, const char* name, const char* value);

/* Get the value(s) of an option.
 * Same number and type of arguments as for mlis_option_set, but here all arguments
 * must be pointers to suitable variables.
 * Returns 1 on success, and < 0 on error.
 */
int mlis_option_get(MLIS_Ctx* ctx, MLIS_Option id, ...);

/* Generate an image.
 * If a callback is set, it will be called during the process to report the
 * progress, and you may abort the generation returning a negative value.
 * This negative value will be returned by this function.
 */
int mlis_generate(MLIS_Ctx* ctx);

/* Access the resulting image.
 * idx: indicates the image position (usually zero).
 */
MLIS_Image* mlis_image_get(MLIS_Ctx* ctx, int idx);

/* Get textual description of the parameters used for the last generation.
 * Imitates stable-diffusion-webui create_infotext.
 * idx: indicates the image position (usually zero).
 */
const char* mlis_infotext_get(MLIS_Ctx* ctx, int idx);

/* Set up the backend and load the model's header.
 * This is not required since it is automatically done before generation (if
 * needed), but it can be useful for error reporting.
 */
int mlis_setup(MLIS_Ctx* ctx);

/* Access an internal tensor for reading or writing.
 * For advanced uses only.
 */
MLIS_Tensor* mlis_tensor_get(MLIS_Ctx* ctx, MLIS_TensorId id);

/* Get information about a backend.
 * To list all available backends, iterate starting with idx=0 until it
 * returns NULL.
 */
const MLIS_BackendInfo* mlis_backend_info_get(MLIS_Ctx* ctx, unsigned idx,
	int flags);

/* String-Id conversion functions. */

const char * mlis_stage_str(MLIS_Stage id);
const char * mlis_stage_desc(MLIS_Stage id);  // Pretty description
MLIS_Stage mlis_stage_fromz(const char* str);

const char * mlis_method_str(MLIS_Method id);
MLIS_Method mlis_method_fromz(const char* str);

const char * mlis_sched_str(MLIS_Scheduler id);
MLIS_Scheduler mlis_sched_fromz(const char* str);

const char* mlis_loglvl_str(MLIS_LogLvl id);
MLIS_LogLvl mlis_loglvl_fromz(const char* str);

const char* mlis_model_type_str(MLIS_ModelType id);
const char* mlis_model_type_desc(MLIS_ModelType id);
MLIS_ModelType mlis_model_type_fromz(const char* str);

const char* mlis_option_str(MLIS_Option id);
MLIS_Option mlis_option_fromz(const char* str);

/* The following functions allow you to do some internal operations manually.
 * It is not needed normally.
 * These interfaces may change in the future.
 */

int mlis_image_encode(MLIS_Ctx* ctx, const MLIS_Tensor* image, MLIS_Tensor* latent,
	int flags);

int mlis_image_decode(MLIS_Ctx* ctx, const MLIS_Tensor* latent, MLIS_Tensor* image,
	int flags);

int mlis_mask_encode(MLIS_Ctx* ctx, const MLIS_Tensor* mask, MLIS_Tensor* lmask,
	int flags);

/* Tokenize <text> using <model> (usually MLIS_MODEL_CLIP).
 * Return the number of tokens or a negative value on error.
 * ptokens: will be set to point to an array with the tokens id's.
 */
int mlis_text_tokenize(MLIS_Ctx* ctx, const char* text, int32_t** ptokens,
	MLIS_SubModel model);

/* Encode a text using a CLIP model.
 * Returns the embeddings and feature tensors (optional).
 * Requires a CLIP model with the text projection tensor.
 * model_idx: model index starting from 0 (SD1 has 1, SDXL has 2).
 * The cosine similarity between two feature tensors can be used to estimate the
 * similarity between to inputs.
 */
int mlis_clip_text_encode(MLIS_Ctx* ctx, const char* text,
	MLIS_Tensor* embed, MLIS_Tensor* feat, MLIS_SubModel model, int flags);

enum {  // Flags for mlis_clip_text_encode
	MLIS_CTEF_NO_NORM = 1,
};

//int mlis_text_cond_encode(MLIS_Ctx* ctx, const char* text,
//	MLIS_Tensor* cond, MLIS_Tensor* label, int flags);

/* Tensor operations.
 * For advanced uses only.
 */

void mlis_tensor_free(MLIS_Tensor*);
size_t mlis_tensor_count(const MLIS_Tensor*);
void mlis_tensor_resize(MLIS_Tensor*, int n0, int n1, int n2, int n3);
void mlis_tensor_resize_like(MLIS_Tensor*, const MLIS_Tensor*);
void mlis_tensor_copy(MLIS_Tensor*, const MLIS_Tensor*);
// Cosine similarity
float mlis_tensor_similarity(const MLIS_Tensor*, const MLIS_Tensor*);

/* Iteration over the four dimensions of a tensor T.
 * Example:
MLIS_Tensor ten={0};
mlis_tensor_resize(&ten, 32, 32, 32, 32);
mlis_tensor_for(ten, i) {
	ten.d[ip] = i0*i0 + i1*i1 + i2*i2 + i3*i3;
}
 */
#define mlis_tensor_for(T, L) \
	for (int L##p=0, L##0=0, L##1=0, L##2=0, L##3=0, \
		L##0n=(T).n[0], L##1n=(T).n[1], L##2n=(T).n[2], L##3n=(T).n[3]; \
		(L##0 < L##0n) || \
		(L##0=0, L##1++, L##1 < L##1n) || \
		(L##1=0, L##2++, L##2 < L##2n) || \
		(L##2=0, L##3++, L##3 < L##3n) ; \
		++L##0, ++L##p)

#if defined(MLIS_IMPLEMENTATION) && defined(__GNUC__)
#pragma GCC visibility pop
#endif

#endif /* MLIMGSYNTH_H */
