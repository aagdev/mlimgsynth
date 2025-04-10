"""
Copyright 2024, Alejandro A. García <aag@zorzal.net>
SPDX-License-Identifier: MIT

Python wrapper for the MLImgSynth library.
"""
import os, sys, ctypes

## Constants

MLIS_VERSION = 0x000400
MLIS_VERSION_STR = "0.4.0"

MLIS_E_UNKNOWN			= -1
MLIS_E_VERSION			= -2
MLIS_E_UNK_OPT			= -3
MLIS_E_OPT_VALUE		= -4
MLIS_E_PROMPT_OPT		= -5
MLIS_E_FILE_NOT_FOUND	= -6
MLIS_E_NAN				= -7
MLIS_E_IMAGE			= -8

MLIS_STAGE_IDLE			= 0
MLIS_STAGE_COND_ENCODE	= 1
MLIS_STAGE_IMAGE_ENCODE	= 2
MLIS_STAGE_IMAGE_DECODE	= 3
MLIS_STAGE_DENOISE		= 4

MLIS_METHOD_NONE		= 0
MLIS_METHOD_EULER		= 1
MLIS_METHOD_HEUN		= 2
MLIS_METHOD_TAYLOR3		= 3
MLIS_METHOD_DPMPP2M		= 4
MLIS_METHOD_DPMPP2S		= 5
MLIS_METHOD__LAST		= 5

MLIS_SCHED_NONE			= 0
MLIS_SCHED_UNIFORM		= 1
MLIS_SCHED_KARRAS		= 2
MLIS_SCHED__LAST		= 2

MLIS_LOGLVL_NONE	= 0
MLIS_LOGLVL_ERROR	= 10
MLIS_LOGLVL_WARNING	= 20
MLIS_LOGLVL_INFO	= 30
MLIS_LOGLVL_VERBOSE	= 40
MLIS_LOGLVL_DEBUG	= 50
MLIS_LOGLVL_MAX		= 255
MLIS_LOGLVL__INCREASE = 0x100 | 10
MLIS_LOGLVL__DECREASE = 0x200 | 10

MLIS_TENSOR_IMAGE		= 1
MLIS_TENSOR_MASK		= 2
MLIS_TENSOR_LATENT		= 3
MLIS_TENSOR_LMASK		= 4
MLIS_TENSOR_COND		= 5
MLIS_TENSOR_LABEL		= 6
MLIS_TENSOR_NCOND		= 7
MLIS_TENSOR_NLABEL		= 8
MLIS_TENSOR_TMP			= 0x100

MLIS_TUF_IMAGE			= 1
MLIS_TUF_MASK			= 2
MLIS_TUF_LATENT			= 4
MLIS_TUF_LMASK			= 8
MLIS_TUF_CONDITIONING	= 16

MLIS_MODEL_TYPE_NONE	= 0
MLIS_MODEL_TYPE_SD1		= 1
MLIS_MODEL_TYPE_SD2		= 2
MLIS_MODEL_TYPE_SDXL	= 3
MLIS_MODEL_TYPE__LAST	= 3
	
MLIS_MODEL_NONE			= 0
MLIS_MODEL_UNET			= 1
MLIS_MODEL_VAE			= 2
MLIS_MODEL_TAE			= 3
MLIS_MODEL_CLIP			= 4
MLIS_MODEL_CLIP2		= 5

MLIS_OPT_NONE = 0
MLIS_OPT_BACKEND = 1
MLIS_OPT_MODEL = 2
MLIS_OPT_TAE = 3
MLIS_OPT_LORA_DIR = 4
MLIS_OPT_LORA = 5
MLIS_OPT_LORA_CLEAR = 6
MLIS_OPT_PROMPT = 7
MLIS_OPT_NPROMPT = 8
MLIS_OPT_IMAGE_DIM = 9
MLIS_OPT_BATCH_SIZE = 10
MLIS_OPT_CLIP_SKIP = 11
MLIS_OPT_CFG_SCALE = 12
MLIS_OPT_METHOD = 13
MLIS_OPT_SCHEDULER = 14
MLIS_OPT_STEPS = 15
MLIS_OPT_F_T_INI = 16
MLIS_OPT_F_T_END = 17
MLIS_OPT_S_NOISE = 18
MLIS_OPT_S_ANCESTRAL = 19
MLIS_OPT_IMAGE = 20
MLIS_OPT_IMAGE_MASK = 21
MLIS_OPT_NO_DECODE = 22
MLIS_OPT_TENSOR_USE_FLAGS = 23
MLIS_OPT_SEED = 24
MLIS_OPT_VAE_TILE = 25
MLIS_OPT_UNET_SPLIT = 26
MLIS_OPT_THREADS = 27
MLIS_OPT_DUMP_FLAGS = 28
MLIS_OPT_AUX_DIR = 29
MLIS_OPT_CALLBACK = 30
MLIS_OPT_ERROR_HANDLER = 31
MLIS_OPT_LOG_LEVEL = 32
MLIS_OPT_MODEL_TYPE = 33
MLIS_OPT_WEIGHT_TYPE = 34
MLIS_OPT__LAST = 34

MLIS_CTEF_NO_NORM = 1

## Structures

class MLIS_Image_C(ctypes.Structure):
	_fields_ = [
		("d", ctypes.POINTER(ctypes.c_uint8)),  # Pointer to RGB data
		("sz", ctypes.c_size_t),  # Data size in bytes (normally w*h*c)
		("w", ctypes.c_int),  # Width
		("h", ctypes.c_int),  # Height
		("c", ctypes.c_int),  # Number of channels (normally 3)
		("flags", ctypes.c_int),  # Internal use, do not change.
	]
#end

class MLIS_Image:
	def __init__(self, cimg):
		self.data = ctypes.string_at(cimg.d, cimg.sz)  #bytes
		self.w = int(cimg.w)
		self.h = int(cimg.h)
		self.c = int(cimg.c)
#end

class MLIS_Tensor_C(ctypes.Structure):
	_fields_ = [
		("d", ctypes.POINTER(ctypes.c_float)),  # Pointer to data
		("n", ctypes.c_int * 4),  # Shape
		("flags", ctypes.c_int),  # Internal use, do not change.
	]
#end

class MLIS_Tensor:
	def __init__(self, cten):
		sz = cten.n[0] * cten.n[1] * cten.n[2] * cten.n[3] * 4
		self.data = ctypes.string_at(cten.d, sz)  #bytes
		self.n = tuple(cten.n)

	def similarity(self, other):
		d1 = ctypes.cast(self.data, ctypes.POINTER(ctypes.c_float))
		d2 = ctypes.cast(other.data, ctypes.POINTER(ctypes.c_float))
		t1 = MLIS_Tensor_C(d1, self.n, 0)
		t2 = MLIS_Tensor_C(d2, other.n, 0)
		s = mlis_lib.mlis_tensor_similarity(ctypes.byref(t1), ctypes.byref(t2))
		return s
#end

## Find the library
mlis_lib_path = os.getenv("MLIS_LIB_PATH", None)
if not mlis_lib_path:
	if sys.platform.startswith("win"):
		mlis_lib_name = "libmlimgsynth.dll"
	elif sys.platform.startswith("darwin"):
		mlis_lib_name = "libmlimgsynth.dylib"
	else:
		mlis_lib_name = "libmlimgsynth.so"
	for base in (".", "..", "lib", "../lib", ""):
		mlis_lib_path = os.path.join(base, mlis_lib_name)
		if os.path.exists(mlis_lib_path):
			break;

## Load the library
mlis_lib = ctypes.CDLL(mlis_lib_path)

mlis_lib.mlis_ctx_create_i.restype = ctypes.c_void_p
mlis_lib.mlis_ctx_create_i.argtypes = [ctypes.c_int]
mlis_lib.mlis_ctx_destroy.restype = None
mlis_lib.mlis_ctx_destroy.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
mlis_lib.mlis_errstr_get.restype = ctypes.c_char_p
mlis_lib.mlis_errstr_get.argtypes = [ctypes.c_void_p]
mlis_lib.mlis_option_set.restype = ctypes.c_int
mlis_lib.mlis_option_set.argtypes = [ctypes.c_void_p, ctypes.c_int]
mlis_lib.mlis_option_set_str.restype = ctypes.c_int
mlis_lib.mlis_option_set_str.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
mlis_lib.mlis_generate.restype = ctypes.c_int
mlis_lib.mlis_generate.argtypes = [ctypes.c_void_p]
mlis_lib.mlis_image_get.restype = ctypes.POINTER(MLIS_Image_C)
mlis_lib.mlis_image_get.argtypes = [ctypes.c_void_p, ctypes.c_int]
mlis_lib.mlis_infotext_get.restype = ctypes.c_char_p
mlis_lib.mlis_infotext_get.argtypes = [ctypes.c_void_p, ctypes.c_int]
mlis_lib.mlis_setup.restype = ctypes.c_int
mlis_lib.mlis_setup.argtypes = [ctypes.c_void_p]
mlis_lib.mlis_tensor_get.restype = ctypes.POINTER(MLIS_Tensor_C)
mlis_lib.mlis_tensor_get.argtypes = [ctypes.c_void_p, ctypes.c_int]
mlis_lib.mlis_clip_text_encode.restype = ctypes.c_int
mlis_lib.mlis_clip_text_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
	ctypes.POINTER(MLIS_Tensor_C), ctypes.POINTER(MLIS_Tensor_C),
	ctypes.c_int, ctypes.c_int]
mlis_lib.mlis_tensor_similarity.restype = ctypes.c_float
mlis_lib.mlis_tensor_similarity.argtypes = [
	ctypes.POINTER(MLIS_Tensor_C), ctypes.POINTER(MLIS_Tensor_C)]

#TODO: all functions

## Python interface
class MLImgSynth:
	def __init__(self):
		self._ctx = mlis_lib.mlis_ctx_create_i(MLIS_VERSION)
		if not self._ctx:
			raise RuntimeError("Failed to create MLIS context")
	#end

	def __del__(self):
		mlis_lib.mlis_ctx_destroy(ctypes.byref(ctypes.c_void_p(self._ctx)))
	#end

	def option_set(self, option, *args):
		if type(option) == str:
			s_opt = option.encode("utf8")
			s_args = ",".join(str(x) for x in args).encode("utf8")
			r = mlis_lib.mlis_option_set_str(self._ctx, s_opt, s_args)
		elif type(option) == int:
			r = mlis_lib.mlis_option_set(self._ctx, option, *args)
		else:
			raise RuntimeError("'option' must be str or int")
		if r < 0:
			raise RuntimeError("Failed to set option '%s': %s" % (
				option, self.errstr_get()))
	#end

	def setup(self):
		"Set up the backend and model. Optional."
		r = mlis_lib.mlis_setup(self._ctx)
		if r < 0:
			raise RuntimeError("Failed to setup: %s" % (self.errstr_get()))
	#end

	def generate(self):
		"Generate images."
		r = mlis_lib.mlis_generate(self._ctx)
		if r < 0:
			raise RuntimeError("Failed to generate image: %s" % (self.errstr_get()))
	#end

	def image_get(self, idx=0):
		"Get generated images data."
		img_ptr = mlis_lib.mlis_image_get(self._ctx, idx)
		if not img_ptr:
			raise RuntimeError("Failed to get image %d" % idx)
		img = MLIS_Image(img_ptr.contents)
		return img
	#end

	def infotext_get(self, idx=0):
		"Get text describing the generation parameters."
		info = mlis_lib.mlis_infotext_get(self._ctx, idx)
		if info is None:
			raise RuntimeError("Failed to get infotext %d" % idx)
		info = info.decode('utf8')
		return info

	def errstr_get(self):
		"Return an string describing the last error."
		errstr = mlis_lib.mlis_errstr_get(self._ctx)
		if errstr is not None:
			errstr = errstr.decode("utf8")
		return errstr

	def clip_text_encode(self, text, features=False, no_norm=True, 
			model_idx=MLIS_MODEL_CLIP):
		s_text = text.encode("utf8")
		t_embed = mlis_lib.mlis_tensor_get(self._ctx, MLIS_TENSOR_TMP);
		t_feat = None
		flags = 0
		if features:
			t_feat = mlis_lib.mlis_tensor_get(self._ctx, MLIS_TENSOR_TMP+1);
		if no_norm:
			flags |= MLIS_CTEF_NO_NORM
		
		r = mlis_lib.mlis_clip_text_encode(self._ctx, s_text, t_embed, t_feat, 
			model_idx, flags)
		if r < 0:
			raise RuntimeError("Failed to encode text with CLIP: %s" % (
				self.errstr_get()))
		
		embed = MLIS_Tensor(t_embed.contents)
		if features:
			feat = MLIS_Tensor(t_feat.contents)
			return embed, feat
		else:
			return embed
	#end
#end

# Simple test
if __name__ == '__main__':
	mlis = MLImgSynth()
	# Set an option using its id
	mlis.option_set(MLIS_OPT_IMAGE_DIM, 512, 512)
	# Set an option using its name
	mlis.option_set("cfg-scale", 7.0)
	
	## Compute similarity between two text prompts
	if 0:
		mlis.option_set("model",
			"models/sd_v2-1_768-nonema-pruned-fp16.safetensors")
		embed1, feat1 = mlis.clip_text_encode("a blue cat", features=True)
		print(len(embed1.data), embed1.n)
		print(len(feat1.data), feat1.n)
		embed2, feat2 = mlis.clip_text_encode("blue cat", features=True)
		#print(feat1.data[:4], feat2.data[:4])
		print("Similarity: %.3f" % feat1.similarity(feat2))  #0.956
