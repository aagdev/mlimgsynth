# Makefile
targets = test_rng tstore-util demo_mlimgsynth mlimgsynth \
	test_text_tokenize_clip
targets_dlib = libmlimgsynth

# Put your custom definitions in Makefile.local instead of changing this file
-include Makefile.local

include src/ccommon/base.mk
VPATH = src:src/ccommon:src/ccompute
cppflags += -Isrc -Iinclude
ldflags += -L.

### Dependencies
# math
ldlibs += -lm

# ggml
ifndef GGML_INCLUDE_PATH
GGML_INCLUDE_PATH := ggml/include
endif
ifndef GGML_LIB_PATH
GGML_LIB_PATH := ggml/Release/src
endif
cppflags += -I$(GGML_INCLUDE_PATH)
ldflags += -L$(GGML_LIB_PATH)
# ggml headers give several warnings with C99
cflags += -Wno-pedantic

tstore-util: ldlibs += -lggml -lggml-base
libmlimgsynth: ldlibs += -lggml -lggml-base
ifndef MLIS_NO_RUNPATH
tstore-util: ldflags += -Wl,-rpath,$(GGML_LIB_PATH)
libmlimgsynth: ldflags += -Wl,-rpath,$(GGML_LIB_PATH)
endif

# ggml scheduler is need for incomplete backends (no longer needed for vulkan)
ifdef MLIS_GGML_SCHED
libmlimgsynth: cppflags += -DUSE_GGML_SCHED=1
endif

# Flash Attention (not working yet, crashes)
ifdef MLIS_FLASH_ATTENTION
libmlimgsynth: cppflags += -DUSE_FLASH_ATTENTION
endif

# png
ifndef MLIS_NO_PNG
mlimgsynth: ldlibs += -lpng
mlimgsynth: cppflags += -DUSE_LIB_PNG
mlimgsynth: image_io_png.o
endif

# jpeg
ifndef MLIS_NO_JPEG
mlimgsynth: ldlibs += -ljpeg
mlimgsynth: cppflags += -DUSE_LIB_JPEG
mlimgsynth: image_io_jpeg.o
endif

# libmlimgsynth
demo_mlimgsynth: ldlibs += -lmlimgsynth
mlimgsynth: ldlibs += -lmlimgsynth
test_text_tokenize_clip: ldlibs += -lmlimgsynth
ifndef MLIS_NO_RUNPATH
demo_mlimgsynth: ldflags += -Wl,-rpath,.
mlimgsynth: ldflags += -Wl,-rpath,.
test_text_tokenize_clip: ldflags += -Wl,-rpath,.
endif

# GCC 13.3.1 20240614 warns about dstr_appendz and dstr_insertz
# I think the code is ok, but I will check later
FLAGS=-Wno-array-bounds -Wno-stringop-overflow

### Module dependencies
tensorstore.o: cppflags += -DTENSORSTORE_USE_GGML -DTENSORSTORE_FMT_GGUF \
	-DTENSORSTORE_FMT_SAFET

objs_base = timing.o alloc.o alloc_gen.o stream.o logging.o
objs_tstore = alloc_arena.o stringstore.o fsutil.o \
	any.o structio.o structio_json.o \
	tensorstore.o tensorstore_safet.o tensorstore_gguf.o

### Binary targets
test_rng: $(objs_base) rng_philox.o test_rng.o

tstore-util: $(objs_base) $(objs_tstore) main_tstore_util.o

libmlimgsynth: $(objs_base) $(objs_tstore) rng_philox.o localtensor.o \
	ggml_extend.o mlblock.o mlblock_nn.o tae.o vae.o clip.o unet.o lora.o \
	solvers.o sampling.o tensor_name_conv.o mlimgsynth.o

demo_mlimgsynth: demo_mlimgsynth.o

mlimgsynth: $(objs_base) image.o image_io.o image_io_pnm.o \
	localtensor.o main_mlimgsynth.o

test_text_tokenize_clip: test_text_tokenize_clip.o
