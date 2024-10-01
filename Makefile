# Makefile
targets = test_rng tstore-util mlimgsynth

# Put your custom definitions in Makefile.local instead of changing this file
-include Makefile.local

include src/ccommon/base.mk
VPATH = src:src/ccommon:src/ccompute
cppflags += -Isrc
ldlibs += -lm

### Dependencies

# ggml
ifndef GGML_INCLUDE_PATH
GGML_INCLUDE_PATH := ggml/include
endif
ifndef GGML_LIB_PATH
#GGML_LIB_PATH := ggml/Debug/src
#GGML_LIB_PATH := ggml/RelWithDebInfo/src
GGML_LIB_PATH := ggml/Release/src
endif
cppflags += -I$(GGML_INCLUDE_PATH)
cflags += -Wno-pedantic
ldlibs += -lggml
ldflags += -L$(GGML_LIB_PATH) -Wl,-rpath,$(GGML_LIB_PATH)

## ggml scheduler is need for incomplete backends (no longer needed for vulkan)
ifdef MLIS_GGML_SCHED
mlimgsynth: cppflags += -DUSE_GGML_SCHED=1
endif

## Flash Attention (not working yet, crashes)
ifdef MLIS_FLASH_ATTENTION
mlimgsynth: cppflags += -DUSE_FLASH_ATTENTION
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

### Module dependencies
tensorstore.o: cppflags += -DTENSORSTORE_USE_GGML -DTENSORSTORE_FMT_GGUF \
	-DTENSORSTORE_FMT_SAFET

# ccommon
common = timing.o alloc.o alloc_gen.o stream.o logging.o \
	alloc_arena.o stringstore.o fsutil.o \
	any.o structio.o structio_json.o image.o image_io.o image_io_pnm.o

### Binary targets
test_rng: $(common) rng_philox.o test_rng.o

tstore-util: $(common) tensorstore.o tensorstore_safet.o tensorstore_gguf.o \
	main_tstore_util.o

mlimgsynth: $(common) rng_philox.o localtensor.o \
	tensorstore.o tensorstore_safet.o tensorstore_gguf.o \
	ggml_extend.o mlblock.o mlblock_nn.o tae.o vae.o clip.o unet.o \
	solvers.o util.o main_mlimgsynth.o
