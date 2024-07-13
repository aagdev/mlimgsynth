# Makefile
targets = rng-test st-util mlimgsynth

include src/ccommon/base.mk
VPATH = src:src/ccommon
cppflags += -Isrc
ldlibs += -lm

### Dependencies
# ccommon
common = timing.o alloc.o stream.o logging.o alloc_small.o stringstore.o fsutil.o \
	any.o structio.o structio_json.o image.o image_io.o image_io_pnm.o

# ggml
ifndef GGML_INCLUDE_PATH
GGML_INCLUDE_PATH := ggml/include
endif
ifndef GGML_LIB_PATH
#GGML_LIB_PATH := ggml/Debug/src
GGML_LIB_PATH := ggml/Release/src
endif
cppflags += -I$(GGML_INCLUDE_PATH)
cflags += -Wno-pedantic
ldlibs += -lggml
ldflags += -L$(GGML_LIB_PATH) -Wl,-rpath,$(GGML_LIB_PATH)

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

### Binary targets
rng-test: $(common) rng_philox.o rng-test.o

st-util: $(common) ids.o tensorstore.o safetensors.o st-util.o

#mlimgsynth: cppflags += -DUSE_FLASH_ATTENTION
mlimgsynth: cppflags += -DUSE_GGML_SCHED=1
mlimgsynth: $(common) ids.o localtensor.o tensorstore.o safetensors.o \
	ggml_extend.o mlblock.o mlblock_nn.o rng_philox.o tae.o vae.o clip.o unet.o \
	solvers.o util.o mlimgsynth.o
