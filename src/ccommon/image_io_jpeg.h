/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#pragma once
#include "image_io.h"

#ifdef IMGIO_JPEG_IMPL
#include "vector.h"
#include <jpeglib.h>
#include <setjmp.h>

struct img_codec_jpeg_error_mgr {
	struct jpeg_error_mgr	errmgr;
	jmp_buf					escape;
};

struct CodecJpegLoad {
	struct jpeg_decompress_struct cinfo;
	struct img_codec_jpeg_error_mgr jerr;
};

struct CodecJpegSave {
	struct jpeg_compress_struct cinfo;
	struct img_codec_jpeg_error_mgr jerr;

	struct CodecJpegText { DynStr key, value; } *metadata;  //vector
	int quality;
};
#endif

typedef struct CodecJpegLoad CodecJpegLoad;
typedef struct CodecJpegSave CodecJpegSave;

bool imgio_jpeg_detect(Stream* s, const char* fileext);

int  imgio_jpeg_load_init(CodecJpegLoad* codec, ImageIO* imgio);
void imgio_jpeg_load_free(CodecJpegLoad* codec, ImageIO* imgio);
int  imgio_jpeg_load_op(CodecJpegLoad* codec, ImageIO* imgio, Image* img);

int  imgio_jpeg_save_init(CodecJpegSave* codec, ImageIO* imgio);
void imgio_jpeg_save_free(CodecJpegSave* codec, ImageIO* imgio);
int  imgio_jpeg_save_op(CodecJpegSave* codec, ImageIO* imgio, Image* img);

extern const ImageCodec img_codec_jpeg;

