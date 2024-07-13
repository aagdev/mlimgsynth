/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#pragma once
#include "image_io.h"

#ifdef IMGIO_PNG_IMPL
#include "vector.h"
struct CodecPng {
	struct CodecPngText { DynStr key, value; } *metadata;  //vector
	int comp_lvl;
};
#endif

typedef struct CodecPng CodecPng;

bool imgio_png_detect(Stream* s, const char* fileext);

int imgio_png_load(void* self, ImageIO* imgio, Image* img);

int  imgio_png_save_init(CodecPng* S, ImageIO* imgio);
void imgio_png_save_free(CodecPng* S, ImageIO* imgio);
int  imgio_png_save_op(CodecPng* S, ImageIO* imgio, Image* img);
int  imgio_png_value_set(CodecPng* S, ImageIO* imgio,
		int id, const void* buf, unsigned bufsz);

extern const ImageCodec img_codec_png;
