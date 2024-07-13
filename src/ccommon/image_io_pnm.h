/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#pragma once
#include "image_io.h"

bool imgio_pnm_detect(Stream* s, const char* fileext);

int imgio_pnm_load(void* self, ImageIO* imgio, Image* img);

int imgio_pnm_save(void* self, ImageIO* imgio, Image* img);

extern const ImageCodec img_codec_pnm;

