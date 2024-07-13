/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "ccommon/image_io.h"

int img_save_file_info(const Image* img, const char* path,
	const char* info_key, const char* info_text);
