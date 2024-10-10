/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * Conversions from the multiple model tensor naming schemes to internal names.
 */
#pragma once
#include "ccommon/vector.h"
#include "ccommon/strslice.h"

int tnconv_sd(StrSlice name, DynStr *out);

enum tensor_name_convert_result_t {
	TNCONV_R_UNUSED = 0,
	TNCONV_R_GOOD = 1,
	TNCONV_R_QKV_PROJ = 2,
};
