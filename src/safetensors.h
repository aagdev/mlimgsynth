/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * Load/save tensor from a file with the SafeTensor format.
 */
#pragma once
#include "tensorstore.h"

int safet_load_head(TensorStore*, Stream*, const char* prefix);

int safet_save_head(TensorStore*, Stream*, const char* prefix);
