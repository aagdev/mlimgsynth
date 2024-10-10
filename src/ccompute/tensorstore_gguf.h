/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * Load tensors from a GGUF file.
 */
#pragma once
#include "tensorstore.h"

extern const TensorStoreFormat ts_cls_gguf;

int tstore_detect_gguf(Stream* stm);

int tstore_read_gguf(TensorStore* ts, Stream* stm, TSCallback* cb);
