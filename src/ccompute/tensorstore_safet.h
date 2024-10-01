/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * Load/save tensor from a file with the SafeTensor format.
 */
#pragma once
#include "tensorstore.h"

extern const TensorStoreFormat ts_cls_safet;

int tstore_detect_safet(Stream* stm);

int tstore_read_safet(TensorStore*, Stream*);

int tstore_write_safet(TensorStore*, Stream*);
