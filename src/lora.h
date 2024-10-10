/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * LoRA: low rank adaptation.
 * Ref.: Hu et al. (2021) "LoRA..."
 */
#pragma once
#include "ccompute/tensorstore.h"
#include "mlblock.h"

int lora_apply(TensorStore* ts_dst, TensorStore* ts_lora, float mult,
	MLCtx* ctx);
