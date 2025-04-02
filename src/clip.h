/* Copyright 2024-2025, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * CLIP text to embeddings encoder for conditioning in SD.
 */
#pragma once
#include "mlblock.h"
#include "localtensor.h"
#include "ccommon/strslice.h"

typedef struct {
	int n_vocab;
	int n_token;  // max_position_embeddings
	int d_embed;
	int n_interm;
	int n_head;
	int n_layer;    // num_hidden_layers
	uint32_t tok_start, tok_end, tok_pad;
} ClipParams;

extern const ClipParams g_clip_vit_l_14;		//SD 1.x and SDXL
extern const ClipParams g_clip_vit_h_14;		//SD 2.x
extern const ClipParams g_clip_vit_bigg_14;		//SDXL

/* Encode a text in to a list of tokens.
 * Return the number of tokens put into <out>.
 * <ptokvec> is a pointer to a vector of tokens where new tokens will be appended.
 */
int clip_tokenize(const ClipParams* P, StrSlice text, int32_t** ptokvec);

/* Decode a token into an string (zero terminated).
 * Returns the number of bytes written, or negative in case of error.
 */
int clip_token_decode(const ClipParams* P, int32_t token,
	size_t bufsz, char* buf);

/* Get the string corresponding to a token.
 * For debuging purposes, uses an internal buffer.
 * Returns "<|INVALID|>" if not found.
 */
const char* clip_token_str(const ClipParams* P, int32_t token);

// In : vector of token ids [n_token]
// Out: embeddings [d_embed, n_token]
MLTensor* mlb_clip_text(MLCtx* C, MLTensor* tokens, MLTensor* cust_emb,
	const ClipParams* P, int clip_skip, bool norm);

// In : embeddings [d_embed, n_token]
// Out: features vector [d_embed]
MLTensor* mlb_clip_text_proj(MLCtx* C, MLTensor* embed, int i_tok_end);

int clip_text_encode(MLCtx* C, const ClipParams* P,
	const int32_t *tokvec, LocalTensor* embed, LocalTensor* feat,
	int clip_skip, bool norm);
