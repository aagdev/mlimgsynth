/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * CLIP text to embeddings encoder for conditioning in SD.
 */
#pragma once
#include "mlblock.h"
#include "localtensor.h"
#include "ccommon/stringstore.h"

typedef struct {
	StringStore vocab;
} ClipTokenizer;

void clip_tokr_free(ClipTokenizer*);

static inline
bool clip_good(const ClipTokenizer* S) { return strsto_count(&S->vocab); }

int clip_tokr_vocab_load(ClipTokenizer*, const char* path);

const char* clip_tokr_word_from_token(const ClipTokenizer* S, int32_t i);

/* Examples:
 * "A dog longjumps...": "a</w>" "dog</w>" "long" "jumps</w>" "...</w>"
 */
int clip_tokr_tokenize(ClipTokenizer*, const char* text, int32_t** poutvec);

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
