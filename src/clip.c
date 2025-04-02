/* Copyright 2024-2025, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 */
#include "clip.h"
#include "ccommon/ccommon.h"
#include "ccommon/bisect.h"
#include "ccommon/stream.h"
#include "ccommon/logging.h"
#include "ccommon/unicode.h"
#include "ccommon/unicode_data.h"
#include "str_match_util.h"
#include "ggml_extend.h"
#include "mlblock_nn.h"

#define MLN(NAME,X)  mlctx_tensor_add(C, (NAME), (X))

// The GGML scheduler have problems with inplace operations (2024-07-13)
#if USE_GGML_SCHED
	#define ggml_gelu_inplace  ggml_gelu
	#define ggml_gelu_quick_inplace  ggml_gelu_quick
#endif

const ClipParams g_clip_vit_l_14 = {
	.n_vocab	= 49408,
	.n_token	= 77,
	.d_embed	= 768,
	.n_interm	= 3072,  //d_embed*4
	.n_head		= 12,
	.n_layer	= 12,
	.tok_start	= 49406,
	.tok_end	= 49407,
	.tok_pad	= 49407,
};

const ClipParams g_clip_vit_h_14 = {
	.n_vocab	= 49408,
	.n_token	= 77,
	.d_embed	= 1024,
	.n_interm	= 4096,  //d_embed*4
	.n_head		= 16,
	.n_layer	= 24,
	.tok_start	= 49406,
	.tok_end	= 49407,
	.tok_pad	= 0,
};

const ClipParams g_clip_vit_bigg_14 = {
	.n_vocab	= 49408,
	.n_token	= 77,
	.d_embed	= 1280,
	.n_interm	= 5120,  //d_embed*4
	.n_head		= 20,
	.n_layer	= 32,
	.tok_start	= 49406,
	.tok_end	= 49407,
	.tok_pad	= 0,
};

/* Tokenizer
 * ref: https://github.com/openai/CLIP : clip/simple_tokenizer.py
 */

const
struct BpeMerge {
	int32_t left, right;
} g_clip_merges[] = {
#include "clip_merges.c.h"
};

int32_t g_clip_merges_index[COUNTOF(g_clip_merges)] = {-1};

static inline
int merge_cmp(const struct BpeMerge* A, const struct BpeMerge* B)
{
	int64_t a = (((int64_t)A->left) << 32) | (int64_t)A->right;
	int64_t b = (((int64_t)B->left) << 32) | (int64_t)B->right;
	return ccSIGN(a - b);
}	

int32_t clip_tokr_merge_get(int32_t left, int32_t right)
{
	if (g_clip_merges_index[0] == -1) {
		// Initialize sorted index
		for (unsigned i=0; i<COUNTOF(g_clip_merges); ++i) {
			BISECT_RIGHT_DECL(found, idx, 0, i,
				merge_cmp(&g_clip_merges[g_clip_merges_index[i_]], 
					&g_clip_merges[i]) )
			
			assert( !found );
			memmove(g_clip_merges_index+idx+1, g_clip_merges_index+idx,
				(i - idx) * sizeof(*g_clip_merges_index));
			g_clip_merges_index[idx] = i;
		}		
	}

	// Search
	BISECT_RIGHT_DECL(found, idx, 0, COUNTOF(g_clip_merges_index),
		merge_cmp(&g_clip_merges[g_clip_merges_index[i_]],
			&(struct BpeMerge){left, right}) )

	return found ? g_clip_merges_index[idx]+512 : 0x7fffffff;
}

int32_t clip_tokr_token_to_merge(int32_t token, int32_t* right)
{
	if (!(512 <= token && token < 512+COUNTOF(g_clip_merges)))
		return -1;
	int32_t merge = token - 512;
	*right = g_clip_merges[merge].right;
	return g_clip_merges[merge].left;
}

/* Given a byte from an utf8-encoded string, returns its token in the vocabulary.
 * Tokens 0 to 255 of CLIP's vocabulary are used for this.
 * Tokens 256 to 512 are the same, but with the end of word indicator.
 */
int clip_tokr_byte_to_token(char byte)
{
	int b = (uint8_t)byte;
	if      (b <=  32) b = b + 188;
	else if (b <= 126) b = b - 33;
	else if (b <= 160) b = b + 94;
	else if (b <= 172) b = b - 67;
	else if (b == 173) b = 255;
	else               b = b - 68;
	assert( 0 <= b && b <= 255 );
	return b;
}

int clip_tokr_token_to_byte(int token)
{
	if      (token <=  93) token += 33;
	else if (token <= 105) token += 67;
	else if (token <= 187) token += 68;
	else if (token <= 220) token -= 188;
	else if (token <= 254) token -= 94;
	else if (token == 255) token = 173;
	else return -1;
	assert( 0 <= token && token <= 255 );
	return token;
}

/* Given a word, writes a list byte tokens to start bpe.
 * Returns the number of tokens or negative in case of error.
 */
int clip_tokr_word_to_byte_tokens(const StrSlice word, size_t tokens_max,
	int32_t* tokens)
{
	char buf[8];
	size_t count=0;
	const char *cur = strsl_begin(word), *end = strsl_end(word);
	while (cur < end) {
		uint32_t cp = utf8_decode_next(&cur, end);
		cp = unicode_lower(cp);  // Lower case
		char *e = utf8_encode_next(buf, cp);
		for (char *c=buf; c<e; ++c) {
			if (count == tokens_max) {
				log_error("word too long (%d)", (int)strsl_len(word));
				return -1;
			}
			count++;
			*tokens++ = clip_tokr_byte_to_token(*c);
		}
	}
	return count;
}

/* Perform byte pair encoding (bpe) with merges.
 */
int clip_tokr_bpe_merges(const StrSlice word, size_t tokens_max, int32_t* tokens)
{
	int count;
	
	// Word to byte tokens
	TRYR( count = clip_tokr_word_to_byte_tokens(word, tokens_max, tokens) );
	if (count == 0) return 0;  // Empty word
	tokens[count-1] += 256;  // Mark last token as end-of-word

	// Recursively merge tokens
	while (count > 1) {
		//if (log_line_begin(LOG_LVL_DEBUG)){
		//	for (int i=0; i<count; ++i)
		//		log_line_strf(" %d '%s'", tokens[i],
		//			clip_token_str(&g_clip_vit_l_14, tokens[i]) );
		//	log_line_end();
		//}
		// Find best merge (smallest token)
		int32_t best_tok = 0x7fffffff;
		int best_pos = 0;
		for (int i=1; i<count; ++i) {
			int32_t tok = clip_tokr_merge_get(tokens[i-1], tokens[i]);
			assert( tok >= 512 );
			if (tok < best_tok) {
				best_tok = tok;
				best_pos = i;
			}
		}
		if (best_tok == 0x7fffffff)  // No merge found
			break;
		// Merge tokens
		tokens[best_pos-1] = best_tok;
		for (int i=best_pos+1; i<count; ++i) tokens[i-1] = tokens[i];
		count--;
	}

	return count;
}

/* Get the next chunk of text according to CLIP tokenizer rules.
 * Original regex (with IGNORECASE):
 * <\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+
 */
StrSlice clip_tokr_word_split(StrSlice *pss)
{
	const char *cur = strsl_begin(*pss),
	           *end = strsl_end(*pss),
			   *beg = cur;

	// Skip whitespace
	beg = cur = str_unicode_space_skip(cur, end);

	// Match
	int cat_in_progress = 0;
	while (cur < end) {
		const char *prev = cur;

		// Match strings
		int m = str_match_advance_multiple(&cur, end, true,
			//"<|startoftext|>", "<|endoftext|>",  no use in practice
			"'s", "'t", "'re", "'ve", "'m", "'ll", "'ve");
		if (m >= 0) {
			if (cat_in_progress) cur = prev;
			break;
		}

		// Match codepoint categories
		uint32_t cp = utf8_decode_next(&cur, end);
		int cat = chr_ascii_space_is(cp) ? 'Z' : unicode_category_major(cp);
		if (cat == 'Z') {  // Whitespace
			cur = prev;
			break;
		}
		if (cat != 'N' && cat != 'L') cat = 'P';
		if (!cat_in_progress) cat_in_progress = cat;
		else if (cat != cat_in_progress) {
			cur = prev;
			break;
		}
	}

	*pss = strsl_fromr(cur, end);
	return strsl_fromr(beg, cur);
}

int clip_tokenize(const ClipParams* P, StrSlice text, int32_t** pout)
{
	int R=1, count;

	assert( P->n_vocab == COUNTOF(g_clip_merges)+512+2 );
	
	int pos = vec_count(*pout),
	    max = pos + strsl_len(text);
	vec_realloc(*pout, max);  // pre-alloc

	while (1) {
		StrSlice word = clip_tokr_word_split(&text);
		if (!strsl_len(word)) break;  //TODO: test "   "
		//log_debug("word: '%.*s'", (int)strsl_len(word), strsl_begin(word));

		TRY( count = clip_tokr_bpe_merges(word, max-pos, (*pout)+pos) );
		pos += count;
		vec_resize(*pout, pos);
	}

end:
	if (R<0) log_error("CLIP tokenizer");
	return R;
}

int clip_token_decode(const ClipParams* P, int32_t token,
	size_t bufsz, char* buf)
{
	assert( P->n_vocab == COUNTOF(g_clip_merges)+512+2 );

	if (token < 0) {
		return -1;
	} else if (token <= 256) {
		if (bufsz < 1) return -1;
		buf[0] = clip_tokr_token_to_byte(token);
		return 1;
	}
	else if (token <= 511) {
		if (bufsz < 2) return -1;
		buf[0] = clip_tokr_token_to_byte(token - 256);
		buf[1] = ' ';
		return 2;
	}
	else {
		int r1, r2;
		int32_t right, left;
		TRYR( left = clip_tokr_token_to_merge(token, &right) );
		TRYR( r1 = clip_token_decode(P, left , bufsz, buf) );
		TRYR( r2 = clip_token_decode(P, right, bufsz-r1, buf+r1) );
		return r1 + r2;
	}
}

const char* clip_token_str(const ClipParams* P, int32_t token)
{
	static char buffer[128];
	int r = clip_token_decode(P, token, sizeof(buffer), buffer);
	if (r < 0) return "<|INVALID|>";
	buffer[r] = 0;
	return buffer;
}

/* Model code */

MLTensor* mlb_clip_embeddings(MLCtx* C, MLTensor* x, MLTensor* tw,
	int d_embed, int n_vocab, int n_token)
{
	MLTensor *pw;
	mlctx_block_begin(C);
	// x: [N, n_token]
	
	if (tw) {
		GGML_ASSERT(tw->ne[0] == d_embed);
	} else {
		tw = MLN("token.weight",
			ggml_new_tensor_2d(C->cp, C->c.wtype, d_embed, n_vocab));
	}
	
	pw = MLN("position.weight",
		ggml_new_tensor_2d(C->cp, GGML_TYPE_F32, d_embed, n_token));
	
	// token_embedding
	x = ggml_reshape_3d(C->cc, x, x->ne[0], 1, x->ne[1]);
	x = ggml_get_rows(C->cc, tw, x);
	x = ggml_reshape_3d(C->cc, x, x->ne[0], x->ne[1], x->ne[3]);

	// token_embedding + position_embedding
	x = ggml_add(C->cc, x, pw);  // [N, n_token, d_embed]
	return x;
}

MLTensor* mlb_clip_mlp(MLCtx* C, MLTensor* x,
	int d_model, int n_interm)
{
	mlctx_block_begin(C);
	// x: [N, n_token, d_model]

	x = MLN("fc1", mlb_nn_linear(C, x, n_interm, true));
	if (d_model == 1024 || d_model == 1280) {  //SD2 or SDXL
		x = ggml_gelu_inplace(C->cc, x);
	} else {  //SD1
		x = ggml_gelu_quick_inplace(C->cc, x);
	}
	x = MLN("fc2", mlb_nn_linear(C, x, d_model, true));
	return x;
}

MLTensor* mlb_clip_layer(MLCtx* C, MLTensor* x,
	int d_model, int n_head, int n_interm, bool mask)
{
	MLTensor *x0=x;	
	mlctx_block_begin(C);
	// x: [N, n_token, d_model]

	x = MLN("norm1", mlb_nn_layer_norm(C, x, true, true, 0));
	x = MLN("attn", mlb_attn_mhead(C, x,x,x,
		d_model, d_model, n_head, mask, true, true));
	x0 = x = ggml_add(C->cc, x0, x);
	x = MLN("norm2", mlb_nn_layer_norm(C, x, true, true, 0));
	x = MLN("mlp", mlb_clip_mlp(C, x, d_model, n_interm));
	x = ggml_add(C->cc, x0, x);
	return x;
}

// transformer
MLTensor* mlb_clip_encoder(MLCtx* C, MLTensor* x,
	int n_layer, int d_model, int n_head, int n_interm, bool mask)
{
	char name[64];
	mlctx_block_begin(C);
	// x: [N, n_token, d_model]

	for (int i=0; i<n_layer; ++i) {
		sprintf(name, "layers.%d", i);
		x = MLN(name, mlb_clip_layer(C, x, d_model, n_head, n_interm, mask));
		// [N, n_token, d_model]
	}
	return x;
}

MLTensor* mlb_clip_text(MLCtx* C, MLTensor* x, MLTensor* cust_emb_w,
	const ClipParams* P, int clip_skip, bool norm)
{
	mlctx_block_begin(C);
	// x: [N, n_token]

	x = MLN("embed", mlb_clip_embeddings(C, x, cust_emb_w,
		P->d_embed, P->n_vocab, P->n_token));
	// [N, n_token, d_embed]

	int n_layer = P->n_layer;
	if (clip_skip > 1) n_layer -= clip_skip-1;
	x = MLN("encoder", mlb_clip_encoder(C, x,
		n_layer, P->d_embed, P->n_head, P->n_interm, true));
	// [N, n_token, d_embed]

	if (norm)
		x = MLN("ln_final", mlb_nn_layer_norm(C, x, true, true, 0));
	// [N, n_token, d_embed]

	return x;
}

MLTensor* mlb_clip_text_proj(MLCtx* C, MLTensor* x, int i_tok_end)
{
	//mlctx_block_begin(C);
	// x: [N, n_token, d_embed]
	
	int d_embed = x->ne[0],
	    n_proj = d_embed;  //always good?

	MLTensor *p = MLN("text_proj",
		ggml_new_tensor_2d(C->cp, GGML_TYPE_F32, n_proj, d_embed));
	p = ggml_cont(C->cc, ggml_transpose(C->cc, p));

	// Take features from the end token
	x = ggml_view_1d(C->cc, x, d_embed, x->nb[1] * i_tok_end);

	x = ggml_mul_mat(C->cc, p, x);
	// [d_embed]
	
	return x;
}

int clip_text_encode(MLCtx* C, const ClipParams* P,
	const int32_t *tokvec, LocalTensor* embed, LocalTensor* feat,
	int clip_skip, bool norm)
{
	int R=1;
	int32_t *tokens=NULL;

	if (feat) { clip_skip=-1; norm=true; }

	// Prepare tokens
	unsigned ntok = vec_count(tokvec);
	if (ntok+2 > P->n_token)
		ERROR_LOG(-1, "prompt too long (max: %d)", P->n_token-2);
	vec_resize(tokens, P->n_token);
	tokens[0] = P->tok_start;
	ARRAY_COPY(tokens+1, tokvec, ntok);
	tokens[ntok+1] = P->tok_end;
	vec_for(tokens,i,ntok+2) tokens[i] = P->tok_pad;
	
	//log_debug_vec("Tokens", tokens, i, 0, "%u", tokens[i]);

	// Prepare computation
	mlctx_begin(C, "CLIP text encode");

	MLTensor *input = mlctx_input_new(C, "tokens", GGML_TYPE_I32, P->n_token,1,1,1);
	MLTensor *t_embed = mlb_clip_text(C, input, NULL, P, clip_skip, norm);

	MLTensor *result=t_embed, *t_feat=NULL;
	if (feat)
		result = t_feat = mlb_clip_text_proj(C, t_embed, ntok+1);

	mlctx_tensor_add(C, "text", result);
	TRY( mlctx_prep(C) );

	// Set input
	ggml_backend_tensor_set(input, tokens, 0, vec_bytesize(tokens));

	// Compute
	TRY( mlctx_compute(C) );

	// Get output
	if (embed)
		ltensor_from_backend(embed, t_embed);
	if (feat)
		ltensor_from_backend(feat, t_feat);

end:
	vec_free(tokens);
	return R;
}
