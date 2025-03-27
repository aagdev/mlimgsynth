/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 */
#include "clip.h"
#include "ccommon/ccommon.h"
#include "ccommon/stream.h"
#include "ccommon/logging.h"
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

void clip_tokr_free(ClipTokenizer* S)
{
	strsto_free(&S->vocab);
}

int clip_tokr_vocab_load(ClipTokenizer* S, const char* path)
{
	int R=1;
	Stream stm={0};
	TRY( stream_open_file(&stm, path, SOF_READ) );
	
	const char *end, *cur;
	while ((cur = stream_read_buffer(&stm, &end)) < end) {
		bool eof = stream_end_is(&stm);

		// Read one string per line and stores it in a string-position bi-map.
		while (1) {
			const char *beg=cur;
			while (cur<end && *cur != '\n') cur++;
			if (beg == cur) break;  //empty
			if (cur == end && !eof) { cur=beg; break; }
			TRY( strsto_add(&S->vocab, strsl_fromr(beg, cur)) );
			if (cur == end) break;
			cur++;
		}
		stream_commit(&stm, cur);
		if (eof) break;
	}
	log_debug("CLIP vocabulary size: %u", strsto_count(&S->vocab));

end:
	if (R<0) log_error("reading CLIP vocabulary from '%s'", path);
	stream_close(&stm, 0);
	return R;
}

const char* clip_tokr_word_from_token(const ClipTokenizer* S, int32_t i)
{
	static char tmps[32];
	StrSlice sl = strsto_get(&S->vocab, i);
	// Remove space separating the two elements
	const char *sep=sl.b, *end=sl.b+sl.s;
	while (sep<end && *sep != ' ') sep++;
	memcpy(tmps, sl.b, sep-sl.b);
	if (sep+1<end) {
		memcpy(tmps+(sep-sl.b), sep+1, end-sep-1);
		tmps[end-sl.b-1] = 0;
	} else {
		tmps[sl.s] = 0;
	}
	return tmps;
}

static inline
bool chr_ascii_is(char c) {
	return 32 <= c && c <= 126;
}

static inline
bool chr_whitespace_is(char c) {
	return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

static inline
bool chr_letter_is(char c) {
	return ('A' <= c && c <= 'Z') || ('a' <= c && c <= 'z');
}

static inline
bool chr_digit_is(char c) {
	return '0' <= c && c <= '9';
}

static inline
bool chr_other_is(char c) {
	return chr_ascii_is(c) &&
		!chr_whitespace_is(c) && !chr_letter_is(c) && !chr_digit_is(c);
}

static inline
void str_lower(DynStr str) {
	dstr_for(str,i,0)
		if ('A' <= str[i] && str[i] <= 'Z') str[i] += 'a' - 'A';
}

int clip_tokr_tokenize(ClipTokenizer* S, const char* cur, int32_t** pout)
{
	int R=1;
	DynStr word=NULL, bigram=NULL;
	const char *end = cur + strlen(cur);

	while (1) {
		while (cur<end && chr_whitespace_is(*cur)) cur++;
		if (cur == end) break;

		// Get next word
		const char *beg = cur++;
		if (!chr_ascii_is(*beg))
			ERROR_LOG(-1, "non-ascii character in prompt");
		if (chr_letter_is(*beg)) {
			while (cur<end && chr_letter_is(*cur)) cur++;
		}
		else if (chr_digit_is(*beg)) {
			while (cur<end && chr_digit_is(*cur)) cur++;
		}
		else if (*beg == '\'') {
			if (cur[0] == 's') cur++;
			else if (cur[0] == 't') cur++;
			else if (cur[0] == 'm') cur++;
			else if (cur[0] == 'd') cur++;
			else if (cur[0] == 'r' && cur[1] == 'e') cur+=2;
			else if (cur[0] == 'v' && cur[1] == 'e') cur+=2;
			else if (cur[0] == 'l' && cur[1] == 'l') cur+=2;
			else goto other;
		}
		else {
other:
			while (cur<end && chr_other_is(*cur)) cur++;
		}

		size_t len = cur-beg;
		dstr_copy(word, len, beg);
		str_lower(word);
		dstr_appendz(word, "</w>");
		
		// Encode list of n-grams with a list of separation positions
		// "dog</w>" -> 0 1 2 7 = ["d", "o", "g</w>"]
		uint8_t *breaks=vec_stack(uint8_t,32);
		vec_resize(breaks,len+1);
		vec_for(breaks,i,0) breaks[i] = i;
		breaks[len] = vec_count(word);  //including </w>

		// List of tokens found. Token = vocabulary position.
		int32_t *tokens=vec_stack(int32_t,32);
		vec_resize(tokens,len);
		vec_for(tokens,i,0) {
			unsigned i1=breaks[i], i2=breaks[i+1];
			tokens[i] = strsto_find(&S->vocab, (StrSlice){word+i1, i2-i1});
			assert( tokens[i] >= 0 );
		}

		// BPE (byte pair encoding)
		unsigned nvocab = strsto_count(&S->vocab);
		while (vec_count(tokens) >= 2) {
			StringInt best_iv=nvocab;
			unsigned best_ib=vec_count(breaks);
			vec_for(breaks,ib,2) {
				uint8_t i1=breaks[ib-2], i2=breaks[ib-1], i3=breaks[ib];
				dstr_copy(bigram, i2-i1, word+i1);
				dstr_push(bigram, ' ');
				dstr_append(bigram, i3-i2, word+i2);
				StringInt iv = strsto_find(&S->vocab, strsl_fromd(bigram));
				//printf("bigram: %s %d\n", bigram, iv);
				if (0 <= iv && iv < best_iv) { best_iv=iv; best_ib=ib; }
			}
			if (best_iv == nvocab) break;
			vec_remove(breaks, best_ib-1, 1);
			vec_remove(tokens, best_ib-2, 1);
			tokens[best_ib-2] = best_iv;
		}
		//printf("wtokens:");
		//vec_for(tokens,i,0)
		//	printf(" %d '%s'", tokens[i], strsto_get(&S->vocab, tokens[i]).b);
		//printf("\n");

		vec_for(tokens,i,0) vec_push(*pout, tokens[i]);
	}

end:
	if (R<0) log_error("CLIP tokenizer");
	dstr_free(bigram);
	dstr_free(word);
	return R;
}

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
