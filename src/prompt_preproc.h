/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#pragma once
#include "ccommon/ccommon.h"
#include "ccommon/strslice.h"
#include "ccommon/vector.h"
#include "ccommon/logging.h"
#include <math.h>

#ifndef MLIMGSYNTH_H
enum { MLIS_E_PROMPT_PARSE = -5 };
#endif

/* Text prompt structure containing the preprocesed text, weights, and loras.
 */
typedef struct PromptText {
	DynStr text;
	DynStr data;
	struct PromptTextChunk {
		StrSlice text;
		float w;  // Weight / attention multiplier
	} *chunks;  //vector
	struct PromptTextLora {
		StrSlice name;
		float w;
	} *loras;  //vector
} PromptText;

static
void prompt_text_free(PromptText* S)
{
	dstr_free(S->text);
	dstr_free(S->data);
	vec_free(S->chunks);
	vec_free(S->loras);
}

static
void prompt_text_clear(PromptText* S)
{
	dstr_resize(S->text, 0);
	dstr_resize(S->data, 0);
	vec_resize(S->chunks, 0);
	vec_resize(S->loras, 0);
}

static
void prompt_text_set_raw(PromptText* S, const StrSlice ss)
{
	prompt_text_clear(S);
	dstr_copy(S->text, strsl_len(ss), strsl_begin(ss));
	vec_resize(S->chunks, 1);
	S->chunks[0] = (struct PromptTextChunk){ strsl_fromd(S->text), 1.0 };
}

static
int prompt_text_option_parse(PromptText* S, StrSlice ss)
{
	int R=1;

	if (strsl_prefix_trim(&ss, strsl_static("lora:")))
	{
		const char *beg=strsl_begin(ss), *sep=beg, *end=strsl_end(ss);
		while (sep < end && *sep != ':') sep++;  // Find multiplier option
		
		float mult=1;
		if (*sep == ':') {  // Optional multiplier
			char *tail=NULL;
			mult = strtof(sep+1, &tail);
			if (tail != end)
				ERROR_LOG(MLIS_E_PROMPT_PARSE, "prompt: invalid lora multiplier");
		}
		
		//TRY( mlis_cfg_lora_add(S, strsl_fromr(ss.b, sep), mult, MLIS_LF_PROMPT) );

		// Store lora name
		unsigned len = sep - beg;
		dstr_append(S->data, len, beg);

		// Add lora to list
		unsigned nl = vec_count(S->loras);
		vec_append_zero(S->loras, 1);
		S->loras[nl].name = strsl_make(dstr_end(S->data) - len, len);
		S->loras[nl].w = mult;
	}
	else {
		ERROR_LOG(MLIS_E_PROMPT_PARSE, "prompt: unknown option '%.*s'",
			(int)ss.s, ss.b);
	}

end:
	return R;
}

/* Parse prompt like in stable-diffusion-webui.
 * "normal text" -> 1 chunk
 * "normal (weighted by 1.1) normal" -> 3 chunks
 * "normal ((weighted by 1.1*1.1)) normal" -> 3 chunks
 * "normal [weighted by 1/1.1) normal" -> 3 chunks
 * "normal (weighted by 1.5:1.5) normal" -> 3 chunks
 * "normal BREAK normal" -> "normal  normal"  (ignores BREAK for now)
 */
static
int prompt_text_set_parse(PromptText* S, const StrSlice ss)
{
	int R=1;

	prompt_text_clear(S);

	// Reserve memory so that pointers are not invalidated.
	dstr_realloc(S->text, strsl_len(ss)*2);
	dstr_realloc(S->data, strsl_len(ss)*2);

	vec_resize_zero(S->chunks, 1);
	S->chunks[0].text = strsl_make(S->text, 0);
	S->chunks[0].w = 1;
	
	int n_paren=0, n_braket=0;

	strsl_for(ss, cur, end, 0)
	{
		if (*cur == '\\') {  // Escape
			if (cur+1 < end) {
				cur++;
				char c = *cur;
				switch (c) {
				case 'n':  c = '\n';  break;
				}
				dstr_push(S->text, c);
			}
		}
		else if (*cur == '(' || *cur == ')' || *cur == '[' || *cur == ']') {
			switch (*cur) {
			case '(':  n_paren++;  break;
			case ')':  n_paren--;  break;
			case '[':  n_braket++; break;
			case ']':  n_braket--; break;
			}
			if (n_paren < 0 || n_braket < 0)
				ERROR_LOG(MLIS_E_PROMPT_PARSE,
					"prompt: unmatched ')' or ']'");
			//if (n_paren > 0 && n_braket > 0)
			//	ERROR_LOG(MLIS_E_PROMPT_PARSE,
			//		"prompt: mix of emphasis with '(' and '['");

			const char *e = dstr_end(S->text);
			//unsigned lvl = n_paren - n_braket;
			float w = pow(1.1, n_paren - n_braket);  //TODO: cfg?
			
			unsigned ic = vec_count(S->chunks) -1;
			if (S->chunks[ic].text.b == e) {
				S->chunks[ic].w = w;
			} else {
				// Finish previous chunk
				S->chunks[ic].text.s = e - S->chunks[ic].text.b;
				// New chunk
				vec_append_zero(S->chunks, 1);
				ic++;
				S->chunks[ic].text.b = e;
				S->chunks[ic].w = w;
			}
		}
		else if (*cur == ':' && (n_paren > 0 || n_braket > 0)) {
			if (!(n_paren == 1 && n_braket == 0))
				ERROR_LOG(MLIS_E_PROMPT_PARSE,
					"prompt: custom emphasis multiplier outside of '()'");

			char *tail=NULL;
			float w=0;
			if (cur+1 < end) {
				cur++;
				w = strtof(cur, &tail);  //TODO: restrict to an slice
			}
			if (!(tail && tail < end && *tail == ')'))
				ERROR_LOG(MLIS_E_PROMPT_PARSE,
					"prompt: invalid emphasis with ':'");

			cur = tail-1;
			vec_last(S->chunks, 0).w = w;
		}
		else if (*cur == '<') {
			const char *e=cur+1;
			while (e < end && *e != '>') ++e;
			if (*e != '>')
				ERROR_LOG(MLIS_E_PROMPT_PARSE, "prompt: '<' not matched with '>'");
			TRY( prompt_text_option_parse(S, strsl_fromr(cur+1, e)) );
			cur = e;
		}
		else if (*cur == 'B' && cur+5 < end && !memcmp(cur, "BREAK", 5)) {
			cur += 4;
		}
		else dstr_push(S->text, *cur);
	}

	// Finish last chunk
	unsigned ic = vec_count(S->chunks) - 1;
	S->chunks[ic].text = strsl_fromr(S->chunks[ic].text.b, dstr_end(S->text));

#ifndef NDEBUG
	vec_for(S->chunks, i, 0) {
		assert( strsl_begin(S->chunks[i].text) >= S->text );
		assert( strsl_end(S->chunks[i].text) <= dstr_end(S->text) );
	}
#endif

end:
	return R;
}
