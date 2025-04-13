/* Copyright 2025, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * Test of the prompt preprocessing.
 */
#include "prompt_preproc.h"
#include "test_common.h"

typedef struct {
	const char *text;
	float w;
} TestChunk;

typedef struct {
	unsigned n_chunk, n_lora;
	TestChunk *chunks;
	TestChunk *loras;
} TestPrompt;

#define assert_chunk(A, B, ...) do { \
	const struct PromptTextChunk a = (A); \
	const TestChunk b = (B); \
	if (strsl_cmpz(a.text, b.text) || a.w != b.w) { \
		DynStr errstr=NULL; \
		dstr_printf(errstr, "'%.*s' %g != '%s' %g", (int)strsl_len(a.text), \
			strsl_begin(a.text), a.w, b.text, b.w); \
		error(__VA_ARGS__); \
	} \
} while(0)

#define assert_lora(A, B, ...) do { \
	const struct PromptTextLora a = (A); \
	const TestChunk b = (B); \
	if (strsl_cmpz(a.name, b.text) || a.w != b.w) { \
		DynStr errstr=NULL; \
		dstr_printf(errstr, "'%.*s' %g != '%s' %g", (int)strsl_len(a.name), \
			strsl_begin(a.name), b.w, b.text, b.w); \
		error(__VA_ARGS__); \
	} \
} while(0)

#define CHUNKS(...) \
	.n_chunk=sizeof((TestChunk[]){__VA_ARGS__, {0}})/sizeof(TestChunk)-1, \
	.chunks=(TestChunk[]){__VA_ARGS__, {0}}

#define LORAS(...) \
	.n_lora=sizeof((TestChunk[]){__VA_ARGS__, {0}})/sizeof(TestChunk)-1, \
	.loras=(TestChunk[]){__VA_ARGS__, {0}}

#define TEST(TEXT, ...) \
	test((TEXT), (TestPrompt){__VA_ARGS__})

static
void assert_prompt(const PromptText pt, const TestPrompt exp, const char *text)
{
	assert_int( vec_count(pt.chunks), exp.n_chunk,
		"in '%s':\nchunks returned: %d, expected: %d", text, a, b);
	
	assert_int( vec_count(pt.loras), exp.n_lora,
		"in '%s':\nloras returned: %d, expected: %d", text, a, b);

	for (unsigned i=0; i<exp.n_chunk; ++i) {
		assert_chunk(pt.chunks[i], exp.chunks[i],
			"in '%s':\nchunk %u: %s", text, i, errstr);
	}

	for (unsigned i=0; i<exp.n_lora; ++i) {
		assert_lora(pt.loras[i], exp.loras[i],
			"in '%s':\nlora %u: %s", text, i, errstr);
	}
}

static
void test(const char* text, const TestPrompt exp)
{
	debug("%s", text);

	int r;
	PromptText pt={0};
	r = prompt_text_set_parse(&pt, strsl_fromz(text));
	if (r < 0)
		error("prompt_text_set_parse('%s'): 0x%x", text, -r);
	
	assert_prompt(pt, exp, text);
	
	prompt_text_free(&pt);
}

static
void test_raw(const char* text)
{
	debug("%s", text);
	PromptText pt={0};
	prompt_text_set_raw(&pt, strsl_fromz(text));
	assert_prompt(pt, (TestPrompt){ CHUNKS({text, 1}) }, text);
	prompt_text_free(&pt);
}

int main(int argc, char* argv[])
{
	test_raw("a (dog:1.5) jumping [in] the ((park))");

	// Simple
	TEST("a dog jumping",
		CHUNKS({"a dog jumping", 1}));
	// Emphasis
	TEST("a (dog) jumping",
		CHUNKS({"a ", 1}, {"dog", 1.1}, {" jumping", 1}));
	TEST("a [dog] jumping",
		CHUNKS({"a ", 1}, {"dog", 1/1.1}, {" jumping", 1}));
	TEST("a ((dog)) jumping",
		CHUNKS({"a ", 1}, {"dog", 1.1*1.1}, {" jumping", 1}));
	TEST("a (dog:1.5) jumping",
		CHUNKS({"a ", 1}, {"dog", 1.5}, {" jumping", 1}));
	// Loras
	TEST("a dog jum<lora:LORA NAME>ping",
		CHUNKS({"a dog jumping", 1}),
		LORAS({"LORA NAME", 1}));
	TEST("a dog jum<lora:LORA NAME:0.8>ping",
		CHUNKS({"a dog jumping", 1}),
		LORAS({"LORA NAME", 0.8}));
	// Escapes
	TEST("a \\(dog\\) jumping",
		CHUNKS({"a (dog) jumping", 1}));
	TEST("a dog jum\\<lora:LORA NAME>ping",
		CHUNKS({"a dog jum<lora:LORA NAME>ping", 1}));

	log("TEST OK "__FILE__);
	return 0;
}
