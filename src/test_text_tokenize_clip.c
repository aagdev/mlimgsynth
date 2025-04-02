/* Copyright 2025, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * Test of the CLIP tokenizer.
 */
#include "mlimgsynth.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define error(...) do { \
	printf("ERROR "); \
	printf(__VA_ARGS__); \
	printf("\n"); \
	exit(1); \
} while (0)

#define log(...) do { \
	printf(__VA_ARGS__); \
	printf("\n"); \
} while (0)

#ifdef NDEBUG
#define debug(...)
#else
#define debug(...) do { \
	printf("DEBUG "); \
	printf(__VA_ARGS__); \
	printf("\n"); \
} while (0)
#endif

#define TEST(TEXT, ...) \
	test(ctx, (TEXT), (const int32_t[]){__VA_ARGS__, -1})

#define TEST_EMPTY(TEXT) \
	test(ctx, (TEXT), (const int32_t[]){-1})

static
void test(MLIS_Ctx* ctx, const char* text, const int32_t* expected)
{
	debug("%s", text);

	const int32_t *tokens=NULL;
	int r = mlis_text_tokenize(ctx, text, &tokens, MLIS_MODEL_CLIP);
	if (r < 0)
		error("mlis_tokenize('%s'): 0x%x", text, -r);
	
	int i;
	for (i=0; i<r; ++i) {
		if (tokens[i] != expected[i])
			error("in '%s':\ntoken[%d] = %d != %d",
				text, i, tokens[i], expected[i]);
	}
	if (expected[i] != -1)
		error("in '%s':\n%d tokens returned, but expected more", text, r);
}

int main(int argc, char* argv[])
{
	MLIS_Ctx *ctx = mlis_ctx_create();
	mlis_option_set(ctx, MLIS_OPT_MODEL_TYPE, MLIS_MODEL_TYPE_SD1);

	// Simple
	TEST("a dog jumping", 320, 1929, 11476);
	// Superflous spacing
	TEST("   a   dog\t\tjumping\r\n", 320, 1929, 11476);
	// Merges are important
	TEST("an illustration", 550, 6052);  
	// Quotes
	TEST("a sign saying \"Here lies Cesar\"", 320, 2292, 4455, 257, 763, 3205, 28603, 257);
	TEST("a sign saying 'Here lies Cesar'", 320, 2292, 4455, 262, 763, 3205, 28603, 262);
	// Number
	TEST("2025", 17, 15, 17, 276);
	// English contractions
	TEST("A'veA'llA's", 320, 1200, 320, 1342, 320, 568);
	// Empty
	TEST_EMPTY("");
	// Space only
	TEST_EMPTY("  \t  \n");
	// Puntuation
	TEST("a dog, a house.", 320, 1929, 267, 320, 1212, 269);
	// UTF-8
	TEST("coraz\xc3\xb3n", 851, 854, 13926);
	// Unicode dash in between ascii ones
	TEST("cat---dog-\xe2\x80\x94-rabbit", 2368, 11079, 1929, 12, 6718, 268, 10274);
	// Unicode word split. Japanese: "Maa, machinanasai."
	TEST("\xe3\x81\xbe\xe3\x81\x82\xe3\x80\x81\xe3\x81\x8a\xe5\xbe\x85\xe3\x81\xa1\xe3\x81\xaa\xe3\x81\x95\xe3\x81\x84\xe3\x80\x82", 4813, 122, 4813, 480, 45262, 4813, 232, 161, 122, 227, 4813, 94, 29104, 4813, 243, 38850, 38000);
	// Long text. Split words.
	TEST("Stable Diffusion is a deep learning, text-to-image model released in 2022 based on diffusion techniques.", 10492, 18656, 9364, 533, 320, 3383, 2378, 267, 4160, 268, 531, 268, 2867, 2863, 3410, 530, 17, 15, 17, 273, 2812, 525, 18656, 9364, 1782, 697, 7715, 269);

	log("done");
	mlis_ctx_destroy(&ctx);
	return 0;
}
