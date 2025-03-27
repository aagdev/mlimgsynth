/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
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

#define TEST(TEXT, ...) \
	test(ctx, (TEXT), (const int32_t[]){__VA_ARGS__, -1})

static
void test(MLIS_Ctx* ctx, const char* text, const int32_t* expected)
{
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

	TEST("a dog jumping", 320, 1929, 11476);
	TEST("an illustration", 550, 6052);  //merge important
	TEST("2025", 17, 15, 17, 276);  //number
	//TEST("coraz\xc3\xb3n", 851, 854, 13926);  //utf8
	TEST("Stable Diffusion is a deep learning, text-to-image model released in 2022 based on diffusion techniques.", 10492, 18656, 9364, 533, 320, 3383, 2378, 267, 4160, 268, 531, 268, 2867, 2863, 3410, 530, 17, 15, 17, 273, 2812, 525, 18656, 9364, 1782, 697, 7715, 269);  //long, split words

	log("done");
	mlis_ctx_destroy(&ctx);
	return 0;
}
