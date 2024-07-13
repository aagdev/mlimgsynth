/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#define IMGIO_PNG_IMPL
#include "image_io_png.h"
#include "logging.h"
#include <png.h>

//#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>

#ifndef IMAGE_IO_ALLOCATOR
#define IMAGE_IO_ALLOCATOR  g_allocator
#endif

/*
	Type detect
*/
bool imgio_png_detect(Stream* s, const char* fileext)
{
	if (s) {
		const unsigned char *c = s->cursor;
		if (c[0] == 0x89 && c[1] == 'P' && c[2] == 'N' && c[3] == 'G')
			return true;
	}
	else if (fileext) {
		if (!strcmp(fileext, "png"))
			return true;
	}
	return false;
}

/*
	Read
*/
static void png_read_data(png_structp ctx, png_bytep area, png_size_t size)
{
	Stream* s = (Stream*)png_get_io_ptr(ctx);
	stream_read(s, size, area);
}

int imgio_png_load(void* self, ImageIO* imgio, Image* img)
{
	png_structp png_ptr=0;
	png_infop info_ptr=0;
	png_bytep* row_pointers=0;
	int result = IMG_ERROR_UNKNOWN;

	png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL,NULL,NULL);
	if (!png_ptr) {
		result = IMG_ERROR_OUT_OF_MEMORY;
		goto end;
	}

	info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr) {
		result = IMG_ERROR_OUT_OF_MEMORY;
		goto end;
	}

	if (setjmp(png_jmpbuf(png_ptr)))
	{	/* We will be here in case of any error */
		result = IMG_ERROR_LOAD;
		goto end;
	}

	//png_set_error_fn(png_ptr, error_ptr, error_fn, warning_fn)

	// Setup input
	png_set_read_fn(png_ptr, imgio->s, png_read_data);

	// Read header
    png_uint_32 width, height;
    int bit_depth, color_type, interlace_type;
	png_read_info(png_ptr, info_ptr);
	png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth,
		&color_type, &interlace_type, NULL, NULL);

	//log_debug("png: color_type: %d, bit_depth: %d, interlace_type: %d",
	//	color_type, bit_depth, interlace_type);

	// Convert paletted images to full RGB
	if (color_type & PNG_COLOR_MASK_PALETTE) {
		png_set_palette_to_rgb(png_ptr);
		color_type &= ~PNG_COLOR_MASK_PALETTE;
	}

	// Convert transparency information to a full alpha channel
	if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS) &&
		~imgio->oflags & IMG_OF_NO_ALPHA)
	{
		png_set_tRNS_to_alpha(png_ptr);
		color_type |= PNG_COLOR_MASK_ALPHA;
	}

	// Convert less than one byte/color images to one byte/color
	if (bit_depth < 8) {
		if (~color_type & PNG_COLOR_MASK_COLOR) //gray
			png_set_expand_gray_1_2_4_to_8(png_ptr);
		else
			png_set_packing(png_ptr);
	}
	else if (bit_depth > 8) {
		// Convert 16 bit/color to 8 bit/color
		png_set_strip_16(png_ptr);
		// TODO: optionally, keep this
		// Big-endian to little-endian
		//png_set_swap(png_ptr);
	}

	if (color_type & PNG_COLOR_MASK_ALPHA) {
		// Strip alpha channel ?
		if (imgio->oflags & IMG_OF_NO_ALPHA)
			png_set_strip_alpha(png_ptr);

		// Gray+Alpha -> Gray or RGBA
		else if (~color_type & PNG_COLOR_MASK_COLOR)
		{
			if (imgio->oflags & IMG_OF_GRAY)
				png_set_strip_alpha(png_ptr);
			else
				png_set_gray_to_rgb(png_ptr);
		}
	}
	// Convert color to grayscale (optional)
	else if (color_type & PNG_COLOR_MASK_COLOR && imgio->oflags & IMG_OF_GRAY)
	{
		png_set_rgb_to_gray(png_ptr, 1, -1, -1);
	}

	// Get values after the transformations
	png_read_update_info(png_ptr, info_ptr);
	png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth,
		&color_type, &interlace_type, NULL, NULL);

	//unsigned bytes_per_pixel = bit_depth/8;
	if (bit_depth != 8)
		return IMG_ERROR_UNSUPPORTED_FORMAT;

	ImgFormat format = IMG_FORMAT_NULL;
	switch (color_type) {
	case PNG_COLOR_TYPE_GRAY:
		//bytes_per_pixel *= 1;
		format = IMG_FORMAT_GRAY;
		break;
	case PNG_COLOR_TYPE_RGB:
		//bytes_per_pixel *= 3;
		format = IMG_FORMAT_RGB;
		break;
	case PNG_COLOR_TYPE_RGB_ALPHA:
		//bytes_per_pixel *= 4;
		format = IMG_FORMAT_RGBA;
		break;
	default:
		result = IMG_ERROR_UNSUPPORTED_FORMAT;
		goto end;
	}

	// Allocate image data buffer
	int r = img_resize(img, width, height, format, 0);
	if (r) return r;

	// Allocate and store pointers to each row
	row_pointers = malloc(sizeof(png_bytep)*height);
	if (!row_pointers) {
		result = IMG_ERROR_OUT_OF_MEMORY;
		goto end;
	}
	for (png_uint_32 i=0; i<height; ++i)
		row_pointers[i] = img->data + img->pitch * i;

	// Read the entire image in one go
	// This deals with the de-interlacing
	png_read_image(png_ptr, row_pointers);

	// Read extra data (i.e.: comments)
	//png_read_end(png_ptr, info_ptr);

	result = IMG_RESULT_OK;

end:
	if (row_pointers)
		free(row_pointers);
	if (png_ptr)
		png_destroy_read_struct(&png_ptr, info_ptr ? &info_ptr : NULL, NULL);

	return result;
}

/*
	Save
*/
static void png_write_data(png_structp png_ptr, png_bytep src, png_size_t size)
{
    Stream* s = (Stream*)png_get_io_ptr(png_ptr);
    stream_write(s, size, src);
}

static void png_flush_data(png_structp png_ptr)
{
    Stream* s = (Stream*)png_get_io_ptr(png_ptr);
	stream_flush(s);
}

int imgio_png_save_init(CodecPng* S, ImageIO* imgio)
{
	*S = (CodecPng){0};
	return 0;
}

void imgio_png_save_free(CodecPng* S, ImageIO* imgio)
{
	vec_for(S->metadata,i,0) {
		dstr_free(S->metadata[i].value);
		dstr_free(S->metadata[i].key);
	}
	vec_free(S->metadata);
}

int imgio_png_save_op(CodecPng* S, ImageIO* imgio, Image* img)
{
	int R = IMG_RESULT_OK;
	png_structp png_ptr=NULL;
	png_infop info_ptr=NULL;
	png_text *texts=NULL;
	png_bytep *row_pointers=NULL;

	png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL,NULL,NULL);
	if (!png_ptr) RETURN( IMG_ERROR_OUT_OF_MEMORY );
	//TODO: custom allocator?

	info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr) RETURN ( IMG_ERROR_OUT_OF_MEMORY );

    png_set_write_fn(png_ptr, imgio->s, png_write_data, png_flush_data);

	// Set up error handling
	if (setjmp(png_jmpbuf(png_ptr)))
	{	// libpng jumps here in case of error
		RETURN( IMG_ERROR_SAVE );
	}

	// Set up image parameters
	int bit_depth = 8;
	int color_type = 0;
	switch (img->format) {
	case IMG_FORMAT_GRAY:
		color_type = PNG_COLOR_TYPE_GRAY;
		break;
	case IMG_FORMAT_RGB:
		color_type = PNG_COLOR_TYPE_RGB;
		break;
	case IMG_FORMAT_RGBA:
		color_type = PNG_COLOR_TYPE_RGB_ALPHA;
		break;
	default:
		RETURN( IMG_ERROR_UNSUPPORTED_FORMAT );
	}

	png_set_IHDR(png_ptr, info_ptr, img->w, img->h,
		bit_depth, color_type, PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

	// Configure compression
	if (S->comp_lvl > 0)
		png_set_compression_level(png_ptr, S->comp_lvl);

	// Set meta data text
	unsigned ntext = vec_count(S->metadata);
	if (ntext) {
		texts = alloc_alloc(IMAGE_IO_ALLOCATOR, sizeof(png_text) * ntext);
		vec_for(S->metadata, i, 0) {
			texts[i] = (png_text){
				.compression = PNG_TEXT_COMPRESSION_NONE,
				.key  = S->metadata[i].key,
				.text = S->metadata[i].value,
				.text_length = dstr_count(S->metadata[i].value),
			};
		}
		png_set_text(png_ptr, info_ptr, texts, ntext);
	}

	// Allocate and store pointers to each row
	row_pointers = alloc_alloc(IMAGE_IO_ALLOCATOR, sizeof(png_bytep) * img->h);

	for (png_uint_32 i=0; i<img->h; ++i)
		row_pointers[i] = img->data + img->pitch * i;

	png_set_rows(png_ptr, info_ptr, row_pointers);

	png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

end:
	alloc_free(IMAGE_IO_ALLOCATOR, row_pointers);
	alloc_free(IMAGE_IO_ALLOCATOR, texts);
	if (png_ptr)
		png_destroy_write_struct(&png_ptr, info_ptr ? &info_ptr : NULL);
	return R;
}

int imgio_png_value_set(CodecPng* S, ImageIO* imgio,
	int id, const void* buf, unsigned bufsz)
{
	switch (id) { 
	case IMG_VALUE_COMPRESSION:
		S->comp_lvl = !!*((unsigned*)buf);
		break;
	case IMG_VALUE_METADATA: {
		const char *key=buf, *value=buf;
		while (*value++);  //skip first zero-terminated string
		vec_append_zero(S->metadata, 1);
		dstr_copyz(vec_last(S->metadata,0).key, key);
		dstr_copyz(vec_last(S->metadata,0).value, value);
	} break;
	default:
		return IMG_ERROR_UNSUPPORTED_VALUE;
	}
	return 0;
}

/*
	Codec
*/
const ImageCodec img_codec_png = {
	imgio_png_detect,
	{
		imgio_png_load,
		IMG_CODEC_F_ACCEPT_STREAM,
	},
	{
		(int (*)(void*, ImageIO*, Image*)) imgio_png_save_op,
		IMG_CODEC_F_ACCEPT_STREAM,
		sizeof(CodecPng),
		(int (*)(void*, ImageIO*)) imgio_png_save_init,
		(void (*)(void*, ImageIO*)) imgio_png_save_free,
		NULL,  //seek
		NULL,  //value_get
		(int (*)(void*, ImageIO*, int, const void*, unsigned))
			imgio_png_value_set,
	},
	"PNG", "png"
};

