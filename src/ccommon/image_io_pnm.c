/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#include <stdlib.h>
#include "image_io.h"

static inline int whitespace_is(char c) {
	return (c == ' ' || c == '\t' || c == '\r' || c == '\n');
}

/*
	Type detect
*/
bool imgio_pnm_detect(Stream* s, const char* fileext)
{
	if (s) {
		const unsigned char *c = s->cursor;
		if (c[0] == 'P' && ('1' <= c[1] && c[1] <= '6') &&
			whitespace_is(c[2]))
			return true;
	}
	else if (fileext) {
		if (fileext[0] == 'p' &&
			(fileext[1] == 'n' || fileext[1] == 'p' || fileext[1] == 'g' ||
				fileext[1] == 'b') &&
			fileext[2] == 'm')
			return true;
	}
	return false;
}

/*
	Read
*/

static inline
char* field_next(char* cur, char* end)
{
	while (cur<end && !whitespace_is(*cur)) cur++;
	while (cur<end && whitespace_is(*cur)) cur++;
	return cur;
}

int imgio_pnm_load(void* self, ImageIO* imgio, Image* img)
{
	int r=0;

	if (stream_read_prep(imgio->s, 0) < 8)
		return IMG_ERROR_LOAD;

	// Read file header
	char *end, *cur=stream_buffer_get(imgio->s, &end);

	if (*cur++ != 'P') {
		r = IMG_ERROR_LOAD;
		goto error;
	}

	int bypp=0;
	ImgFormat format = IMG_FORMAT_NULL;
	switch (*cur++) {
	case '5':
		format = IMG_FORMAT_GRAY;
		bypp = 1;
		break;
	case '6':
		format = IMG_FORMAT_RGB;
		bypp = 3;
		break;
	default:
		r = IMG_ERROR_UNSUPPORTED_FORMAT;
		goto error;
	}

	int width = atoi( (cur = field_next(cur, end)) );
	int height = atoi( (cur = field_next(cur, end)) );
	int depth = atoi( (cur = field_next(cur, end)) );

	if (width < 1 || height < 1 || depth < 1) {
		r = IMG_ERROR_LOAD;
		goto error;
	}
	if (depth != 255) {	//TODO
		r = IMG_ERROR_UNSUPPORTED_FORMAT;
		goto error;
	}

	cur = field_next(cur, end);
	stream_commit(imgio->s, cur);

	// Allocate image
	r = img_resize(img, width, height, format, 0);
	if (r)
		goto error;

	// Load binary data
	size_t line_size = img->w * bypp;
	unsigned char* imgcur = img->data;
	for (unsigned y=0; y<img->h; ++y) {
		if (stream_read(imgio->s, line_size, imgcur) != line_size) {
			r = IMG_ERROR_LOAD;
			goto error;
		}
		imgcur += img->pitch;
	}

	return 0;

error:
	return r;
}

/*
	Save
*/

int imgio_pnm_save(void* unused, ImageIO* imgio, Image* img)
{
	if (stream_write_prep(imgio->s, 0) < 8)
		return IMG_ERROR_SAVE;

	size_t line_size=0;
	switch (img->format) {
	case IMG_FORMAT_GRAY:
		line_size = img->w;
		stream_printf(imgio->s, "P5 %d %d 255\n", img->w, img->h);
		break;
	case IMG_FORMAT_RGB:
		line_size = img->w * 3;
		stream_printf(imgio->s, "P6 %d %d 255\n", img->w, img->h);
		break;
	case IMG_FORMAT_RGBA:
		line_size = img->w * 4;
		//http://netpbm.sourceforge.net/doc/pam.html
		stream_printf(imgio->s,
			"P7\nWIDTH %d\nHEIGHT %d\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n",
			img->w, img->h);
		break;
	default:
		return IMG_ERROR_UNSUPPORTED_FORMAT;
	}

	unsigned char* imgcur = img->data;
	for (unsigned y=0; y<img->h; ++y) {
		if (stream_write(imgio->s, line_size, imgcur) != line_size)
			return IMG_ERROR_SAVE;
		imgcur += img->pitch;
	}

	return IMG_RESULT_OK;
}

/*
	Codec
*/
const ImageCodec img_codec_pnm = {
	imgio_pnm_detect,
	{
		imgio_pnm_load,
		IMG_CODEC_F_ACCEPT_STREAM,
	},
	{
		imgio_pnm_save,
		IMG_CODEC_F_ACCEPT_STREAM,
	},
	"PNM", "pnm"
};
