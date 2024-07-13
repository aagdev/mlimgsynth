/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#include "image_io.h"
#include "logging.h"
#include "str_util.h"
#include "alloc.h"
#include <assert.h>

#ifndef IMAGE_IO_ALLOCATOR
#define IMAGE_IO_ALLOCATOR  g_allocator
#endif

/*
	Codecs
*/

#define MAX_CODECS 31
const ImageCodec * imgio_codecs[MAX_CODECS+1] = {
	NULL
};

int img_codec_register(const ImageCodec* codec)
{
	int i;
	for (i=0; imgio_codecs[i]; ++i)
		if (imgio_codecs[i] == codec)
			return 0;

	if (i >= MAX_CODECS)
		return -1;

	imgio_codecs[i] = codec;
	return 1;
}

const ImageCodec* img_codec_detect_stream(Stream* s)
{
	if (stream_read_prep(s,0) < 8)
		return 0;

	for (int i=0; imgio_codecs[i]; ++i)
		if (imgio_codecs[i]->detect &&
			imgio_codecs[i]->load.op &&
			imgio_codecs[i]->detect(s, 0))
			return imgio_codecs[i];

	return 0;
}

const ImageCodec* img_codec_detect_ext(const char* ext, int oflags)
{
	char buffer[8];
	str_tolower(buffer, sizeof(buffer), ext);

	const bool save = oflags & IMG_OF_SAVE;
	for (int i=0; imgio_codecs[i]; ++i)
		if (imgio_codecs[i]->detect &&
			(( save && imgio_codecs[i]->save.op) ||
			 (!save && imgio_codecs[i]->load.op)) &&
			imgio_codecs[i]->detect(0, buffer) )
			return imgio_codecs[i];

	return 0;
}

const ImageCodec* img_codec_detect_filename(const char* filename, int oflags)
{
	const char* ext = strrchr(filename, '.');
	if (!ext) return 0;
	ext++;

	return img_codec_detect_ext(ext, oflags);
}

const ImageCodec* img_codec_by_name(const char* name)
{
	for (int i=0; imgio_codecs[i]; ++i) {
		if (!imgio_codecs[i]->name) continue;
		if (!str_cmp_i(imgio_codecs[i]->name, name))
			return imgio_codecs[i];
	}
	return 0;
}

/*
	Image I/O
*/

int imgio_stream_alloc(ImageIO* obj)
{
	if (obj->s)
		return IMG_ERROR_PARAMS;
	if (!obj->filename)
		return IMG_ERROR_UNSUPPORTED_INPUT_TYPE;

	Stream * p = alloc_new(IMAGE_IO_ALLOCATOR, Stream, 1);

	if (stream_open_file(p, obj->filename,
		(obj->oflags & IMG_OF_SAVE) ? SOF_CREATE : SOF_READ) < 0)
	{
		alloc_free(IMAGE_IO_ALLOCATOR, p);
		return IMG_ERROR_FILE_OPEN;
	}

	obj->s = p;
	obj->flags |= IMGIO_F_OWN_STREAM;

	return 0;
}

int imgio_codec_detect(ImageIO* obj)
{
	if (obj->oflags & IMG_OF_SAVE) {
		if (obj->filename)
			obj->codec = img_codec_detect_filename(obj->filename, obj->oflags);
	}
	else {
		if (!obj->s) {
			imgio_stream_alloc(obj);
			// An error here can be ok,
			// for example some LibAV URL are not files
		}
		if (obj->s)
			obj->codec = img_codec_detect_stream(obj->s);
	}

	if (!obj->codec)
		return IMG_ERROR_UNKNOWN_CODEC;

	return 0;
}

int imgio_open_inner(ImageIO* obj)
{
	assert(obj->codec);

	const ImageCodecSub* cs =
		(obj->oflags & IMG_OF_SAVE) ? &obj->codec->save : &obj->codec->load;

	if (obj->s && cs->flags & IMG_CODEC_F_ACCEPT_STREAM) {
	}
	else if (obj->filename && cs->flags & IMG_CODEC_F_ACCEPT_FILENAME) {
	}
	else if (obj->filename && cs->flags & IMG_CODEC_F_ACCEPT_STREAM) {
		int r = imgio_stream_alloc(obj);
		if (r) return r;
	}
	else
		return IMG_ERROR_UNSUPPORTED_INPUT_TYPE;

	// Codec alloc
	if (cs->obj_size) {
		obj->internal = alloc_realloc(IMAGE_IO_ALLOCATOR, obj->internal, cs->obj_size);
		obj->flags |= IMGIO_F_OWN_INTERNAL;
	}

	if (cs->init) {
		int r = cs->init(obj->internal, obj);
		if (r) return r;
	}

	return 0;
}

int imgio_open(ImageIO* obj)
{
	int r=0;

	if (!obj->codec) {
		r = imgio_codec_detect(obj);
		if (r && r != IMG_ERROR_UNKNOWN_CODEC)
			return r;
	}

	if (obj->codec) {
		r = imgio_open_inner(obj);
	}
	else {
		// Test all codecs without detection
		for (int i=0; imgio_codecs[i]; ++i) {
			if (imgio_codecs[i]->detect ||
				(obj->oflags & IMG_OF_SAVE &&
				 (!imgio_codecs[i]->save.op ||
				  ~imgio_codecs[i]->save.flags & IMG_CODEC_F_TRY_DETECT) ) ||
				(~obj->oflags & IMG_OF_SAVE &&
				 (!imgio_codecs[i]->load.op ||
				  ~imgio_codecs[i]->load.flags & IMG_CODEC_F_TRY_DETECT) )
				)
				continue;
			obj->codec = imgio_codecs[i];
			r = imgio_open_inner(obj);
			if (!r)
				break;
		}
		if (r)
			obj->codec = 0;
	}

	obj->filename = 0;	// This pointer may not be safe

	if (r) {
		obj->codec = 0;
		imgio_free(obj);
	}

	return r;
}

#define imgio_open_BEGIN \
	imgio_free(obj);

int imgio_open_stream(ImageIO* obj, Stream* s, int flags,
	const ImageCodec* codec)
{
	imgio_open_BEGIN;
	obj->s = s;
	obj->filename = 0;
	obj->codec = codec;
	obj->oflags = flags;
	return imgio_open(obj);
}

int imgio_open_filename(ImageIO* obj, const char* fname, int flags,
	const ImageCodec* codec)
{
	imgio_open_BEGIN;
	obj->s = 0;
	obj->filename = fname;
	obj->codec = codec;
	obj->oflags = flags;
	return imgio_open(obj);
}

void imgio_free(ImageIO* obj)
{
	if (obj->codec) {
		const ImageCodecSub* cs =
			(obj->oflags & IMG_OF_SAVE) ? &obj->codec->save : &obj->codec->load;
		if (cs->free)
			cs->free(obj->internal, obj);
		obj->codec = 0;
	}
	if (obj->flags & IMGIO_F_OWN_INTERNAL && obj->internal) {
		alloc_free(IMAGE_IO_ALLOCATOR, obj->internal);
		obj->internal = 0;
	}
	if (obj->flags & IMGIO_F_OWN_STREAM && obj->s) {
		stream_close(obj->s, 0);
		alloc_free(IMAGE_IO_ALLOCATOR, obj->s);
		obj->s = 0;
	}
	obj->flags = 0;
}

int img_load_file(Image* img, const char* filename)
{
	int r=0;

	Stream s={0};
	if (stream_open_file(&s, filename, SOF_READ) < 0)
		return IMG_ERROR_FILE_OPEN;

	ImageIO imgio={0};
	r = imgio_open_stream(&imgio, &s, 0, 0);
	if (r) goto end;

	r = imgio_load(&imgio, img);

end:
	imgio_free(&imgio);
	stream_close(&s, 0);
	return r;
}

int img_save_file(const Image* img, const char* filename)
{
	int r=0;

	const ImageCodec* codec = img_codec_detect_filename(filename, IMG_OF_SAVE);
	if (!codec) return IMG_ERROR_UNKNOWN_CODEC;

	Stream s={0};
	if (stream_open_file(&s, filename, SOF_CREATE) < 0)
		return IMG_ERROR_FILE_OPEN;

	ImageIO imgio={0};
	r = imgio_open_stream(&imgio, &s, IMG_OF_SAVE, codec);
	if (r) goto end;

	r = imgio_save(&imgio, img);

end:
	imgio_free(&imgio);
	stream_close(&s, 0);
	return r;
}
