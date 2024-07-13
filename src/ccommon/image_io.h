/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 *
 * Interface to read and write image and video data in multiple formats.
 */
#pragma once
#include "image.h"
#include "stream.h"

enum ImageError {
	IMG_RESULT_OK						= 0,
	IMG_ERROR_UNKNOWN					= -0x301,
	IMG_ERROR_PARAMS					= -0x302,
	IMG_ERROR_OUT_OF_MEMORY				= -0x303,
	IMG_ERROR_FILE_OPEN					= -0x304,
	IMG_ERROR_READ						= -0x305,
	IMG_ERROR_UNKNOWN_CODEC				= -0x306,
	IMG_ERROR_UNSUPPORTED_FUNCTION		= -0x307,
	IMG_ERROR_UNSUPPORTED_FORMAT		= -0x308,
	IMG_ERROR_UNSUPPORTED_PARAM			= -0x309,
	IMG_ERROR_UNSUPPORTED_INPUT_TYPE	= -0x30a,
	IMG_ERROR_LOAD						= -0x30b,
	IMG_ERROR_SAVE						= -0x30c,
	IMG_ERROR_INVALID_IMAGE				= -0x30d,
	IMG_ERROR_SEEK						= -0x30e,
	IMG_ERROR_EOF						= -0x30f,
	IMG_ERROR_AGAIN						= -0x310,	//try again later
	IMG_ERROR_UNSUPPORTED_VALUE			= -0x311,
};

/*
	Codec
*/
struct ImageIO;

enum ImageSeekMode {
	IMG_SEEK_SET = 0,
	IMG_SEEK_CUR = 1,
	IMG_SEEK_END = 2,
};

enum ImageCodecFlag {
	IMG_CODEC_F_ACCEPT_STREAM	= 1,
	IMG_CODEC_F_ACCEPT_FILENAME	= 2,
	IMG_CODEC_F_TRY_DETECT		= 4,
};

typedef struct {
	int (*op)(void*, struct ImageIO*, Image*);
	int flags;
	unsigned obj_size;
	int (*init)(void*, struct ImageIO*);
	void (*free)(void*, struct ImageIO*);
	int (*seek)(void*, struct ImageIO*, long, int);
	int (*value_get)(void*, struct ImageIO*, int, void*, unsigned);
	int (*value_set)(void*, struct ImageIO*, int, const void*, unsigned);
} ImageCodecSub;

typedef struct {
	bool (*detect)(Stream*, const char*);
	ImageCodecSub load;
	ImageCodecSub save;
	const char* name;
	const char* ext;
} ImageCodec;

int img_codec_register(const ImageCodec* codec);

const ImageCodec* img_codec_detect_stream(Stream* s);
const ImageCodec* img_codec_detect_ext(const char* ext, int oflags);
const ImageCodec* img_codec_detect_filename(const char* filename, int oflags);
const ImageCodec* img_codec_by_name(const char* name);

/*
	Image I/O
*/

enum ImageIOFlag {
	IMGIO_F_OWN_STREAM		= 1,
	IMGIO_F_OWN_INTERNAL	= 2,
	IMGIO_F_END_FOUND		= 4,
};

enum ImageIOOpenFlag {
	//IMG_OF_NO_INIT		= 1,
	IMG_OF_SAVE			= 2,
	IMG_OF_FAST			= 4,
	IMG_OF_GRAY			= 8,
	IMG_OF_NO_ALPHA		= 16,
	IMG_OF_ASYNC		= 32,	//asynchronous operation
};

typedef struct ImageIO {
	const ImageCodec *	codec;
	Stream *			s;
	const char *		filename;
	void *				internal;	//codec data
	int					oflags;
	int					flags;
} ImageIO;

void imgio_free(ImageIO* obj);

/**
Check if the image i/o object is ready to be used.
*/
static inline
bool imgio_good(ImageIO* obj) { return obj->codec; }

int imgio_open_stream(ImageIO* obj, Stream* s, int flags,
	const ImageCodec* codec);

int imgio_open_filename(ImageIO* obj, const char* fname, int flags,
	const ImageCodec* codec);

#define IMGIO_CODEC_CALL(NAME, ...) \
	if (!obj->codec) return IMG_ERROR_UNKNOWN_CODEC; \
	const ImageCodecSub* cs = \
		(obj->oflags & IMG_OF_SAVE) ? &obj->codec->save : &obj->codec->load; \
	if (!cs->NAME) return IMG_ERROR_UNSUPPORTED_FUNCTION; \
	return cs->NAME(obj->internal, obj, __VA_ARGS__);

static inline
int imgio_load(ImageIO* obj, Image* img) {
	if (obj->oflags & IMG_OF_SAVE) return IMG_ERROR_UNSUPPORTED_FUNCTION;
	IMGIO_CODEC_CALL(op, img)
}

static inline
int imgio_save(ImageIO* obj, const Image* img) {
	if (~obj->oflags & IMG_OF_SAVE) return IMG_ERROR_UNSUPPORTED_FUNCTION;
	IMGIO_CODEC_CALL(op, (Image*)img)
}

static inline
int imgio_seek(ImageIO* obj, long offset, int mode) {
	IMGIO_CODEC_CALL(seek, offset, mode)
}

enum {
	//unsigned: 0 to 100: jpeg or similar quality (85=default)
	IMG_VALUE_QUALITY			= 1,
	//unsigned: 0 to 9: png/deflate or similar compression level (0=disable, 6=default)
	IMG_VALUE_COMPRESSION		= 2,

	//unsigned: frame number counting from 0
	IMG_VALUE_FRAME_IDX			= 3,
	//unsigned: total number of frames,
	// may be estimated until you reach the last frame
	IMG_VALUE_FRAME_COUNT		= 4,
	//double: default or estimated frame duration in seconds
	IMG_VALUE_FRAME_DURATION	= 5,
	//unsigned: accumulated number of non fatal errors that occurred
	// the meaning varies with the codec, normally is amount of frames that
	// could not be read and were skipped
	IMG_VALUE_ERROR_COUNT		= 6,
	//text:
	// for read: buf="tag\0" and set bufsz, returns value length
	// for writing: buf="tag\0value\0"
	// Use the tag "comment" for a generic comment.
	IMG_VALUE_METADATA			= 7,

	//none: prompts the codec to reload some external configuration
	IMG_VALUE_RELOAD			= 8,

	//double: camera exposure time in seconds
	IMG_VALUE_EXPOSURE			= 101,

	//double: camera gain (1.0 normal)
	IMG_VALUE_GAIN				= 102,

	//ImgRectS: camera AOI (crop rectangle)
	IMG_VALUE_AOI				= 103,

	IMG_VALUE_CUSTOM			= 0x8000,
};

static inline
int imgio_value_get(ImageIO* obj, int id, void* buf, unsigned bufsz) {
	IMGIO_CODEC_CALL(value_get, id, buf, bufsz)
}

static inline
int imgio_value_set(ImageIO* obj, int id, const void* buf, unsigned bufsz) {
	IMGIO_CODEC_CALL(value_set, id, buf, bufsz)
}

/*
	Simplified image file I/O
*/
int img_load_file(Image* img, const char* filename);
int img_save_file(const Image* img, const char* filename);

/*
	Simplified codec registration
*/
#define IMGIO_CODEC_REGISTER_NODEP() do { \
	extern const ImageCodec img_codec_pnm;\
	img_codec_register(&img_codec_pnm); \
	extern const ImageCodec img_codec_imgseq; \
	img_codec_register(&img_codec_imgseq); \
} while (0)

#define IMGIO_CODEC_REGISTER_BASIC() do { \
	IMGIO_CODEC_REGISTER_NODEP(); \
	extern const ImageCodec img_codec_jpeg; \
	img_codec_register(&img_codec_jpeg); \
	extern const ImageCodec img_codec_png; \
	img_codec_register(&img_codec_png); \
} while (0)

#define IMGIO_CODEC_REGISTER_ALL() do { \
	IMGIO_CODEC_REGISTER_BASIC(); \
	extern const ImageCodec img_codec_tiff; \
	img_codec_register(&img_codec_tiff); \
	extern const ImageCodec img_codec_bigtiff; \
	img_codec_register(&img_codec_bigtiff); \
	extern const ImageCodec img_codec_libtiff; \
	img_codec_register(&img_codec_libtiff); \
	extern const ImageCodec img_codec_avimjpg; \
	img_codec_register(&img_codec_avimjpg); \
	extern const ImageCodec img_codec_libav; \
	img_codec_register(&img_codec_libav); \
	extern const ImageCodec img_codec_test; \
	img_codec_register(&img_codec_test); \
} while (0)	
