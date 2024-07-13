/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#define IMGIO_JPEG_IMPL
#include "image_io_jpeg.h"
#include "logging.h"

//#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

//#include "logging.h"
//#define DebugLog(...) log_info(__VA_ARGS__)
#define DebugLog(...) {}

/*
	Type detect
*/
bool imgio_jpeg_detect(Stream* s, const char* fileext)
{
	if (s) {
		const unsigned char *c = s->cursor;
		if (c[0] == 0xff && c[1] == 0xd8 && c[2] == 0xff)
			return true;
	}
	else if (fileext) {
		if (!strcmp(fileext, "jpeg") || !strcmp(fileext, "jpg"))
			return true;
	}
	return false;
}

/*
	libjpeg source manager
*/
typedef struct {
	struct jpeg_source_mgr pub;
	Stream* s;
	void* buf_end;
} stream_src_mgr;

// Called before any read is done
static void init_source(j_decompress_ptr cinfo) {}

// Called whenever the buffer is emptied
// The data still in the buffer is discarded
static boolean fill_input_buffer(j_decompress_ptr cinfo)
{
	stream_src_mgr* src = (stream_src_mgr *)cinfo->src;

//	src->buf_end = (void*)src->pub.next_input_byte;
	if (src->buf_end) stream_commit(src->s, src->buf_end);
	long r = stream_read_prep(src->s, 0);
	if (r < 0) return FALSE;
	src->pub.bytes_in_buffer = r;
	src->pub.next_input_byte = stream_buffer_get(src->s, &src->buf_end);

	DebugLog("fill_input_buffer %d", (int)src->pub.bytes_in_buffer);
#ifdef DEBUG
//	FILE* f = fopen("tmp.bin", "a");
//	fwrite(src->pub.next_input_byte, src->pub.bytes_in_buffer, 1, f);
//	fclose(f);
#endif

    return TRUE;
}

// Skips large amounts of useless data
static void skip_input_data(j_decompress_ptr cinfo, long num_bytes)
{
	stream_src_mgr* src = (stream_src_mgr *)cinfo->src;

	DebugLog("skip_input_data %ld", num_bytes);

	//TODO: seek
	while (num_bytes > src->pub.bytes_in_buffer) {
		num_bytes -= src->pub.bytes_in_buffer;
		fill_input_buffer(cinfo);
	}
	src->pub.bytes_in_buffer -= num_bytes;
	src->pub.next_input_byte += num_bytes;
}

// Called after all data has been read
static void term_source(j_decompress_ptr cinfo)
{
	stream_src_mgr* src = (stream_src_mgr *)cinfo->src;

	DebugLog("term_source %d", (int)src->pub.bytes_in_buffer);

	src->buf_end = (void*)src->pub.next_input_byte;
	stream_commit(src->s, src->buf_end);
}

static void jpeg_stream_src(j_decompress_ptr cinfo, Stream* s)
{
	stream_src_mgr* src;

	if (cinfo->src == NULL) { /* first time for this JPEG object? */
		cinfo->src = (struct jpeg_source_mgr *) (*cinfo->mem->alloc_small)(
				(j_common_ptr)cinfo, JPOOL_PERMANENT, sizeof(stream_src_mgr));
	}

	src = (stream_src_mgr *) cinfo->src;

	src->pub.init_source 		= init_source;
	src->pub.fill_input_buffer	= fill_input_buffer;
	src->pub.skip_input_data	= skip_input_data;
	src->pub.resync_to_restart	= jpeg_resync_to_restart;
	src->pub.term_source		= term_source;

	src->s = s;
	src->buf_end = NULL;
	src->pub.bytes_in_buffer = 0; /* forces fill_input_buffer on first read */
	src->pub.next_input_byte = NULL; /* until buffer loaded */
}

/*
	libjpeg destination manager
*/
typedef struct {
	struct jpeg_destination_mgr pub;
	Stream * s;
	unsigned char *cur, *end;  //write buffer
} stream_dst_mgr;

static void init_destination(j_compress_ptr cinfo) {}

// Writes the entire buffer
static boolean empty_output_buffer(j_compress_ptr cinfo)
{
    stream_dst_mgr * dest = (stream_dst_mgr *)cinfo->dest;

	/* Warning: the values in next_output_byte and free_in_buffer are
	 * misleading, we must write all the buffer */
	stream_commit(dest->s, dest->end);
	if (stream_write_prep(dest->s, 0) < 1)
		return FALSE;
	
	dest->cur = stream_buffer_get(dest->s, &dest->end);
	dest->pub.next_output_byte	= (JOCTET*)dest->cur;
	dest->pub.free_in_buffer	= dest->end - dest->cur;

    return TRUE;
}

static void term_destination(j_compress_ptr cinfo)
{
	stream_dst_mgr * dest = (stream_dst_mgr *)cinfo->dest;
	
	stream_commit(dest->s, dest->end - dest->pub.free_in_buffer);
	if (stream_write_prep(dest->s, 0) < 1)
		return;
}

static int jpeg_stream_dest(j_compress_ptr cinfo, Stream* s)
{
	stream_dst_mgr* dest;

	if (stream_write_prep(s, 0) < 0) return -1;

	if (cinfo->dest == NULL) {
		cinfo->dest = (struct jpeg_destination_mgr *)
			(*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_PERMANENT,
				sizeof(stream_dst_mgr));
		if (!cinfo->dest) return -1;
	}

	dest = (stream_dst_mgr *)cinfo->dest;

	dest->pub.init_destination		= init_destination;
	dest->pub.empty_output_buffer	= empty_output_buffer;
	dest->pub.term_destination		= term_destination;

	dest->s = s;
	dest->cur = stream_buffer_get(dest->s, &dest->end);
	dest->pub.next_output_byte	= (JOCTET*)dest->cur;
	dest->pub.free_in_buffer	= dest->end - dest->cur;
	
	return 1;
}

/*
	libjpeg error manager
*/

static void my_jpeg_error_exit(j_common_ptr cinfo)
{
	struct img_codec_jpeg_error_mgr *err =
		(struct img_codec_jpeg_error_mgr *)cinfo->err;
	longjmp(err->escape, 1);
}

static void my_jpeg_output_message(j_common_ptr cinfo)
{
	char buffer[JMSG_LENGTH_MAX]="";
	cinfo->err->format_message(cinfo, buffer);
	log_error("%s", buffer);
}

/*
	Codec init & free
*/

int imgio_jpeg_load_init(CodecJpegLoad* codec, ImageIO* imgio)
{
	*codec = (CodecJpegLoad){0};

	codec->cinfo.err = jpeg_std_error(&codec->jerr.errmgr);
	codec->jerr.errmgr.error_exit = my_jpeg_error_exit;
	codec->jerr.errmgr.output_message = my_jpeg_output_message;

	jpeg_create_decompress(&codec->cinfo);

	return 0;
}

int imgio_jpeg_save_init(CodecJpegSave* codec, ImageIO* imgio)
{
	*codec = (CodecJpegSave){0};

	codec->cinfo.err = jpeg_std_error(&codec->jerr.errmgr);
	codec->jerr.errmgr.error_exit = my_jpeg_error_exit;
	codec->jerr.errmgr.output_message = my_jpeg_output_message;

	jpeg_create_compress(&codec->cinfo);

	codec->quality = 85;
	return 0;
}

void imgio_jpeg_load_free(CodecJpegLoad* codec, ImageIO* imgio)
{
	jpeg_destroy_decompress(&codec->cinfo);
}

void imgio_jpeg_save_free(CodecJpegSave* S, ImageIO* imgio)
{
	vec_for(S->metadata,i,0) {
		dstr_free(S->metadata[i].value);
		dstr_free(S->metadata[i].key);
	}
	vec_free(S->metadata);

	jpeg_destroy_compress(&S->cinfo);
}

/*
	Load
*/
int imgio_jpeg_load_op(CodecJpegLoad* codec, ImageIO* imgio, Image* img)
{
	struct jpeg_decompress_struct * cinfo = &codec->cinfo;

	if (setjmp(codec->jerr.escape)) {
		/* If we get here, libjpeg found an error */
		return IMG_ERROR_LOAD;
	}
	jpeg_stream_src(cinfo, imgio->s);
	jpeg_read_header(cinfo, TRUE);

	ImgFormat format = IMG_FORMAT_NULL;
	if (cinfo->out_color_space == JCS_GRAYSCALE ||
		imgio->oflags & IMG_OF_GRAY)
	{
		cinfo->out_color_space = JCS_GRAYSCALE;
		format = IMG_FORMAT_GRAY;
	}
	else {
		cinfo->out_color_space = JCS_RGB;
		format = IMG_FORMAT_RGB;
	}
	cinfo->quantize_colors = FALSE;

	if (imgio->oflags & IMG_OF_FAST) {
		cinfo->scale_num = 1;
		cinfo->scale_denom = 1;
		cinfo->dct_method = JDCT_FASTEST;
		cinfo->do_fancy_upsampling = FALSE;
	}

	jpeg_calc_output_dimensions(cinfo);

	int r = img_resize(img, cinfo->output_width, cinfo->output_height,
		format, cinfo->output_width * cinfo->output_components);
	if (r) return r;

	jpeg_start_decompress(cinfo);

	JSAMPROW rowptr[1];
	while (cinfo->output_scanline < cinfo->output_height) {
		rowptr[0] = (JSAMPROW)img->data + img->pitch * cinfo->output_scanline;
		jpeg_read_scanlines(cinfo, rowptr, (JDIMENSION)1);
	}

	jpeg_finish_decompress(cinfo);

	return IMG_RESULT_OK;
}

/*
	Save
*/
int imgio_jpeg_save_op(CodecJpegSave* codec, ImageIO* imgio, Image* img)
{
	int r=0;
	DynStr tmps=NULL;
	struct jpeg_compress_struct * cinfo = &codec->cinfo;

	if (setjmp(codec->jerr.escape)) {
		/* If we get here, libjpeg found an error */
		r = IMG_ERROR_SAVE; goto end;
	}
	if (jpeg_stream_dest(cinfo, imgio->s) < 0) { r=IMG_ERROR_SAVE; goto end; }

	cinfo->image_width = img->w;
	cinfo->image_height = img->h;

	switch (img->format) {
	case IMG_FORMAT_RGB:
		cinfo->in_color_space = JCS_RGB;
		cinfo->input_components = 3;
		break;
	case IMG_FORMAT_GRAY:
		cinfo->in_color_space = JCS_GRAYSCALE;
		cinfo->input_components = 1;
		break;
	default:
		r = IMG_ERROR_UNSUPPORTED_FORMAT; goto end;
	}

	jpeg_set_defaults(cinfo);
	jpeg_set_quality(cinfo, codec->quality, TRUE);
	jpeg_start_compress(cinfo, TRUE);

	vec_forp(struct CodecJpegText, codec->metadata, p, 0) {
		dstr_copyd(tmps, p->key);
		if (!dstr_empty(p->key)) dstr_push(tmps, '\n');
		dstr_appendd(tmps, p->value);
		jpeg_write_marker(cinfo, JPEG_COM, (const JOCTET*)tmps, dstr_count(tmps));
	}

	JSAMPROW rowptr[1];
	while (cinfo->next_scanline < cinfo->image_height) {
		rowptr[0] = (JSAMPROW)img->data + cinfo->next_scanline * img->pitch;
		jpeg_write_scanlines(cinfo, rowptr, 1);
	}

	jpeg_finish_compress(cinfo);

	r = IMG_RESULT_OK;
end:
	dstr_free(tmps);
	return r;
}

int imgio_jpeg_save_value_set(CodecJpegSave* codec, ImageIO* imgio,
	int id, const void* buf, unsigned bufsz)
{
	switch (id) { 
	case IMG_VALUE_QUALITY:
		codec->quality = *((int*)buf);
		break;
	case IMG_VALUE_METADATA: {
		const char *key=buf, *value=buf;
		while (*value++);  //skip first zero-terminated string
		vec_append_zero(codec->metadata, 1);
		dstr_copyz(vec_last(codec->metadata,0).key, key);
		dstr_copyz(vec_last(codec->metadata,0).value, value);
	} break;
	default:
		return IMG_ERROR_UNSUPPORTED_VALUE;
	}
	return 0;
}

/*
	Codec
*/
const ImageCodec img_codec_jpeg = {
	imgio_jpeg_detect,
	{
		(int (*)(void*, ImageIO*, Image*)) imgio_jpeg_load_op,
		IMG_CODEC_F_ACCEPT_STREAM,
		sizeof(CodecJpegLoad),
		(int (*)(void*, ImageIO*)) imgio_jpeg_load_init,
		(void (*)(void*, ImageIO*)) imgio_jpeg_load_free,
	},
	{
		(int (*)(void*, ImageIO*, Image*)) imgio_jpeg_save_op,
		IMG_CODEC_F_ACCEPT_STREAM,
		sizeof(CodecJpegSave),
		(int (*)(void*, ImageIO*)) imgio_jpeg_save_init,
		(void (*)(void*, ImageIO*)) imgio_jpeg_save_free,
		NULL, //seek
		NULL, //value_get
		(int (*)(void*, ImageIO*, int, const void*, unsigned))
			imgio_jpeg_save_value_set
	},
	"JPEG", "jpeg"
};

