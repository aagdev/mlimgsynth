/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#include "image.h"
#include "ccommon.h"
#include "alloc.h"
#include <stdlib.h>
#include <string.h>

#ifndef IMAGE_DEFAULT_ALIGNMENT
#define IMAGE_DEFAULT_ALIGNMENT 64
#endif

#ifndef IMAGE_ALLOCATOR
#define IMAGE_ALLOCATOR  g_allocator
#endif

/*
Color
HSV <-> RGB
ref.: http://code.google.com/p/streumix-frei0r-goodies/wiki/Integer_based_RGB_HSV_conversion
HSV2RGB(RGB2HSV( )) = identity (0 errors verified)
*/
#define HSV_ABITS  IMG_HSV_ABITS
#define HSV_SSCALE (255 << HSV_ABITS)
#define HSV_HSCALE (256 << HSV_ABITS)	//hue scale

ImgColor img_color_hsv2rgb(const ImgColorHSV hsv)
{
	const int round_sum = 1 << (HSV_ABITS - 1);
	int a = hsv.a >> HSV_ABITS;

	int v = hsv.v >> HSV_ABITS;
	if (hsv.s == 0)
		return (ImgColor){v, v, v, a};

	const int region = 6 * hsv.h / HSV_HSCALE;	// h/60

	int m = hsv.v * (HSV_SSCALE - hsv.s) / HSV_SSCALE;
	int x = (hsv.v * hsv.s/HSV_HSCALE)
		* (HSV_HSCALE
			- abs(6 * hsv.h - 2 * (region >> 1) * HSV_HSCALE - HSV_HSCALE));

	x = ((x + hsv.v * (HSV_SSCALE - hsv.s)) / HSV_SSCALE + round_sum) >> HSV_ABITS;
	m = m >> HSV_ABITS;

	switch (region) {
		case 0:		return (ImgColor){v, x, m, a};
		case 1:		return (ImgColor){x, v, m, a};
		case 2:		return (ImgColor){m, v, x, a};
		case 3:		return (ImgColor){m, x, v, a};
		case 4:		return (ImgColor){x, m, v, a};
		default:	return (ImgColor){v, m, x, a};
	}
}

ImgColorHSV img_color_rgb2hsv(const ImgColor rgb)
{
	const int rgb_min = MIN3(rgb.r, rgb.g, rgb.b);
	const int rgb_max = MAX3(rgb.r, rgb.g, rgb.b);
	const int chroma  = rgb_max - rgb_min;

	int a = rgb.a << HSV_ABITS;
	int v = rgb_max << HSV_ABITS;
	if (v == 0)
		return (ImgColorHSV){0, 0, v, a};

	int s = HSV_SSCALE * chroma / rgb_max;
	if (s == 0)
		return (ImgColorHSV){0, 0, v, a};

	int h;
	if (rgb_max == rgb.r) {
		h = HSV_HSCALE * (6*chroma + rgb.g - rgb.b) / (6*chroma);
		if (h > HSV_HSCALE) h -= HSV_HSCALE;
	} else if (rgb_max == rgb.g)
		h = HSV_HSCALE * (2*chroma + rgb.b - rgb.r) / (6*chroma);
	else
		h = HSV_HSCALE * (4*chroma + rgb.r - rgb.g) / (6*chroma);

	return (ImgColorHSV){h, s, v, a};
}

/*
	Image
*/
void img_free(Image* img)
{
	if (img->data && img->flags & IMG_F_OWN_MEM)
		alloc_free(IMAGE_ALLOCATOR, img->data);
	
	*img = (Image){0};
}

int img_resize(Image* img, unsigned w, unsigned h, ImgFormat fmt,
	unsigned pitch)
{
	if (img->w == w && img->h == h && img->format == fmt &&
			(!pitch || img->pitch == pitch) && img->data)
		return 0;

	if (img->data && !(img->flags & IMG_F_OWN_MEM))
		return -1;

	unsigned bypp=0;
	switch (fmt) {
	case IMG_FORMAT_NULL:	bypp = 0;	break;
	case IMG_FORMAT_GRAY:	bypp = 1;	break;
	case IMG_FORMAT_RGB:	bypp = 3;	break;
	case IMG_FORMAT_RGBA:	bypp = 4;	break;
	default:
		return -1;//IMG_ERROR_UNSUPPORTED_PARAM;
	}

	if (!pitch) {
		const unsigned a = IMAGE_DEFAULT_ALIGNMENT;
		pitch = (w * bypp + a-1) / a * a;
	}
	else if (pitch < w * bypp)
		return -1;//IMG_ERROR_PARAMS;

	size_t sz = h * pitch;
	void* p = img->data;
	if (sz > 0) {
		p = alloc_realloc(IMAGE_ALLOCATOR, p, sz);
		if (!p) return -1;//IMG_ERROR_OUT_OF_MEMORY;
	}

	img->data = p;
	img->w = w;
	img->h = h;
	img->pitch = pitch;
	img->bypp = bypp;
	img->format = fmt;
	img->flags |= IMG_F_OWN_MEM;

	return 0;
}

int img_copy(Image* dst, const Image* src)
{
	int r = img_resize(dst, src->w, src->h, src->format, src->pitch);
	if (r < 0) return r;
	memcpy(dst->data, src->data, dst->h * dst->pitch);
	return 0;
}

void img_view_make(Image* dst, const Image* src, ImgRect rect)
{
	img_free(dst);

	if (rect.x < 0) { rect.w += rect.x; rect.x = 0; }
	if (rect.y < 0) { rect.h += rect.y; rect.y = 0; }

	rect.w = MAXg(MINg(rect.x + rect.w, (int)src->w) - rect.x, 0);
	rect.h = MAXg(MINg(rect.y + rect.h, (int)src->h) - rect.y, 0);

	if (rect.w < 0) rect.w = 0;
	if (rect.h < 0) rect.h = 0;

	dst->data = src->data + src->pitch * rect.y + src->bypp * rect.x;
	dst->w = rect.w;
	dst->h = rect.h;
	dst->pitch = src->pitch;
	dst->bypp = src->bypp;
	dst->format = src->format;
}

//TODO: macro the switch(img->format) and color set code?
void img_fill(Image* img, const ImgColor color)
{
	unsigned x, y;
	switch (img->format) {
	case IMG_FORMAT_GRAY: {
		unsigned char v = MAX3(color.r, color.g, color.b) * color.a / 255;
		for (y=0; y<img->h; ++y) {
			unsigned char* drow = img->data + img->pitch * y;
			for (x=0; x<img->w; ++x, drow+=img->bypp) {
				*drow = v;
			}
		}
		} break;
	case IMG_FORMAT_RGB:
		for (y=0; y<img->h; ++y) {
			unsigned char* drow = img->data + img->pitch * y;
			for (x=0; x<img->w; ++x, drow+=img->bypp) {
				*(drow)   = color.r;  //TODO: endianness
				*(drow+1) = color.g;
				*(drow+2) = color.b;
			}
		}
		break;
	case IMG_FORMAT_RGBA:
		for (y=0; y<img->h; ++y) {
			unsigned char* drow = img->data + img->pitch * y;
			for (x=0; x<img->w; ++x, drow+=img->bypp) {
				*(drow)   = color.r;  //TODO: endianness
				*(drow+1) = color.g;
				*(drow+2) = color.b;
				*(drow+3) = color.a;
			}
		}
		break;
	default:
		break;
	}
}
