/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 *
 * Inteface to store and manipulate images in memory.
 *
 * Example:
 *   Image img={0};
 *   TRY( img_resize(&img, 512, 256, IMG_FORMAT_RBG, 0) );
 *   img_fill(&img, (ImgColor){255,0,0});
 *   img_free(&img);
 */
#pragma once
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

//TODO: define error codes

/*
	Point
*/
typedef struct ImgPoint {
	int x, y;
} ImgPoint;

#define IMG_POINT_UNPACK(V)  (V).x, (V).y

/*
	Rect
*/
typedef struct ImgRectS {
	int x, y, w, h;
} ImgRectS;

typedef struct ImgRectP {
	int x1, y1, x2, y2;
} ImgRectP;

#define IMG_RECTS_UNPACK(V)  (V).x, (V).y, (V).w, (V).h
#define IMG_RECTP_UNPACK(V)  (V).x1, (V).y1, (V).x2, (V).y2

typedef ImgRectS ImgRect;
#define IMG_RECT_UNPACK IMG_RECTS_UNPACK
#define IMG_RECT_FMT "%d,%d:%dx%d"

#define IMG_RECT_P1(R)  (*((ImgPoint*)&(R)))
#define IMG_RECT_P2(R)  (*(((ImgPoint*)&(R))+1))

static inline
bool img_rect_inside_is(const ImgRect* r, const ImgPoint* p) {
	return	r->x <= p->x && p->x < r->x+r->w &&
			r->y <= p->y && p->y < r->y+r->h;
}

/*
	Color
*/
typedef struct ImgColor {
	uint8_t r, g, b, a;
} ImgColor;

#define IMG_COLOR_UNPACK_RGB(V)  (V).r, (V).g, (V).b
#define IMG_COLOR_UNPACK(V)  (V).r, (V).g, (V).b, (V).a

typedef enum ImgFormat {
#define IMG_FORMAT_F_COLOR  0x100
#define IMG_FORMAT_F_ALPHA  0x200
	IMG_FORMAT_NULL		= 0,
	IMG_FORMAT_GRAY		= 1,
	IMG_FORMAT_RGB		= 2 | IMG_FORMAT_F_COLOR,
	IMG_FORMAT_RGBA		= 3 | IMG_FORMAT_F_COLOR | IMG_FORMAT_F_ALPHA,
} ImgFormat;

typedef unsigned long ImgColorInt;

static inline
ImgColorInt img_color_map(const ImgColor c, ImgFormat fmt);

enum ImgColorTransform {
	IMG_COLOR_TRANF_NULL		= 0,
	IMG_COLOR_TRANF_BGR			= 1,
	IMG_COLOR_TRANF_GRB			= 2,
	IMG_COLOR_TRANF_GRAY_MIN	= 3,
	IMG_COLOR_TRANF_GRAY_MAX	= 4,
	IMG_COLOR_TRANF_INVERSE		= 5,
};

static inline
ImgColor img_color_transform(const ImgColor col, unsigned tranf);

enum {
	IMG_HSV_ABITS = 4,  //aditional pression bits
	IMG_HSV_VSCALE = (255 << IMG_HSV_ABITS),
	IMG_HSV_SSCALE = (255 << IMG_HSV_ABITS),
	IMG_HSV_HSCALE = (256 << IMG_HSV_ABITS),	//hue scale
	IMG_HSV_ASCALE = (255 << IMG_HSV_ABITS),
};

typedef struct ImgColorHSV {
	uint16_t h, s, v, a;
} ImgColorHSV;

ImgColor img_color_hsv2rgb(const ImgColorHSV hsv);

ImgColorHSV img_color_rgb2hsv(const ImgColor rgb);

/*
	Image
*/
typedef enum ImgFlags {
	IMG_F_OWN_MEM		= 1,
} ImgFlags;

typedef struct Image {
	unsigned char *	data;
	unsigned		w, h;
	unsigned		pitch;	//bytes per line
	unsigned		bypp;	//bytes per pixel
	ImgFormat		format;
	int				flags;
} Image;

void img_free(Image* img);

static inline
bool img_empty(const Image* img) {
	return !img || !img->w || !img->h || !img->data;
}

int img_resize(Image* img, unsigned w, unsigned h, ImgFormat fmt,
	unsigned pitch);

int img_copy(Image* dst, const Image* src);

void img_view_make(Image* dst, const Image* src, const ImgRect rect);

void img_fill(Image* img, const ImgColor color);

static inline
void img_zero(Image* img);

static inline
ImgColor img_pixel_get(const Image* img, unsigned x, unsigned y);

#define IMG_INDEX(I,X,Y) \
	((I).data[ (I).pitch * (Y) + (I).bypp * (X) ])

#define IMG_INDEX3(I,X,Y,C) \
	((I).data[ (I).pitch * (Y) + (I).bypp * (X) + (C)])

/*
	Inline implementation
*/
static inline
ImgColorInt img_color_map(const ImgColor c, ImgFormat fmt)
{
	unsigned long n=0;
	unsigned char * p = (unsigned char *)&n;
	switch (fmt) {
	case IMG_FORMAT_GRAY:
		p[0] = c.r;
		if (p[0] < c.g) p[0] = c.g;
		if (p[0] < c.b) p[0] = c.b;
		break;
	case IMG_FORMAT_RGB:	p[0]=c.r; p[1]=c.g; p[2]=c.b; break;
	case IMG_FORMAT_RGBA:	p[0]=c.r; p[1]=c.g; p[2]=c.b; p[3]=c.a; break;
	default:  break;
	}
	return n;
}

static inline
ImgColor img_color_transform(const ImgColor col, unsigned tranf)
{
	switch (tranf) {
	case IMG_COLOR_TRANF_BGR:
		return (ImgColor){ col.b, col.g, col.r, col.a };
	case IMG_COLOR_TRANF_GRB:
		return (ImgColor){ col.g, col.b, col.r, col.a };
	case IMG_COLOR_TRANF_GRAY_MIN: {
		unsigned char m = col.r < col.g ? col.r : col.g;
		if (col.b < m) m = col.b;
		return (ImgColor){ m, m, m, col.a };
	}
	case IMG_COLOR_TRANF_GRAY_MAX: {
		unsigned char m = col.r > col.g ? col.r : col.g;
		if (col.b > m) m = col.b;
		return (ImgColor){ m, m, m, col.a };
	}
	case IMG_COLOR_TRANF_INVERSE:
		return (ImgColor){ 255-col.b, 255-col.g, 255-col.r, col.a };
	default:
		return col;
	}
}

static inline
void img_zero(Image* img)
{
	if (img->data)
		memset(img->data, 0, img->pitch * img->h);
}

static inline
ImgColor img_pixel_get(const Image* img, unsigned x, unsigned y) {
	unsigned char * p = img->data + img->pitch * y + img->bypp * x;
	switch (img->format) {
	case IMG_FORMAT_GRAY:	return (ImgColor){ *p, *p, *p, 255 };
	case IMG_FORMAT_RGB:	return (ImgColor){ p[0], p[1], p[2], 255 };
	case IMG_FORMAT_RGBA:	return (ImgColor){ p[0], p[1], p[2], p[3] };
	default:				return (ImgColor){0};
	}
}

