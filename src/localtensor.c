/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 */
#include "localtensor.h"
#include "ccommon/stream.h"
#include "ccommon/logging.h"
#include "ccommon/image_io.h"
#include <string.h>

void ltensor_copy_slice(LocalTensor* dst, const LocalTensor* src,
	int n0 , int n1 , int n2 , int n3 ,
	int di0, int di1, int di2, int di3,
	int si0, int si1, int si2, int si3,
	int ds0, int ds1, int ds2, int ds3,
	int ss0, int ss1, int ss2, int ss3 )
{
	// Bound checks  //TODO: complete
	assert( si0 + ss0 * n0 <= src->s[0] );
	assert( si1 + ss1 * n1 <= src->s[1] );
	assert( si2 + ss2 * n2 <= src->s[2] );
	assert( si3 + ss3 * n3 <= src->s[3] );

	assert( di0 + ds0 * n0 <= dst->s[0] );
	assert( di1 + ds1 * n1 <= dst->s[1] );
	assert( di2 + ds2 * n2 <= dst->s[2] );
	assert( di3 + ds3 * n3 <= dst->s[3] );
	
	// Strides for contiguous tensors
	int ss1c = src->s[0];
	int ss2c = src->s[0] * src->s[1]; 
	int ss3c = src->s[0] * src->s[1] * src->s[2]; 

	int ds1c = dst->s[0];
	int ds2c = dst->s[0] * dst->s[1]; 
	int ds3c = dst->s[0] * dst->s[1] * dst->s[2]; 
	
	// Initial positions in memory
	const float *sp = src->d +si0 +si1*ss1c +si2*ss2c +si3*ss3c;
	float       *dp = dst->d +di0 +di1*ds1c +di2*ds2c +di3*ds3c;
	
	// Convert steps to strides
	ss1 *= ss1c;
	ss2 *= ss2c;
	ss3 *= ss3c;
	
	ds1 *= ds1c;
	ds2 *= ds2c;
	ds3 *= ds3c;

	// Set
	for (int i3=0; i3<n3; ++i3)
	for (int i2=0; i2<n2; ++i2)
	for (int i1=0; i1<n1; ++i1)
	for (int i0=0; i0<n0; ++i0)
		dp[i0*ds0 +i1*ds1 +i2*ds2 +i3*ds3] = sp[i0*ss0 +i1*ss1 +i2*ss2 +i3*ss3];
}

float ltensor_minmax(const LocalTensor* S, float* min)
{
	float mn=S->d[0], mx=mn;
	for (unsigned i=1, n=ltensor_nelements(S); i<n; ++i) {
		MINSET(mn, S->d[i]);
		MAXSET(mx, S->d[i]);
	}
	*min = mn;
	return mx;
}

float ltensor_sum(const LocalTensor* S)
{
	unsigned n=ltensor_nelements(S);
	double s=0;
	for (unsigned i=0; i<n; ++i) s += S->d[i];
	return s;
}

float ltensor_mean(const LocalTensor* S)
{
	unsigned n=ltensor_nelements(S);
	double s=0;
	for (unsigned i=0; i<n; ++i) s += S->d[i];
	return s / n;
}

int ltensor_save_path(const LocalTensor* S, const char* path)
{
	int R=1;
	Stream stm={0};
	log_debug("Writing tensor to '%s'", path);
	TRY_LOG( stream_open_file(&stm, path, SOF_CREATE),
		"could not open '%s'", path);
	// Similar to the PNM image format
	stream_printf(&stm, "TENSOR F32 %d %d %d %d\n", LT_SHAPE_UNPACK(*S));
	stream_write(&stm, ltensor_nbytes(S), S->d);
end:
	stream_close(&stm, 0);
	return R;
}

int ltensor_load_path(LocalTensor* S, const char* path)
{
	int R=1;
	Stream stm={0};
	log_debug("Reading tensor from '%s'", path);
	TRY_LOG( stream_open_file(&stm, path, SOF_READ), "could not open '%s'", path);
	char *end, *cur = stream_read_buffer(&stm, &end);
	if (!(cur+24 < end) || memcmp(cur, "TENSOR F32 ", 11)) goto error_fmt;
	cur += 11;
	int s[4]={1,1,1,1};
	for (int i=0; i<4; ++i) {
		int n=0;
		for (; cur<end && '0' <= *cur && *cur <= '9'; ++cur)
			n = n*10 + (*cur - '0');
		s[i] = n;
		if (!(cur < end)) goto error_fmt;
		if (*cur == '\n') { cur++; break; }
		if (i==3 || *cur != ' ') goto error_fmt;
		cur++;
	}
	stream_commit(&stm, cur);
	ltensor_resize(S, s[0], s[1], s[2], s[3]);
	stream_read(&stm, ltensor_nbytes(S), S->d);
end:
	stream_close(&stm, 0);
	return R;
error_fmt:
	ERROR_LOG(-1, "file '%s' is not a valid tensor", path);
}

void ltensor_from_image(LocalTensor* S, const Image* img)
{
	int n0=img->w, n1=img->h, n2=img->bypp;
	ltensor_resize(S, n0, n1, n2, 1);
	for (int y=0; y<n1; ++y) {
		for (int x=0; x<n0; ++x) {
			for (int c=0; c<n2; ++c) {
				float v = IMG_INDEX3(*img, x, y, c) / 255.0f;
				S->d[n0*n1*c +n0*y +x] = v;
			}
		}
	}
}

void ltensor_to_image(const LocalTensor* S, Image* img)
{
	int n0=S->s[0], n1=S->s[1], n2=S->s[2];
	assert(S->s[2] == 3 && S->s[3] == 1);
	img_resize(img, n0, n1, IMG_FORMAT_RGB, 0);
	for (int y=0; y<n1; ++y) {
		for (int x=0; x<n0; ++x) {
			for (int c=0; c<n2; ++c) {
				float v = S->d[n0*n1*c +n0*y +x];
				ccCLAMP(v, 0, 1);
				IMG_INDEX3(*img, x, y, c) = v * 255.0f;
			}
		}
	}
}

int ltensor_img_redblue(const LocalTensor* S, Image* img)
{
	int R=1;

	if (!(S->s[0]>0 && S->s[1]>0 && S->s[2]==1 && S->s[3]==1))
		ERROR_LOG(-1, "ltensor wrong shape for 2d plot: " LT_SHAPE_FMT,
			LT_SHAPE_UNPACK(*S));

	float mn, mx = ltensor_minmax(S, &mn);
	float scale = MAXg(mx, -mn);

	unsigned w=S->s[0], h=S->s[1];
	img_resize(img, w,h, IMG_FORMAT_RGB, 0);

	const float *f = S->d;
	for (unsigned y=0; y<h; ++y) {
		uint8_t *d = &IMG_INDEX(*img, 0, y);
		for (unsigned x=0; x<w; ++x, d+=3, ++f) {
			float v = *f;
			d[0] = v < 0 ? -v*255/scale : 0;
			d[1] = 0;
			d[2] = v > 0 ?  v*255/scale : 0;
		}
	}

end:
	return R;
}

int ltensor_img_redblue_path(const LocalTensor* S, const char* path)
{
	int R=1;
	Image img={0};
	log_debug("Writing tensor visualization to '%s'", path);
	TRY( ltensor_img_redblue(S, &img) );
	TRY( img_save_file(&img, path) );
end:
	img_free(&img);
	return R;
}
