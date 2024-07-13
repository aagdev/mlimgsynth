/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 */
#include "util.h"
#include "ccommon/vector.h"
#include "ccommon/logging.h"

int img_save_file_info(const Image* img, const char* path,
	const char* info_key, const char* info_text)
{
	int R=1;
	Stream stm={0};
	ImageIO imgio={0};
	DynStr tmps=NULL;

	const ImageCodec* codec = img_codec_detect_filename(path, IMG_OF_SAVE);
	if (!codec) ERROR_LOG(-1, "Could not found an image codec to save '%s'", path);

	TRY_LOG( stream_open_file(&stm, path, SOF_CREATE),
		"Could not open '%s'", path );

	TRY( imgio_open_stream(&imgio, &stm, IMG_OF_SAVE, codec) );

	if (info_text && info_key) {
		dstr_copyz(tmps, info_key);
		dstr_push(tmps, '\0');
		dstr_appendz(tmps, info_text);
		int r = imgio_value_set(&imgio, IMG_VALUE_METADATA, tmps,
				dstr_count(tmps)+1);
		if (r<0)
			log_warning("Could not write '%s' in '%s'", info_key, path);
	}

	TRY( imgio_save(&imgio, img) );

end:
	dstr_free(tmps);
	imgio_free(&imgio);
	stream_close(&stm, 0);
	return R;
}
