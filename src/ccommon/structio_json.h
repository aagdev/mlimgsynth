/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#pragma once
#include "structio.h"

enum StioStreamJsonFlag {
	STIO_SF_JSON_PRETTY = STIO_IF_CUSTOM, //WIP
};

extern const StioClass stio_class_json;

int stio_json_write(StioStream* sio, StioCtx* ctx, StioItem* itm);

int stio_json_read(StioStream* sio, StioCtx* ctx, StioItem* itm);

