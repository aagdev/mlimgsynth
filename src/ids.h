/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * Global string-id bidirectional map.
 */
#pragma once
#include "ccommon/stringstore.h"

#define LIST_OF_IDS \
	S(NULL) \
	S(ML_BLOCK_BEGIN) \
	S(vae) \
	S(clip) \
	S(uniform) \
	S(karras) \
	S(euler) \
	S(heun) \
	S(taylor3) \
	S2(dpmpp2m,"dpm++2m") \
	S2(dpmpp2s,"dpm++2s") \
	S(check) \
	S2(list_backends,"list-backends") \
	S2(vae_encode,"vae-encode") \
	S2(vae_decode,"vae-decode") \
	S2(vae_test,"vae-test") \
	S2(clip_encode,"clip-encode") \
	S2(generate,"generate") \

enum {
#define S(X) ID_##X,
#define S2(X,_) ID_##X,
	LIST_OF_IDS
#undef S2
#undef S
	ID__END
};

extern StringStore g_ss;

void ids_init();

static inline
const char * id_str(int id) {
	return strsto_get(&g_ss, id).b;
}

static inline int id_fromsl(const StrSlice sl) {
	return sl.s ? strsto_add(&g_ss, sl) : 0;
}

static inline int id_fromz(const char* str) {
	return str && str[0] ? strsto_add(&g_ss, strsl_fromz(str)) : 0;
}

/* Implementation */
#ifdef IDS_IMPLEMENTATION
#undef IDS_IMPLEMENTATION
StringStore g_ss;

const char * ids_str[ID__END] = {
#define S(X)  #X,
#define S2(X,S) S,
	LIST_OF_IDS
#undef S2
#undef S
};

void ids_init() {
	for (unsigned i=0; i<ID__END; ++i)	
		strsto_add2(&g_ss, strsl_fromz(ids_str[i]), i, true);
}
#endif
