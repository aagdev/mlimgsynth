/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 */
#include "ids.h"

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
