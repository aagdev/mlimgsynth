/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * mlimgsynth library option_get implementation.
 */
OPTION( MODEL ) {
	ARG_STR( S->c.path_model );
}
OPTION( MODEL_TYPE ) {
	ARG_ENUM( S->c.model_type, mlis_model_type_froms );
}
OPTION( PROMPT ) {
	ARG_STR( S->c.prompt );
}
OPTION( NPROMPT ) {
	ARG_STR( S->c.nprompt );
}
//TODO: complete
