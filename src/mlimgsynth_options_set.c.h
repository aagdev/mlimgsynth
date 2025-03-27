/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * mlimgsynth library option_set implementation.
 */
OPTION( BACKEND ) {
	ARG_STR(name  , 0, 65535)
	ARG_STR(params, 0, 65535)
	dstr_copy(S->c.backend, name.s, name.b);
	dstr_copy(S->c.be_params, params.s, params.b);
	S->rflags &= ~MLIS_READY_BACKEND;
}
OPTION( MODEL ) {
	ARG_STR_NO_PARSE(path, 1, 65535)
	dstr_copy(S->c.path_model, path.s, path.b);
	S->rflags &= ~MLIS_READY_MODEL;
}
OPTION( TAE ) {
	ARG_STR_NO_PARSE(path, 0, 65535)
	dstr_copy(S->c.path_tae, path.s, path.b);
	bool en = !dstr_empty(S->c.path_tae);
	ccFLAG_SET(S->c.flags, MLIS_CF_USE_TAE, en);
}
OPTION( MODEL_TYPE ) {
	ARG_ENUM(id, mlis_model_type_froms)
	TRY( mlis_model_type_set(S, id) );
}
OPTION( AUX_DIR ) {
	ARG_STR_NO_PARSE(path, 0, 65535)
	dstr_copy(S->c.path_aux, path.s, path.b);
}
OPTION( LORA_DIR ) {
	ARG_STR_NO_PARSE(path, 0, 65535)
	dstr_copy(S->c.path_lora_dir, path.s, path.b);
}
OPTION( LORA ) {
	ARG_STR(path, 1, 65535)
	ARG_FLOAT(mult, 0, 1, 1);
	TRY( mlis_cfg_lora_add(S, path, mult, 0) );
}
OPTION( LORA_CLEAR ) {
	mlis_cfg_loras_free(S);	
}
OPTION( PROMPT ) {
	ARG_STR_NO_PARSE(prompt, 0, 65535)
	mlis_cfg_prompt_set(S, prompt);
}
OPTION( NPROMPT ) {
	ARG_STR_NO_PARSE(prompt, 0, 65535)
	dstr_copy(S->c.nprompt, prompt.s, prompt.b);
}
OPTION( IMAGE_DIM ) {
	ARG_INT(w, 0, 65535, 0)
	ARG_INT(h, 0, 65535, 0)
	S->c.width  = w;
	S->c.height = h;
}
OPTION( BATCH_SIZE ) {
	ARG_INT(i, 0, 1024, 0)
	S->c.n_batch = i;
}
OPTION( CLIP_SKIP ) {
	ARG_INT(i, 0, 255, 0)
	S->c.clip_skip = i;
}
OPTION( CFG_SCALE ) {
	ARG_FLOAT(f, 0, 255, NAN)
	S->c.cfg_scale = f;
}
OPTION( METHOD ) {
#ifdef ARG_IS_STR
	StrSlice ss = strsl_fromz(value);
	if (strsl_suffixz_trim(&ss, "_a")) {  // Shortcut for ancestral methods
		int id = mlis_method_froms(ss);
		if (id < 0)
			ERROR_LOG(MLIS_E_OPT_VALUE, "invalid method name '%s'", value);
		S->sampler.c.method = id;
		S->sampler.c.s_ancestral = 1;
		goto done;
	}
#endif
	ARG_ENUM(id, mlis_method_froms)
	S->sampler.c.method = id;
}
OPTION( SCHEDULER ) {
	ARG_ENUM(id, mlis_sched_froms)
	S->sampler.c.sched = id;
}
OPTION( STEPS ) {
	ARG_INT(i, 0, 1000, 0)
	S->sampler.c.n_step = i;
}
OPTION( F_T_INI ) {
	ARG_FLOAT(f, 0, 1, NAN)
	S->sampler.c.f_t_ini = f;
}
OPTION( F_T_END ) {
	ARG_FLOAT(f, 0, 1, NAN)
	S->sampler.c.f_t_end = f;
}
OPTION( S_NOISE ) {
	ARG_FLOAT(f, 0, 255, NAN)
	S->sampler.c.s_noise = f;
}
OPTION( S_ANCESTRAL ) {
	ARG_FLOAT(f, 0, 255, NAN)
	S->sampler.c.s_ancestral = f;
}
OPTION( IMAGE ) {
	ARG_C(img, const MLIS_Image*)
	if (img->c != 3 && img->c != 4)
		ERROR_LOG(MLIS_E_IMAGE,
			"invalid number of channels in image: %d", img->c);
	if (mlis_tensor_from_image(&S->image, img) < 0)
		ERROR_LOG(MLIS_E_IMAGE, "invalid image");
	S->c.tuflags |= MLIS_TUF_IMAGE;
	
	if (S->image.n[2] == 4) {  // Take mask from last channel (alpha)
		unsigned w = S->image.n[0];
		unsigned h = S->image.n[1];
		mlis_tensor_resize(&S->mask, w, h, 1, 1);
		memcpy(S->mask.d, S->image.d+(w*h*3*4), w*h*4);
		S->image.n[2] = 3;
		S->c.tuflags |= MLIS_TUF_MASK;
	}
}
OPTION( IMAGE_MASK ) {
	ARG_C(img, const MLIS_Image*)
	if (img->c != 1)
		ERROR_LOG(MLIS_E_IMAGE,
			"invalid number of channels in image mask: %d", img->c);
	if (mlis_tensor_from_image(&S->mask, img) < 0)
		ERROR_LOG(MLIS_E_IMAGE, "invalid image mask");
	S->c.tuflags |= MLIS_TUF_MASK;
}
OPTION( NO_DECODE ) {
	ARG_BOOL(en)
	ccFLAG_SET(S->c.flags, MLIS_CF_NO_DECODE, en);
}
OPTION( TENSOR_USE_FLAGS ) {
	ARG_FLAGS(fl)
	S->c.tuflags = fl;
}
OPTION( SEED ) {
#ifdef ARG_IS_STR
	if (!vcur[0]) goto done;  // Empty string -> keep random seed
#endif
	ARG_UINT64(i)
	g_rng.seed = i;  //TODO: local rng
}
OPTION( VAE_TILE ) {
	ARG_INT(i, 0, 65535, 0)
	S->c.vae_tile = i;
}
OPTION( UNET_SPLIT ) {
	ARG_BOOL(en)
	ccFLAG_SET(S->c.flags, MLIS_CF_UNET_SPLIT, en);
}
OPTION( THREADS ) {
	ARG_INT(i, 0, 65535, 0)
	S->c.n_thread = i;
	S->rflags &= ~MLIS_READY_BACKEND;  //this is overkill...
}
OPTION( DUMP_FLAGS ) {
	ARG_FLAGS(fl)
	S->c.dump_flags = fl;
}
OPTION( CALLBACK ) {
	ARG_C(func, MLIS_Callback)
	ARG_C(user, void*)
	S->callback = func;
	S->callback_ud = user;
}
OPTION( ERROR_HANDLER ) {
	ARG_C(func, MLIS_ErrorHandler)
	ARG_C(user, void*)
	S->errh = func;
	S->errh_ud = user;
}
OPTION( LOG_LEVEL ) {
	// Warning: this sets a global configuration, not associated with the context.

#ifdef ARG_IS_STR
	int lvls = mlis_loglvl_fromz(vcur);
	if (lvls >= 0) {
		log_level_set(lvls);
		goto done;
	}
#endif

	ARG_INT(lvl, 0, 0x2ff, -1)
	if ((lvl & 0xf00) == 0x100) {
		// Increase verbosity, starting directly from INFO.
		if (!log_level_check(LOG_LVL_INFO))
			log_level_set(LOG_LVL_INFO);
		else
			log_level_inc(lvl & 0xff);
	} else if ((lvl & 0xf00) == 0x200)
		log_level_inc(-(lvl & 0xff));
	else
		log_level_set(lvl);
}
