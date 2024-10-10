/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 */
#include "lora.h"
#include "ccommon/logging.h"
#include "ggml.h"
#include <math.h>

/*static
int tstensor_to_ggml(struct ggml_tensor **out, struct ggml_context *ctx, 
	TSTensorEntry* te, TSTensorData* td, int wtype, int tdgflags)
{
	struct ggml_tensor* ten=NULL;
	int tsdt = tstore_dtype_from_ggml(wtype);	
	ten = ggml_new_tensor_4d(ctx, wtype, TSTENSOR_SHAPE4_UNPACK(*te));
	TRYR( tstore_tensor_data_get(te, tsdt, tdgflags, td) );
	assert( td->size == ggml_nbytes(ten) );
	ten->data = td->data;
	*out = ten;
	return 1;
}

int lora_apply_inner(TSTensorEntry* dst, TSTensorEntry* ld, TSTensorEntry* lu,
	TSTensorEntry *ls, TSTensorEntry *la, float mult, MLCtx ctx)
{
	int R=1;
	TSTensorData td_ld={0}, td_lu={0}, td_ls={0}, td_dst={0};

	struct ggml_context *ctx=NULL;
	struct ggml_cgraph *gf;
	struct ggml_tensor *t_ld, *t_lu, *t_ls, *t_dst, *t_lora;
	
	unsigned n_inner = ld->shape[ld->shape_n-1],
	         n0 = tstore_tensor_count(ld) / n_inner,
			 n1 = tstore_tensor_count(lu) / n_inner;

	if (!(dst->shape_n >= 2 &&
		ld->shape_n == dst->shape_n &&
		lu->shape_n == dst->shape_n &&
		tstore_tensor_count(dst) == n0 * n1))
	{
		ERROR_LOG(-1, "lora up/down invalid shapes");
	}

	// ggml init
	size_t size = ggml_tensor_overhead() * 32 + ggml_graph_overhead()
		+ tstore_tensor_count(ld)*4 + tstore_tensor_count(dst)*4*2;
	ctx = ggml_init((struct ggml_init_params){ size, NULL, true });

	// Scale get
	float scale=1;
	if (ls) {
		TRY( tstensor_to_ggml(&t_ls, ctx, ls, &td_ls, GGML_TYPE_F32, 0) );
		scale = *(float*)t_ls->data;
	}
	else if (la) {
		TRY( tstensor_to_ggml(&t_ls, ctx, la, &td_ls, GGML_TYPE_F32, 0) );
		scale = *(float*)t_ls->data / n_inner;
	}
	scale *= mult;
	assert( scale > 0 );
	
	// Load tensors
	int wtype = GGML_TYPE_F16;  //TODO: param
	t_ld  = ggml_new_tensor_4d(ctx, wtype, TSTENSOR_SHAPE4_UNPACK(*ld ));
	t_lu  = ggml_new_tensor_4d(ctx, wtype, TSTENSOR_SHAPE4_UNPACK(*lu ));
	t_dst = ggml_new_tensor_4d(ctx, wtype, TSTENSOR_SHAPE4_UNPACK(*dst));
	
	t_ld  = ggml_reshape_2d(ctx, t_ld , n0, n_inner);
	t_lu  = ggml_reshape_2d(ctx, t_lu , n_inner, n1);
	t_dst = ggml_reshape_2d(ctx, t_dst, n0, n1);

	// Make graph
	//ggml_set_no_alloc(ctx, false);
	gf = ggml_new_graph(ctx);
	t_ld   = ggml_cont(ctx, ggml_transpose(ctx, t_ld));
	t_lora = ggml_mul_mat(ctx, t_lu, t_ld);
	t_lora = ggml_cont(ctx, ggml_transpose(ctx, t_lora));
	t_lora = ggml_scale_inplace(ctx, t_lora, scale);
	t_dst  = ggml_add_inplace(ctx, t_dst, t_lora);
	ggml_build_forward_expand(gf, t_dst);

	// Alloc and set inputs

	TRY( tstensor_to_ggml(&t_ld , ctx, ld , &td_ld , wtype, 0) );
	TRY( tstensor_to_ggml(&t_lu , ctx, lu , &td_lu , wtype, 0) );
	TRY( tstensor_to_ggml(&t_dst, ctx, dst, &td_dst, wtype,
			TSTDG_F_PERM | TSTDG_F_WRITE) );
	
	// Compute
	int r = ggml_graph_compute_with_ctx(ctx, gf, 0);
	if (r) ERROR_LOG(-1, "ggml compute: %d", r);

	// Store
	assert( ggml_nbytes(t_dst) == td_dst.size );
	memcpy(td_dst.data, t_dst->data, td_dst.size);

end:
	tstore_tdata_free(&td_dst);
	tstore_tdata_free(&td_lu);
	tstore_tdata_free(&td_ld);
	ggml_free(ctx);
	return R;
}*/

int lora_apply_inner(TSTensorEntry* dst, TSTensorEntry* ld, TSTensorEntry* lu,
	TSTensorEntry *ls, TSTensorEntry *la, float mult, MLCtx* C)
{
	int R=1;
	TSTensorData td_ld={0}, td_lu={0}, td_dst={0};
	
	unsigned n_inner = ld->shape[ld->shape_n-1],
	         n0 = tstore_tensor_count(ld) / n_inner,
			 n1 = tstore_tensor_count(lu) / n_inner;

	if (!(dst->shape_n >= 2 &&
		ld->shape_n == dst->shape_n &&
		lu->shape_n == dst->shape_n &&
		tstore_tensor_count(dst) == n0 * n1))
	{
		ERROR_LOG(-1, "lora up/down invalid shapes");
	}

	// Must init ggml before any tensor conversion
	mlctx_begin(C, "lora");
	C->c.quiet = true;

	// Scale get
	float scale=1;
	if (ls) {
		TRY( tstore_tensor_data_get(ls, TS_DTYPE_F32, 0, &td_ld) );
		scale = *(float*)td_ld.data;
	}
	else if (la) {
		TRY( tstore_tensor_data_get(la, TS_DTYPE_F32, 0, &td_ld) );
		scale = *(float*)td_ld.data / n_inner;
	}
	scale *= mult;
	assert( scale > 0 );
	
	// Get data
	int wtype = C->c.wtype;
	int tsdt = tstore_dtype_from_ggml(wtype);
	assert( tsdt > 0 );
	
	TRY( tstore_tensor_data_get(ld , tsdt, 0, &td_ld ) );
	TRY( tstore_tensor_data_get(lu , tsdt, 0, &td_lu ) );
	TRY( tstore_tensor_data_get(dst, tsdt, TSTDG_F_PERM | TSTDG_F_WRITE, &td_dst) );

	// Make graph
	MLTensor *t_ld, *t_lu, *t_dst, *t_out;
	t_ld  = mlctx_input_new(C, "ld" , wtype, n0, n_inner, 1, 1);
	t_lu  = mlctx_input_new(C, "lu" , wtype, n_inner, n1, 1, 1);
	t_dst = mlctx_input_new(C, "dst", wtype, n0, n1, 1, 1);
	
	t_out = ggml_cont(C->cc, ggml_transpose(C->cc, t_ld));
	t_out = ggml_mul_mat(C->cc, t_lu, t_out);
	t_out = ggml_cont(C->cc, ggml_transpose(C->cc, t_out));
	t_out = ggml_scale_inplace(C->cc, t_out, scale);
	t_out = ggml_add_inplace(C->cc, t_dst, t_out);
	
	mlctx_tensor_add(C, "output", t_out);
	TRY( mlctx_prep(C) );

	// Set inputs
	ggml_backend_tensor_set(t_ld , td_ld .data, 0, td_ld .size);
	ggml_backend_tensor_set(t_lu , td_lu .data, 0, td_lu .size);
	ggml_backend_tensor_set(t_dst, td_dst.data, 0, td_dst.size);

	// Compute
	TRY( mlctx_compute(C) );

	// Store output
	assert( ggml_nbytes(t_out) == td_dst.size );
	ggml_backend_tensor_get(t_out, td_dst.data, 0, td_dst.size);
	
	// Check
	float v=0;
	if (wtype == GGML_TYPE_F16)
		v = ggml_fp16_to_fp32(*(ggml_fp16_t*)td_dst.data);
	else if (wtype == GGML_TYPE_F32)
		v = *(float*)td_dst.data;
	if (!isfinite(v))
		ERROR_LOG(-1, "NaN in LoRA result");

end:
	tstore_tdata_free(&td_dst);
	tstore_tdata_free(&td_lu);
	tstore_tdata_free(&td_ld);
	mlctx_free(C);
	C->c.quiet = false;
	return R;
}

int lora_apply(TensorStore* ts_dst, TensorStore* ts_lora, float mult,
	MLCtx* ctx)
{
	int R=1;
	StrSlice name={0};
	TSTensorData td={0};
	DynStr tmps=NULL;

	vec_forp(TSTensorEntry, ts_lora->tensors, ld, 0) {
		name = strsto_get(ts_lora->ss, ld->key);
		if (!( strsl_suffix_trim(&name, strsl_static(".lora_down.weight")) ))
			continue;

		dstr_copy(tmps, name.s, name.b);
		dstr_appendz(tmps, ".weight");
		TSTensorEntry *dst = tstore_tensor_get(ts_dst, tmps);
		if (!dst) ERROR_LOG(-1, "lora tensor not found in model: %s", tmps);

		dstr_copy(tmps, name.s, name.b);
		dstr_appendz(tmps, ".lora_up.weight");
		TSTensorEntry *lu = tstore_tensor_get(ts_lora, tmps);
		if (!lu) ERROR_LOG(-1, "lora up tensor not found: %s", tmps);

		dstr_copy(tmps, name.s, name.b);
		dstr_appendz(tmps, ".scale");
		TSTensorEntry *ls = tstore_tensor_get(ts_lora, tmps);

		dstr_copy(tmps, name.s, name.b);
		dstr_appendz(tmps, ".alpha");
		TSTensorEntry *la = tstore_tensor_get(ts_lora, tmps);

		// Apply
		log_debug("lora apply %.*s", (int)name.s, name.b);
		TRY( lora_apply_inner(dst, ld, lu, ls, la, mult, ctx) );
	}

end:
	if (R<0) log_error("lora tensor '%.*s': %x", (int)name.s, name.b, -R);
	tstore_tdata_free(&td);
	dstr_free(tmps);
	return R;
}
