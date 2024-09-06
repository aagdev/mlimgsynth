/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 */
#include "mlblock.h"
#include "ccommon/timing.h"
#include <inttypes.h>

#define F_MIB  (1.0 / (1024.0*1024.0))

#define mllog_debug2(...)  if (!C->c.quiet) log_debug2(__VA_ARGS__)
#define mllog_debug(...)   if (!C->c.quiet) log_debug(__VA_ARGS__)
#define mllog_info(...)    if (!C->c.quiet) log_info(__VA_ARGS__)
#define mllog_warn(...)    if (!C->c.quiet) log_warn(__VA_ARGS__)
#define mllog_error(...)   log_error(__VA_ARGS__)

#define id_fromz(X)  strsto_add(C->ss, strsl_fromz(X))
#define id_str(X)  strsto_get(C->ss, X).b

void mlctx_free(MLCtx* C)
{
	if (C->allocr) {
		ggml_gallocr_free(C->allocr);
		C->allocr=NULL;
	}
	if (C->sched) {
		ggml_backend_sched_free(C->sched);
		C->sched = NULL;
	}
	if (C->bkbuf) {
		ggml_backend_buffer_free(C->bkbuf);
		C->bkbuf = NULL;
	}
	
	vec_free(C->tensors);
	vec_free(C->inputs);
	C->result = NULL;
	if (C->cc) {
		ggml_free(C->cc);
		ggml_free(C->cp);
		C->cc = NULL;
		C->cp = NULL;
		C->graph = NULL;
	}
}

void mlctx_begin(MLCtx* C, const char* name)
{
	mlctx_free(C);
	IFFALSESET(C->c.n_tensor_max, GGML_DEFAULT_GRAPH_SIZE);
	size_t size = ggml_tensor_overhead() * C->c.n_tensor_max
				+ ggml_graph_overhead();
	C->cc = ggml_init((struct ggml_init_params){ size, NULL, true });
	C->cp = ggml_init((struct ggml_init_params){ size, NULL, true });
	C->c.name = name ? name : "";
	C->info = (struct MLCtxInfo){0};
}

int mlctx_load_prep(MLCtx* C)
{
	int R=1;
	DynStr name=NULL;
	struct SEntry { MLCtxTensor* p; unsigned iname; } * stack=NULL;  //vector

	IFFALSESET(C->c.tpath_sep, '.');

	vec_forrp(MLCtxTensor, C->tensors, p)
	{
		unsigned nlen = dstr_count(name);

		if (p->name == MLB_NAME_BLOCK_BEGIN) {
			if (!vec_count(stack)) ERROR_LOG(-1, "invalid ML graph");
			struct SEntry e = vec_pop(stack);
			dstr_resize(name, e.iname);
		}
		else if (p->name == MLB_NAME_SPLIT) {
		}
		else {
			if (name) dstr_push(name, C->c.tpath_sep);
			dstr_appendz(name, id_str(p->name));

			if (p->tensor->op == GGML_OP_NONE) {  //param
				p->key = id_fromz(name);  //store tensor full name
				dstr_resize(name, nlen);
			}
			else {  //block
				vec_push(stack, ((struct SEntry){p, nlen}));
			}
		}
	}

end:
	if (R<0 && name) mllog_error("tensor '%s'", name);
	dstr_free(name);
	vec_free(stack);
	return R;
}

int mlctx_build(MLCtx* C, MLTensor* result)
{
	int R=1;
	assert(!C->graph);
	
	if (C->c.dump) {
		DynStr path = dstr_stack(64);
		dstr_printf(path, "%s-graph.txt", C->c.name);
		TRYR( mlctx_block_graph_dump_path(C, path) );
	}
	
	mllog_debug("%s result: %s", C->c.name, ggml_tensor_typeshape_desc(result));

	// Parameter tensors
	//C->info.mem_params = ggml_ctx_tensors_total_size(C->cp);
	size_t m=0;
	unsigned n=0;
	vec_forp(MLCtxTensor, C->tensors, p, 0) {
		if (p->tensor && p->tensor->op == GGML_OP_NONE) {
			m += ggml_nbytes(p->tensor);
			n++;
			// Prevents the graph allocator from reusing param tensors.
			// Allows the computation can be repeated without reloading
			// the parameters.
			// Will increase memory usage, but it is not so much usually.
			if (C->c.multi_compute)
				ggml_set_output(p->tensor);
		}
	}
	C->info.mem_params = m;
	mllog_debug("mlblock params n:%u size:%zu", n, m);
	assert( C->info.mem_params < (size_t)1024*1024*1024*1024 );

	assert( vec_last(C->tensors,0).tensor == result );
	ggml_set_output(result);
	C->result = result;

	// Computation graph
	C->graph = ggml_new_graph_custom(C->cc, C->c.n_tensor_max, false);

	vec_forp(MLCtxTensor, C->tensors, p, 0) {
		if (p->name == MLB_NAME_SPLIT) {  //TODO: split == output ?
    		ggml_build_forward_expand(C->graph, p->tensor);
		}
	}

    ggml_build_forward_expand(C->graph, result);
	mllog_debug("graph size:%d n_nodes:%d n_leafs:%d",
		C->graph->size, C->graph->n_nodes, C->graph->n_leafs);

//end:
	return R;
}

int mlctx_alloc(MLCtx* C)
{
	int R=1;
	assert(!C->allocr && !C->sched);

#if !USE_GGML_SCHED
	C->allocr = ggml_gallocr_new(
		ggml_backend_get_default_buffer_type(C->backend) );

	mllog_debug("allocating memory");
	if (!ggml_gallocr_reserve(C->allocr, C->graph))  //this allocates
		ERROR_LOG(-1, "%s could not allocate memory", C->c.name);
	if (!ggml_gallocr_alloc_graph(C->allocr, C->graph))
		ERROR_LOG(-1, "ggml compute graph alloc");

	size_t s = ggml_gallocr_get_buffer_size(C->allocr, 0);
	assert( s < (size_t)1024*1024*1024*1024 );
	C->info.mem_total = s;
	C->info.mem_compute = C->info.mem_total > C->info.mem_params ?
		C->info.mem_total - C->info.mem_params : 0;

#else
	C->bkbuf = ggml_backend_alloc_ctx_tensors(C->cp, C->backend);
	if (!C->bkbuf)
		ERROR_LOG(-1, "could not allocate parameter tensors");

	ggml_backend_buffer_set_usage(C->bkbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

	ggml_backend_t bk_list[2] = {C->backend, C->backend2};
	int n_bk = 1 + !!C->backend2;
	C->sched = ggml_backend_sched_new(bk_list, NULL, n_bk, C->graph->size, false);
	if (!C->sched) ERROR_LOG(-1, "ggml_backend_sched_new");

	//if (!ggml_backend_sched_reserve(C->sched, C->graph))
	//	ERROR_LOG(-1, "ggml_backend_sched_reserve");
	
	if (!ggml_backend_sched_alloc_graph(C->sched, C->graph))
		ERROR_LOG(-1, "ggml_backend_sched_alloc_graph");

	mllog_debug("ggml sched splits:%d copies:%d",
		ggml_backend_sched_get_n_splits(C->sched),
		ggml_backend_sched_get_n_copies(C->sched) );

	C->info.mem_compute = 0;
	for (int i=0; i<n_bk; ++i) {
		size_t s = ggml_backend_sched_get_buffer_size(C->sched, bk_list[i]);
		mllog_debug("%s compute memory: %zu", ggml_backend_name(bk_list[i]), s);
		// Count only the main backend, usually VRAM
		if (i==0) C->info.mem_compute += s;
	}
	C->info.mem_total = C->info.mem_params + C->info.mem_compute;
#endif
	
	mllog_info("%s memory usage: %.1fMiB (params), %.1fMiB (compute)",
		C->c.name, C->info.mem_params * F_MIB, C->info.mem_compute * F_MIB);

end:
	return R;
}

int mlctx_build_alloc(MLCtx* C, MLTensor* result)
{
	TRYR( mlctx_load_prep(C) );
	TRYR( mlctx_build(C, result) );
	TRYR( mlctx_alloc(C) );
	return 1;
}

int tstore_tensor_read(TSTensorEntry* S, struct ggml_tensor* t)
{
	int R=1;

	// Check shape
	int i=GGML_MAX_DIMS;
	//if (S->shape_n <= GGML_MAX_DIMS) {  //TODO
	//	for (i=0; i<S->shape_n; ++i)
	//		if (S->shape[i] != t->ne[i]) break;
	//}
	//if (i != S->shape_n)
	if (ggml_nelements(t) != tstore_tensor_count(S))
		ERROR_LOG(-1, "wrong shape (%u): %ux%ux%ux%u -> "
			"%"PRId64"x%"PRId64"x%"PRId64"x%"PRId64, i,
			S->shape[0], S->shape[1], S->shape[2], S->shape[3],
			t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
	
	// Data type
	int target = tstore_dtype_from_ggml(t->type);
	if (target < 0) ERROR_LOG(-1, "unsupported tensor ggml type %s",
		ggml_type_name(t->type));

	TSTensorData td={0};
	TRY( R = tstore_tensor_data_get(S, target, 0, &td) );
	ggml_backend_tensor_set(t, td.data, 0, td.size);
	tstore_tdata_free(&td);

end:
	return R;
}

int mlctx_tstore_load(MLCtx* C, TensorStore* ts)
{
	int R=1, r;
	
	mllog_info("%s loading params...", C->c.name);
	double t = timing_time();

	vec_forrp(MLCtxTensor, C->tensors, p)
	{
		if (!(p->tensor && p->tensor->op == GGML_OP_NONE)) continue;

		TSTensorEntry *e = tstore_tensor_getk(ts, p->key);
		if (!e) ERROR_LOG(-1, "tensor '%s' not found", id_str(p->key));
		
		mllog_debug2("loading tensor '%s'", id_str(p->key));
		TRY_LOG(r = tstore_tensor_read(e, p->tensor),
			"could not read tensor '%s'", id_str(p->key));
		C->info.n_conv += (r == TSTDG_R_CONVERT);
	}

	C->info.t_load = timing_time() - t;
	mllog_info("%s params loaded (converted: %u) {%.3fs}",
		C->c.name, C->info.n_conv, C->info.t_load);

end:
	return R;
}

int mlctx_compute(MLCtx* C)
{
	int R=1;

	mllog_info("%s compute", C->c.name);
	double t = timing_time();
#if !USE_GGML_SCHED
	int r = ggml_backend_graph_compute(C->backend, C->graph);
#else
	int r = ggml_backend_sched_graph_compute(C->sched, C->graph);
#endif
	C->info.t_compute = timing_time() - t;
	C->info.n_compute++;
	if (r) ERROR_LOG(-1, "ggml compute: %d", r);
	mllog_info("%s done {%.3fs}", C->c.name, C->info.t_compute);

end:
	return R;
}

int mlctx_prep(MLCtx* C)
{
	MLTensor *result = vec_last(C->tensors,0).tensor;
	if (C->tprefix) mlctx_tensor_add(C, C->tprefix, result);
	TRYR( mlctx_build_alloc(C, result) );
	TRYR( mlctx_tstore_load(C, C->tstore) );
	return 1;
}

int mlctx_run_(MLCtx* C, LocalTensor* out, const LocalTensor** inputs)
{
	int R=1;

	TRY( mlctx_prep(C) );

	vec_for(C->inputs,i,0) {
		if (!inputs[i]) break;
		ltensor_to_backend(inputs[i], C->inputs[i]);
	}

	TRY( mlctx_compute(C) );

	if (out) {
		MLTensor *result = vec_last(C->tensors,0).tensor;
		ltensor_from_backend(out, result);
	}

end:
	mlctx_free(C);
	return R;
}

void mlctx_block_graph_dump(const MLCtx* C, Stream* out)
{
	const MLCtxTensor ** stack=NULL;  //vector
	vec_forrp(const MLCtxTensor, C->tensors, p) {
		if (p->name == MLB_NAME_BLOCK_BEGIN) {
			if (!vec_count(stack)) {
				stream_str_put(out, "ERROR INVALID ML BLOCK GRAPH\n");
				goto end;
			}
			vec_pop(stack);
		}
		else {
			vec_for(stack,i,0) stream_str_put(out, "  ");  //indent

			const char *name;
			if (p->name == MLB_NAME_SPLIT) name = "SPLIT";
			else name = id_str(p->name);

			stream_printf(out, "%s: %s " GGML_TYPESHAPE_FMT "\n",
				name, ggml_op_name(p->tensor->op),
					GGML_TYPESHAPE_ARGS(p->tensor) );

			if (p->tensor->op != GGML_OP_NONE && p->name != MLB_NAME_SPLIT)  //block
				vec_push(stack, p);
		}
	}
end:
	vec_free(stack);
}

int mlctx_block_graph_dump_path(const MLCtx* C, const char* path)
{
	int R=1;
	Stream stm={0};
	TRY_LOG(stream_open_file(&stm, path, SOF_CREATE),
		"could not open '%s'", path);
	mlctx_block_graph_dump(C, &stm);
end:
	stream_close(&stm, 0);
	return R;
}
