/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * Utility to work with model/tensor files.
 */
#include "ccommon/timing.h"
#include "ccommon/logging.h"
#include "ccommon/stream.h"
#include "safetensors.h"
#include <inttypes.h>

#define IDS_IMPLEMENTATION
#include "ids.h"

typedef struct Options {
	const char *path_in, *path_out, *tname;
	int cmd;
} Options;

const char help_string[] =
		"Usage: st-util [OPTIONS] [COMMAND]\n"
		"Utility for safetensors format.\n"
		"\n"
		"Commands:\n"
		"  info          Dump information.\n"
		"  checksum      Calculate tensors checksums.\n"
		"  convert16     Convert all float tensors to F16.\n"
		"  extract NAME  Extract one tensor.\n"
		"\n"
		"Options:\n"
		"  -i          Input file (- for stdin)\n"
		"  -o          Output file (default stdout)\n"
		"  -t TEXT     Tensor name to extract.\n"
		"\n"
		"  -q          Quiet: reduces information output\n"
		"  -v          Verbose: increases information output\n"
		"  -d          Enables debug output\n"
		"  -h          Print this message\n"
		;

int options_load(Options* opts, int argc, char* argv[])
{
	if (argc <= 1) {
		puts(help_string);
		return 1;
	}

	int i, j;
	for (i=1; i<argc; ++i) {
		char * arg = argv[i];
		if (arg[0] == '-') {
			char opt;
			for (j=1; (opt = arg[j]); ++j) {
				char * next = (i+1 < argc) ? argv[i+1] : "";
				switch (opt) {
				case 'i':  opts->path_in  = next; i++; break;
				case 'o':  opts->path_out = next; i++; break;
				case 't':  opts->tname = next; i++; break;
				case 'q':  log_level_inc(-LOG_LVL_STEP); break;
				case 'v':  log_level_inc(+LOG_LVL_STEP); break;
				case 'd':  log_level_set(LOG_LVL_DEBUG); break;
				case 'h':
					puts(help_string);
					return 1;
				default:
					log_error("Unknown option '%c'", opt);
					return 1;
				}
			}
		}
		else if (!opts->cmd) {
			opts->cmd = id_fromz(arg);
		}
		else {
			log_error("Excess of arguments");
			return 1;
		}
	}

	return 0;
}

int stream_copyn(Stream* dst, Stream* src, size_t size)
{
	while (size) {
		if (stream_read_prep(src, 0) < 1) return -1;
		if (stream_write_prep(dst, 0) < 0) return -1;
		const uint8_t *send, *scur = stream_buffer_get(src, &send);
		uint8_t *dend, *dcur = stream_buffer_get(dst, &dend);
		size_t bsz = size;
		MINSET(bsz, send-scur);
		MINSET(bsz, dend-dcur);
		memcpy(dcur, scur, bsz);
		stream_commit(dst, dcur+bsz);
		stream_commit(src, scur+bsz);
		size -= bsz;
	}
	return 1;
}

int stream_copyn_f64_f16(Stream* dst, Stream* src, size_t size)
{
	size_t count = size / 8;
	if (count * 8 != size) { log_error("invalid input tensor size"); return -1; }
	while (count) {
		if (stream_read_prep(src, 0)  < 8) { log_error("read");  return -1; }
		if (stream_write_prep(dst, 0) < 0) { log_error("write"); return -1; }
		const uint8_t *send, *scur = stream_buffer_get(src, (void**)&send);
		uint8_t *dend, *dcur = stream_buffer_get(dst, (void**)&dend);
		size_t cnt = count;
		MINSET(cnt, (send-scur)/8);
		MINSET(cnt, (dend-dcur)/2);
		for (size_t i=0; i<cnt; ++i) {
			double sf;
			memcpy(&sf, scur, 8);
			_Float16 df = sf;
			memcpy(dcur, &df, 2);
			scur += 8;
			dcur += 2;
		}
		stream_commit(dst, dcur);
		stream_commit(src, scur);
		count -= cnt;
	}
	return 1;
}

int stream_copyn_f32_f16(Stream* dst, Stream* src, size_t size)
{
	size_t count = size / 4;
	if (count * 4 != size) { log_error("invalid input tensor size"); return -1; }
	while (count) {
		if (stream_read_prep(src, 0)  < 4) { log_error("read");  return -1; }
		if (stream_write_prep(dst, 0) < 0) { log_error("write"); return -1; }
		const uint8_t *send, *scur = stream_buffer_get(src, (void**)&send);
		uint8_t *dend, *dcur = stream_buffer_get(dst, (void**)&dend);
		size_t cnt = count;
		MINSET(cnt, (send-scur)/4);
		MINSET(cnt, (dend-dcur)/2);
		for (size_t i=0; i<cnt; ++i) {
			float sf;
			memcpy(&sf, scur, 4);
			_Float16 df = sf;
			memcpy(dcur, &df, 2);
			scur += 4;
			dcur += 2;
		}
		stream_commit(dst, dcur);
		stream_commit(src, scur);
		count -= cnt;
	}
	return 1;
}

int stream_chksum(Stream* stm, size_t nbytes, uint32_t* pcs)
{
	int R=1;
	uint32_t cs = *pcs;
	const uint8_t *end=NULL, *cur=NULL;
	while (nbytes && (cur = stream_read_buffer(stm, &end)) < end) {
		if (cur+nbytes < end) end = cur + nbytes;
		nbytes -= end-cur;
		for (; cur<end; ++cur) cs += *cur;
		stream_commit(stm, cur);
	}
	if (nbytes && cur == NULL)
		ERROR_LOG(-1, "read at %"PRIu64, stream_pos_get(stm));
	*pcs = cs;
end:
	return R;
}

int convert16(Stream* so, const TensorStore* sti)
{
	int R=1;
	TensorStore sto={ .ss=&g_ss };

	tstore_copy_from(&sto, sti);
	vec_forp(TSTensorEntry, sto.tensors, e, 0) {
		if (e->dtype == TS_DTYPE_F64 || e->dtype == TS_DTYPE_F32)
			e->dtype = TS_DTYPE_F16;
	}
	
	TRY( safet_save_head(&sto, so, NULL) );

	vec_for(sto.tensors,i,0) {
		const TSTensorEntry *ti=&sti->tensors[i], *to=&sto.tensors[i];
		Stream *si = ti->stm;
		TRY_LOG( stream_seek(si, ti->offset, 0), "input seek" );
		TRY_LOG( stream_seek(so, to->offset, 0), "output seek" );
		if (ti->dtype == to->dtype) {
			log_debug2("%s: copy", id_str(ti->key));
			assert( ti->size == to->size );
			TRY( stream_copyn(so, si, ti->size) );
		} else if (ti->dtype == TS_DTYPE_F64) {
			log_debug2("%s: f64 -> f16", id_str(ti->key));
			TRY( stream_copyn_f64_f16(so, si, ti->size) );
		} else if (ti->dtype == TS_DTYPE_F32) {
			log_debug2("%s: f32 -> f16", id_str(ti->key));
			TRY( stream_copyn_f32_f16(so, si, ti->size) );
		} else {
			ERROR_LOG(-1, "%s: conversion not implemented", id_str(ti->key));
		}
	}

	uint64_t osz = stream_pos_get(so);
	assert(osz == sto.os_end);
	log_info("Convertion done. Output size: %llu", (unsigned long long)osz);

end:
	if (R<0) log_error("convert at %llu", (unsigned long long)stream_pos_get(so));
	tstore_free(&sto);
	return R;
}

// Check that all tensors can be read
int checksum(const TensorStore* sti, Stream* out)
{
	int R=1;
	uint32_t chksum_tot=0;
	const TSTensorEntry *t=NULL;
	vec_for(sti->tensors, i, 0)
	{
		t = &sti->tensors[i];
		const char *name = id_str(t->key);
		log_debug2("%s %"PRIu64" %"PRIu64, name, t->offset, t->size);
		TRY_LOG( stream_seek(t->stm, t->offset, 0), "seek to %"PRIu64, t->offset);
		uint32_t chksum=0;
		TRY( stream_chksum(t->stm, t->size, &chksum) );
		stream_printf(out, "%s: 0x%08X\n", name, chksum);
		chksum_tot += chksum;
	}
	stream_printf(out, "TOTAL: 0x%08X\n", chksum_tot);
end:
	if (R<0 && t)
		log_error("check error on tensor '%s' at %"PRIu64" of size %"PRIu64,
			id_str(t->key), t->offset, t->size);
	return R;
}

int extract(const TensorStore* ts, const char* tname, Stream* so)
{
	int R=1;
	TSTensorEntry *t = tstore_tensor_get(ts, tname);
	if (!t) ERROR_LOG(-1, "could find tensor '%s'", tname);

	TRY_LOG( stream_seek(t->stm, t->offset, 0), "seek to %"PRIu64, t->offset);

	stream_printf(so, "TENSOR %s", tstore_dtype_str(t->dtype));
	for (unsigned i=0; i<t->shape_n; ++i) stream_printf(so, " %d", t->shape[i]);
	stream_char_put(so, '\n');
	stream_copyn(so, t->stm, t->size);

end:
	return R;
}

int main(int argc, char* argv[])
{
	int R=0, r;
	Options opts={0};
	Stream si={0}, so={0};
	TensorStore sti={ .ss=&g_ss };

	ids_init();

	if (options_load(&opts, argc, argv)) return 11;

	if (!strcmp(opts.path_in, "-")) {
		r = stream_open_std(&si, STREAM_STD_IN, SOF_READ);
		TRY_LOG(r, "Could not open stdin");
	}
	else if (opts.path_in) {
		r = stream_open_file(&si, opts.path_in, SOF_READ);
		TRY_LOG(r, "Could not open '%s'", opts.path_in );
	}
	else
		ERROR_LOG(-1, "Input not set");

	if (opts.path_out) {
		r = stream_open_file(&so, opts.path_out, SOF_CREATE);
		TRY_LOG(r, "Could not open '%s'", opts.path_out );
	} else {
		r = stream_open_std(&so, STREAM_STD_OUT, SOF_WRITE);
		TRY_LOG(r, "Could not open stdout");
	}

	log_debug("Loading...");
	TRY( safet_load_head(&sti, &si, NULL) );

	if (!opts.cmd) ;
	else if (opts.cmd == id_fromz("info")) {
		tstore_info_dump(&sti, &so);
	}
	else if (opts.cmd == id_fromz("checksum")) {
		TRY( checksum(&sti, &so) );
	}
	else if (opts.cmd == id_fromz("convert16")) {
		TRY( convert16(&so, &sti) );
	}
	else if (opts.cmd == id_fromz("extract")) {
		if (!opts.tname) ERROR_LOG(-1, "tensor name not set");
		TRY( extract(&sti, opts.tname, &so) );
	}
	else {
		ERROR_LOG(-1, "Unknown command '%s'", id_str(opts.cmd));
	}

end:
	if (R<0) log_error("error exit: %d", R);
	tstore_free(&sti);
	stream_close(&so, 0);
	stream_close(&si, 0);
	return -R;
}
