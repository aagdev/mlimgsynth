/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 *
 * Utility to work with model/tensor files.
 */
#include "ccommon/timing.h"
#include "ccommon/logging.h"
#include "ccommon/stream.h"
#include "ccompute/tensorstore.h"
#include <inttypes.h>

#define IDS_IMPLEMENTATION
#include "ids.h"

#define F_MIB  (1.0 / (1024.0*1024.0))
#define F_GIB  (1.0 / (1024.0*1024.0*1024.0))

typedef struct Options {
	const char *path_in, *path_out, *tname, *s_dtype;
	int cmd, n_rep;
} Options;

const char help_string[] =
	"Usage: tstore-util [OPTIONS] [COMMAND]\n"
	"Utility to work with model/tensor files.\n"
	"Formats supported: safetensors, GGUF.\n"
	"\n"
	"Commands:\n"
	"  info          Dump information.\n"
	"  bench         Benchmark tensor reading.\n"
	"  checksum      Calculate tensors checksums.\n"
	"  convert       Convert all float tensors to the target type.\n"
	"  extract       Extract one tensor.\n"
	"\n"
	"Options:\n"
	"  -i          Input file (- for stdin)\n"
	"  -o          Output file (default stdout)\n"
	"  -n INT      Number of time to repeat the benchmark.\n"
	"  -t NAME     Tensor name to extract.\n"
	"  -T TYPE     Tensor type for convert.\n"
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
				case 'T':  opts->s_dtype = next; i++; break;
				case 'n':  opts->n_rep = atoi(next); i++; break;
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

int convert(Stream* so, const TensorStore* sti, const char* s_dtype)
{
	int R=1;
	TensorStore sto={ .ss=&g_ss };
	TSTensorData td={0};

	if (!s_dtype)
		ERROR_LOG(-1, "use -T to set the target tensor type");
	
	int dtype = tstore_dtype_fromz(s_dtype);
	if (!(dtype > 0))
		ERROR_LOG(-1, "unknown target tensor type '%s'", s_dtype);

#define TFILTER(E) \
	((E)->dtype == TS_DTYPE_F64 || \
	 (E)->dtype == TS_DTYPE_F32 || \
	 (E)->dtype == TS_DTYPE_F16 )

	tstore_copy_from(&sto, sti);
	vec_forp(TSTensorEntry, sto.tensors, e, 0) {
		if (!TFILTER(e)) continue;
		e->dtype = dtype;
	}

	double t = timing_time();
	
	extern const TensorStoreFormat ts_cls_safet;  //TODO: option
	TRY( tstore_write(&sto, so, &ts_cls_safet) );

	unsigned n_tensor=0, n_conv=0;
	vec_for(sto.tensors,i,0) {
		TSTensorEntry *ti=&sti->tensors[i], *to=&sto.tensors[i];
		
		log_debug("tensor '%s' %s -> %s", id_str(ti->key),
			tstore_dtype_str(ti->dtype),
			tstore_dtype_str(to->dtype));
		TRY( tstore_tensor_data_get(ti, to->dtype, 0, &td) );

		TRY_LOG( stream_seek(so, to->offset, 0), "output seek" );
		TRY_LOG( stream_write_chk(so, td.size, td.data), "write" );

		n_tensor++;
		n_conv += to->dtype != ti->dtype;
	}

	t = timing_time() - t;
	uint64_t osz = stream_pos_get(so);
	log_info("Conversion done: %u tensors, %u converted, %.2fGiB {%.3fs}",
		n_tensor, n_conv, osz*F_GIB, t);

end:
	if (R<0) log_error("convert at %zu", stream_pos_get(so));
	tstore_tdata_free(&td);
	tstore_free(&sto);
	return R;
#undef TFILTER
}

// Check that all tensors can be read
int bench(const TensorStore* sti)
{
	int R=1;
	double tm = timing_time();
	size_t sz=0;
	uint32_t chksum=0;
	const TSTensorEntry *t=NULL;
	vec_for(sti->tensors, i, 0)
	{
		t = &sti->tensors[i];
		const char *name = id_str(t->key);
		
		log_debug2("%s %"PRIu64" %"PRIu64, name, t->offset, t->size);
		TRY_LOG( stream_seek(t->stm, t->offset, 0), "seek to %"PRIu64, t->offset);
		if (!stream_mmap_is(t->stm))
			TRY_LOG( stream_read_prep(t->stm, t->size), "input read" );
		
		const uint32_t *cur = stream_buffer_get(t->stm, NULL), *end=cur+t->size/4;
		for (; cur<end; ++cur) chksum += *cur;
		sz += t->size;
	}

	tm = timing_time() - tm;
	log_info("Done 0x%08X %u tensors %.3fs %.2fGiB %.2fGiB/s",
		chksum, vec_count(sti->tensors), tm, sz*F_GIB, sz*F_GIB/tm);

end:
	if (R<0 && t)
		log_error("tensor '%s' at %"PRIu64" of size %"PRIu64,
			id_str(t->key), t->offset, t->size);
	return R;
}

int checksum(const TensorStore* sti, Stream* out)
{
	int R=1;
	double tm = timing_time();
	size_t sz=0;
	uint32_t chksum_tot=0;
	const TSTensorEntry *t=NULL;
	vec_for(sti->tensors, i, 0)
	{
		t = &sti->tensors[i];
		const char *name = id_str(t->key);
		
		log_debug2("%s %"PRIu64" %"PRIu64, name, t->offset, t->size);
		TRY_LOG( stream_seek(t->stm, t->offset, 0), "seek to %"PRIu64, t->offset);
		TRY_LOG( stream_read_prep(t->stm, t->size), "input read" );
		
		uint32_t chksum=0;
		const uint32_t *cur = stream_buffer_get(t->stm, NULL), *end=cur+t->size/4;
		for (; cur<end; ++cur) chksum += *cur;
		chksum_tot += chksum;
		sz += t->size;
		
		stream_printf(out, "%s: 0x%08X\n", name, chksum);
		stream_flush(out);  // This slightly slows down the process
	}
	stream_printf(out, "TOTAL: 0x%08X\n", chksum_tot);
	stream_flush(out);

	tm = timing_time() - tm;
	log_info("Done %u tensors {%.3fs %.2fGiB}",
		vec_count(sti->tensors), tm, sz*F_GIB);

end:
	if (R<0 && t)
		log_error("tensor '%s' at %"PRIu64" of size %"PRIu64,
			id_str(t->key), t->offset, t->size);
	return R;
}

int extract(const TensorStore* ts, const char* tname, Stream* out)
{
	int R=1;
	TSTensorEntry *t = tstore_tensor_get(ts, tname);
	if (!t) ERROR_LOG(-1, "could find tensor '%s'", tname);

	TRY_LOG( stream_seek(t->stm, t->offset, 0), "seek to %"PRIu64, t->offset);
	TRY_LOG( stream_read_prep(t->stm, t->size), "input read" ); 
	const uint8_t *cur = stream_buffer_get(t->stm, NULL), *end=cur+t->size;

	stream_printf(out, "TENSOR %s", tstore_dtype_str(t->dtype));
	for (unsigned i=0; i<t->shape_n; ++i)
		stream_printf(out, " %d", t->shape[i]);
	stream_char_put(out, '\n');
	stream_write_chk(out, end-cur, cur);
	stream_flush(out);

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
		r = stream_open_file(&si, opts.path_in, SOF_READ | SOF_MMAP);
		TRY_LOG(r, "Could not open '%s'", opts.path_in );
	}
	else
		ERROR_LOG(-1, "Input not set (use -i)");

	if (opts.path_out) {
		r = stream_open_file(&so, opts.path_out, SOF_CREATE);
		TRY_LOG(r, "Could not open '%s'", opts.path_out );
	} else {
		r = stream_open_std(&so, STREAM_STD_OUT, SOF_WRITE);
		TRY_LOG(r, "Could not open stdout");
	}

	log_debug("Loading...");
	//TRY( tstore_safet_load_head(&sti, &si, NULL) );
	double t = timing_time();
	TRY( tstore_read(&sti, &si, NULL) );
	t = timing_time() - t;
	log_info("Load header {%.3fms}", t*1e3);

	if (!opts.cmd) ;
	else if (opts.cmd == id_fromz("info")) {
		tstore_info_dump(&sti, &so);
	}
	else if (opts.cmd == id_fromz("bench")) {
		IFNPOSSET(opts.n_rep, 4);
		for (unsigned i=0; i<opts.n_rep; ++i)
			TRY( bench(&sti) );
	}
	else if (opts.cmd == id_fromz("checksum")) {
		TRY( checksum(&sti, &so) );
	}
	else if (opts.cmd == id_fromz("convert")) {
		TRY( convert(&so, &sti, opts.s_dtype) );
	}
	else if (opts.cmd == id_fromz("extract")) {
		if (!opts.tname) ERROR_LOG(-1, "tensor name not set");
		TRY( extract(&sti, opts.tname, &so) );
	}
	else {
		ERROR_LOG(-1, "Unknown command '%s'", id_str(opts.cmd));
	}

end:
	if (R<0) log_error("error exit: %x", -R);
	tstore_free(&sti);
	stream_close(&so, 0);
	stream_close(&si, 0);
	return -R;
}
