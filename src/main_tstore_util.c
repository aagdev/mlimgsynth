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

#define F_MIB  (1.0 / (1024.0*1024.0))
#define F_GIB  (1.0 / (1024.0*1024.0*1024.0))

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

typedef struct TStoreUtil {
	TensorStore sti;
	Stream si, so;
	StringStore ss;

	struct {
		const char *cmd, *path_in, *path_out, *tname, *s_dtype;
		int n_rep;
	} c;
} TStoreUtil;

// tsu = tensor store utility
void tsu_free(TStoreUtil* S)
{
	strsto_free(&S->ss);
	tstore_free(&S->sti);
	stream_close(&S->so, 0);
	stream_close(&S->si, 0);
}

// Load options from command line arguments
int tsu_argv_load(TStoreUtil* S, int argc, char* argv[])
{
	int R=1;

	if (argc <= 1) return 0;
	
	// Defaults
	IFNPOSSET(S->c.n_rep, 4);

	int i, j;
	for (i=1; i<argc; ++i) {
		char * arg = argv[i];
		if (arg[0] == '-') {
			char opt;
			for (j=1; (opt = arg[j]); ++j) {
				char * next = (i+1 < argc) ? argv[i+1] : "";
				switch (opt) {
				case 'i':  S->c.path_in  = next; i++; break;
				case 'o':  S->c.path_out = next; i++; break;
				case 't':  S->c.tname = next; i++; break;
				case 'T':  S->c.s_dtype = next; i++; break;
				case 'n':  S->c.n_rep = atoi(next); i++; break;
				case 'q':  log_level_inc(-LOG_LVL_STEP); break;
				case 'v':  log_level_inc(+LOG_LVL_STEP); break;
				case 'd':  log_level_set(LOG_LVL_DEBUG); break;
				case 'h':
					return 0;
				default:
					ERROR_LOG(-1, "Unknown option '%c'", opt);
				}
			}
		}
		else if (!S->c.cmd) {
			S->c.cmd = arg;
		}
		else {
			ERROR_LOG(-1, "Excess of arguments");
		}
	}

end:
	return R;
}

// Initialize after configuration
int tsu_setup(TStoreUtil* S)
{
	int R=1, r;

	S->sti.ss = &S->ss;

	if (!strcmp(S->c.path_in, "-")) {
		r = stream_open_std(&S->si, STREAM_STD_IN, SOF_READ);
		TRY_LOG(r, "Could not open stdin");
	}
	else if (S->c.path_in) {
		r = stream_open_file(&S->si, S->c.path_in, SOF_READ | SOF_MMAP);
		TRY_LOG(r, "Could not open '%s'", S->c.path_in );
	}
	else
		ERROR_LOG(-1, "Input not set (use -i)");

	if (S->c.path_out) {
		r = stream_open_file(&S->so, S->c.path_out, SOF_CREATE);
		TRY_LOG(r, "Could not open '%s'", S->c.path_out );
	} else {
		r = stream_open_std(&S->so, STREAM_STD_OUT, SOF_WRITE);
		TRY_LOG(r, "Could not open stdout");
	}

	log_debug("Loading...");
	double t = timing_time();
	TRY( tstore_read(&S->sti, &S->si, NULL, NULL) );
	t = timing_time() - t;
	log_info("Load header {%.3fms}", t*1e3);

end:
	return R;
}

// Convert all floating point tensors to an specific type
int tsu_dtype_convert(TStoreUtil* S)
{
	int R=1;
	TensorStore sto={ .ss=&S->ss };
	TSTensorData td={0};

	if (!S->c.s_dtype)
		ERROR_LOG(-1, "use -T to set the target tensor type");
	
	int dtype = tstore_dtype_fromz(S->c.s_dtype);
	if (!(dtype > 0))
		ERROR_LOG(-1, "unknown target tensor type '%s'", S->c.s_dtype);

#define TFILTER(E) \
	((E)->dtype == TS_DTYPE_F64 || \
	 (E)->dtype == TS_DTYPE_F32 || \
	 (E)->dtype == TS_DTYPE_F16 )

	tstore_copy_from(&sto, &S->sti);
	vec_forp(TSTensorEntry, sto.tensors, e, 0) {
		if (!TFILTER(e)) continue;
		e->dtype = dtype;
	}

	double t = timing_time();
	
	extern const TensorStoreFormat ts_cls_safet;  //TODO: option
	TRY( tstore_write(&sto, &S->so, &ts_cls_safet, NULL) );

	unsigned n_tensor=0, n_conv=0;
	vec_for(sto.tensors, i, 0) {
		TSTensorEntry *ti = &S->sti.tensors[i],
		              *to = &sto.tensors[i];
		
		log_debug("tensor '%s' %s -> %s", strsto_get(&S->ss, ti->key).b,
			tstore_dtype_str(ti->dtype),
			tstore_dtype_str(to->dtype));
		TRY( tstore_tensor_data_get(ti, to->dtype, 0, &td) );

		TRY_LOG( stream_seek(&S->so, to->offset, 0), "output seek" );
		TRY_LOG( stream_write_chk(&S->so, td.size, td.data), "write" );

		n_tensor++;
		n_conv += to->dtype != ti->dtype;
	}

	t = timing_time() - t;
	uint64_t osz = stream_pos_get(&S->so);
	log_info("Conversion done: %u tensors, %u converted, %.2fGiB {%.3fs}",
		n_tensor, n_conv, osz*F_GIB, t);

end:
	if (R<0) log_error("convert at %zu", stream_pos_get(&S->so));
	tstore_tdata_free(&td);
	tstore_free(&sto);
	return R;
#undef TFILTER
}

// Check that all tensors can be read and report the speed
int tsu_bench(TStoreUtil* S)
{
	int R=1;
	double tm = timing_time();
	size_t sz=0;
	uint32_t chksum=0;
	const TSTensorEntry *t=NULL;
	const char *tname=NULL;

	vec_for(S->sti.tensors, i, 0)
	{
		t = &S->sti.tensors[i];
		tname = strsto_get(&S->ss, t->key).b;
		
		log_debug2("%s %"PRIu64" %"PRIu64, tname, t->offset, t->size);
		TRY_LOG( stream_seek(t->stm, t->offset, 0), "seek to %"PRIu64, t->offset);
		if (!stream_mmap_is(t->stm))
			TRY_LOG( stream_read_prep(t->stm, t->size), "input read" );
		
		const uint32_t *cur = stream_buffer_get(t->stm, NULL), *end=cur+t->size/4;
		for (; cur<end; ++cur) chksum += *cur;
		sz += t->size;
	}

	tm = timing_time() - tm;
	log_info("Done 0x%08X %u tensors %.3fs %.2fGiB %.2fGiB/s",
		chksum, vec_count(S->sti.tensors), tm, sz*F_GIB, sz*F_GIB/tm);

end:
	if (R<0 && t)
		log_error("tensor '%s' at %"PRIu64" of size %"PRIu64,
			tname, t->offset, t->size);
	return R;
}

// Calculate checksums of the data of all tensors
int tsu_checksum(TStoreUtil* S)
{
	int R=1;
	double tm = timing_time();
	size_t sz=0;
	uint32_t chksum_tot=0;
	const TSTensorEntry *t=NULL;
	const char *tname=NULL;

	vec_for(S->sti.tensors, i, 0)
	{
		t = &S->sti.tensors[i];
		tname = strsto_get(&S->ss, t->key).b;
		
		log_debug2("%s %"PRIu64" %"PRIu64, tname, t->offset, t->size);
		TRY_LOG( stream_seek(t->stm, t->offset, 0), "seek to %"PRIu64, t->offset);
		TRY_LOG( stream_read_prep(t->stm, t->size), "input read" );
		
		uint32_t chksum=0;
		const uint32_t *cur = stream_buffer_get(t->stm, NULL), *end=cur+t->size/4;
		for (; cur<end; ++cur) chksum += *cur;
		chksum_tot += chksum;
		sz += t->size;
		
		stream_printf(&S->so, "%s: 0x%08X\n", tname, chksum);
		stream_flush(&S->so);  // This slightly slows down the process
	}
	stream_printf(&S->so, "TOTAL: 0x%08X\n", chksum_tot);
	stream_flush(&S->so);

	tm = timing_time() - tm;
	log_info("Done %u tensors {%.3fs %.2fGiB}",
		vec_count(S->sti.tensors), tm, sz*F_GIB);

end:
	if (R<0 && t)
		log_error("tensor '%s' at %"PRIu64" of size %"PRIu64,
			tname, t->offset, t->size);
	return R;
}

// Extract one tensor
int tsu_tensor_extract(TStoreUtil* S)
{
	int R=1;
		
	if (!S->c.tname)
		ERROR_LOG(-1, "use -t to set the tensor name to extract");
	
	TSTensorEntry *t = tstore_tensor_get(&S->sti, S->c.tname);
	if (!t) ERROR_LOG(-1, "could find tensor '%s'", S->c.tname);

	TRY_LOG( stream_seek(t->stm, t->offset, 0), "seek to %"PRIu64, t->offset);
	TRY_LOG( stream_read_prep(t->stm, t->size), "input read" ); 
	const uint8_t *cur = stream_buffer_get(t->stm, NULL), *end=cur+t->size;

	stream_printf(&S->so, "TENSOR %s", tstore_dtype_str(t->dtype));
	for (unsigned i=0; i<t->shape_n; ++i)
		stream_printf(&S->so, " %d", t->shape[i]);
	stream_char_put(&S->so, '\n');
	stream_write_chk(&S->so, end-cur, cur);
	stream_flush(&S->so);

end:
	return R;
}

int main(int argc, char* argv[])
{
	int R=0, r;
	TStoreUtil tsu={0};

	TRY( r = tsu_argv_load(&tsu, argc, argv) );
	if (!r) {
		puts(help_string);
		return 0;
	}

	TRY( tsu_setup(&tsu) );

#define IF_CMD(NAME) \
	else if (!strcmp(tsu.c.cmd, NAME))

	if (!tsu.c.cmd) {
		puts("No command. Use -h for help.");
	}
	IF_CMD("info") {
		tstore_info_dump(&tsu.sti, &tsu.so);
	}
	IF_CMD("bench") {
		for (unsigned i=0; i<tsu.c.n_rep; ++i)
			TRY( tsu_bench(&tsu) );
	}
	IF_CMD("checksum") {
		TRY( tsu_checksum(&tsu) );
	}
	IF_CMD("convert") {
		TRY( tsu_dtype_convert(&tsu) );
	}
	IF_CMD("extract") {
		TRY( tsu_tensor_extract(&tsu) );
	}
	else {
		ERROR_LOG(-1, "Unknown command '%s'", tsu.c.cmd);
	}

end:
	if (R<0) log_error("error exit: %x", -R);
	tsu_free(&tsu);
	return -R;
}
