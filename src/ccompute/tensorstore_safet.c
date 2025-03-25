/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 */
#include "tensorstore_safet.h"
#include "ccommon/logging.h"
#include "ccommon/structio_json.h"
#include <assert.h>
#include <inttypes.h>

#define id_str(X)  strsto_get(S->ss, X).b

#define SAFET_ALIGNMENT 32

static
uint64_t safet_align(uint64_t offset)
{
	return ((offset + SAFET_ALIGNMENT-1) / SAFET_ALIGNMENT) * SAFET_ALIGNMENT;
}

static
void str_to_lower(unsigned n, char* buf)
{
	for (unsigned i=0; i<n; ++i)
		if ('A' <= buf[i] && buf[i] <= 'Z')
			buf[i] += 'a' - 'A';
}

static
int safet_read_meta(TensorStore* S, StioStream *sio, DynStr* ptmps)
{
	int R=1, r;
	StioItem itm;

	TRY( stio_read(sio, &itm, 0) );
	TRY( stio_item_open_check(&itm, STIO_T_VALUE, ANY_T_MAP) );
	
	while (1) {
		TRY( r = stio_read(sio, &itm, 0) );
		if (r == STIO_R_CTX_END) break;
		TRY( stio_item_type_check(&itm, STIO_T_KEY, ANY_T_STRING) );

		dstr_copy(*ptmps, itm.value.len, itm.value.p.cp);
		
		TRY( r = stio_read(sio, &itm, 0) );
		r = stio_item_type_check(&itm, STIO_T_VALUE, ANY_T_STRING);
		if (r == STIO_E_INCOMPLETE) {
			log_warning("metadata entry '%s' too large, skipping", *ptmps);
			while (1) {
				TRY( r = stio_read_chunk(sio, &itm) );
				if (r == STIO_R_CTX_END) break;
			}
			continue;
		}
		TRY(r);

		TRY( tstore_meta_adds(S, *ptmps, itm.value.p.cp) );
	}

end:
	if (R<0) log_error("safetensors metadata: %x", -R);
	return R;
}

static
int safet_read_tensor(StioStream *sio, TSTensorEntry* entry, const char* name)
{
	int R=1, r;
	TSTensorEntry e={0};
	StioItem itm;

	TRY( stio_read(sio, &itm, 0) );
	TRY( stio_item_open_check(&itm, STIO_T_VALUE, ANY_T_MAP) );
	
	while (1) {
		TRY( r = stio_read(sio, &itm, 0) );
		if (r == STIO_R_CTX_END) break;
		TRY( stio_item_type_check(&itm, STIO_T_KEY, ANY_T_STRING) );

		if (!strcmp(itm.value.p.cp, "dtype")) {
			TRY( r = stio_read(sio, &itm, 0) );
			TRY( stio_item_type_check(&itm, STIO_T_VALUE, ANY_T_STRING) );
			
			str_to_lower(itm.value.len, itm.value.p.cp);
			int dt = tstore_dtype_fromz(itm.value.p.cp); 
			if (!(dt > 0))
				ERROR_LOG(TS_E_DTYPE, "unknown dtype '%s'", itm.value.p.cp);
			e.dtype = dt;
		}
		else if (!strcmp(itm.value.p.cp, "shape")) {
			TRY( stio_read(sio, &itm, 0) );
			TRY( stio_item_open_check(&itm, STIO_T_VALUE, ANY_T_ARRAY) );
			
			unsigned i=0;
			while (1) {
				TRY( r = stio_read(sio, &itm, 0) );
				if (r == STIO_R_CTX_END) break;
				if (i == COUNTOF(e.shape))
					ERROR_LOG(-1, "tensor shape too large");
				TRY( stio_item_type_check(&itm, STIO_T_VALUE, 0) );
				e.shape[i] = anys_uint32_get(&itm.value);
				i++;
			}
			e.shape_n = i;
		}
		else if (!strcmp(itm.value.p.cp, "data_offsets")) {
			TRY( stio_read(sio, &itm, 0) );
			TRY( stio_item_open_check(&itm, STIO_T_VALUE, ANY_T_ARRAY) );
			
			TRY( r = stio_read(sio, &itm, 0) );
			if (r == STIO_R_CTX_END) ERROR_LOG(-1, "tensor data_offsets too short");
			TRY( stio_item_type_check(&itm, STIO_T_VALUE, 0) );
			e.offset = anys_uint64_get(&itm.value);
			
			TRY( r = stio_read(sio, &itm, 0) );
			if (r == STIO_R_CTX_END) ERROR_LOG(-1, "tensor data_offsets too short");
			TRY( stio_item_type_check(&itm, STIO_T_VALUE, 0) );
			e.size = anys_uint64_get(&itm.value);
			
			TRY( r = stio_read(sio, &itm, 0) );
			if (r != STIO_R_CTX_END) ERROR_LOG(-1, "tensor data_offsets too long");
		}
		else {
			ERROR_LOG(TS_E_FORMAT, "unknown tensor key '%s'", itm.value.p.cp);
		}
	}
	
	if (!(e.size >= e.offset))
		ERROR_LOG(TS_E_OVERFLOW,
			"invalid offsets [%"PRIu64", %"PRIu64"]", e.offset, e.size);
	
	e.size -= e.offset;

	if (tstore_tensor_size(&e) != e.size)
		ERROR_LOG(TS_E_FORMAT, "invalid size %"PRIu64" for %s %ux%ux%ux%u",
			e.size,	tstore_dtype_str(e.dtype),
			e.shape[0], e.shape[1], e.shape[2], e.shape[3] );

	// Reverse shape and fill with ones
	for (unsigned i=0; i<e.shape_n/2; ++i)
		ccSWAPT(unsigned, e.shape[i], e.shape[e.shape_n-1-i]);
	for (unsigned i=e.shape_n; i<COUNTOF(e.shape); ++i)
		e.shape[i] = 1;

	e.stm = sio->s;
	*entry = e;

end:
	if (R<0) log_error("safetensors tensor '%s': %x", name, -R);
	return R;
}

int tstore_read_safet(TensorStore* S, Stream* stm, TSCallback* cb)
{
	int R=1, r;
	StioStream sio={0};
	StioItem itm;
	DynStr name=NULL;
	char buffer[2048];

	uint64_t os_data;
	if (stream_read_var(stm, os_data) < 0)
		ERROR_LOG(TS_E_READ, "could not read");
	os_data += 8;

	if (stream_char_get(stm) != '{')
		ERROR_LOG(TS_E_FORMAT, "invalid file");
	stream_unget(stm, 1);
	
	if (os_data > 0xffffff)
		ERROR_LOG(TS_E_OVERFLOW, "header too big: %"PRIu64"", os_data);

	TRY( stio_init(&sio, stm, &stio_class_json, 0, sizeof(buffer), buffer) );
	
	TRY( stio_read(&sio, &itm, 0) );
	TRY( stio_item_open_check(&itm, STIO_T_VALUE, ANY_T_MAP) );

	unsigned n_meta=0, n_tensor=0;
	while (1) {
		TRY( r = stio_read(&sio, &itm, 0) );
		if (r == STIO_R_CTX_END) break;
		TRY( stio_item_type_check(&itm, STIO_T_KEY, ANY_T_STRING) );

		if (!strcmp(itm.value.p.cp, "__metadata__")) {
			TRY( safet_read_meta(S, &sio, &name) );
			n_meta++;
		} else {
			dstr_copy(name, itm.value.len, itm.value.p.cp);
			TSTensorEntry e={0};
			TRY( safet_read_tensor(&sio, &e, name) );
			e.offset += os_data;
			TRY( r = tstore_cb_call(cb, S, &e, &name) );
			if (r > 0) {
				TRY( tstore_tensor_add(S, name, &e) );
				n_tensor++;
			}
		}
	}

	log_debug("safetensors n_meta:%u n_tensor:%u", n_meta, n_tensor);

end:
	if (R<0) log_error("safetensors read at position 0x%zx: %x", stream_pos_get(stm), -R);
	dstr_free(name);
	return R;	
}

static inline
int stream_str_put_escape(Stream* stm, const char* str)
{
	while (*str) {
		char *end, *cur = stream_write_buffer(stm, &end);
		if (!cur) return -1;
		while (*str && cur+1 < end) {
			if (*str == '"') *cur++ = '\\';
			*cur++ = *str++;
		}
		stream_commit(stm, cur);
	}
	return 1;
}

int tstore_write_safet(TensorStore* S, Stream* stm, TSCallback* cb)
{
	int R=0;
	DynStr tmps=NULL;
	uint64_t offset=0, os_data=0, os_end=0;

	if (stream_write_var(stm, offset) < 0)  //placeholder
		ERROR_LOG(TS_E_WRITE, "safetensors could not write");
	
	stream_char_put(stm, '{');
	bool first=true;

	// Meta data
	if (vec_count(S->meta)) {
		stream_str_put(stm, "\"__metadata__\":{");
		vec_for(S->meta,i,0) {
			if (i) stream_char_put(stm, ',');
			stream_char_put(stm, '"');
			stream_str_put_escape(stm, id_str(S->meta[i].key));
			stream_str_put(stm, "\":\"");
			if (S->meta[i].value.t != ANY_T_STRING)
				ERROR_LOG(TS_E_METADATA, "metadata value can only be string");
			stream_str_put_escape(stm, S->meta[i].value.p.cp);
			stream_char_put(stm, '"');
		}
		stream_char_put(stm, '}');
		first=false;
	}

	// Tensors
	vec_forp(TSTensorEntry, S->tensors, e, 0) {
		dstr_resize(tmps, 0);
		if (tstore_cb_call(cb, S, e, &tmps) < 0) continue;
		const char *name = dstr_empty(tmps) ? id_str(e->key) : tmps;

		e->stm = stm;
		e->offset = offset;
		e->size = tstore_tensor_size(e);
		offset += safet_align(e->size);

		if (first) first=false; else stream_char_put(stm, ',');
		stream_char_put(stm, '"');
		stream_str_put_escape(stm, name);
		stream_str_put(stm, "\":{");
		stream_printf(stm, "\"dtype\":\"%s\"", tstore_dtype_str(e->dtype));
		if (e->shape_n) {
			stream_str_put(stm, ",\"shape\":[");
			for (int i=(int)e->shape_n-1, first=1; i>=0; --i) {
				if (first) first=0;
				else stream_char_put(stm, ',');
				stream_printf(stm, "%u", e->shape[i]);
			}
			stream_char_put(stm, ']');
		}
		stream_printf(stm, ",\"data_offsets\":[%"PRIu64",%"PRIu64"]",
			e->offset, (e->offset + e->size));
		stream_char_put(stm, '}');
	}
	os_end = offset;
		
	stream_char_put(stm, '}');

	// Header size / data offset
	os_data = stream_pos_get(stm);
	TRY( stream_seek(stm, 0, 0) );

	// Pad to alignment
	os_data = safet_align(os_data);

	offset = os_data - 8;
	TRY( stream_write_var(stm, offset) );
	TRY( stream_seek(stm, os_data, 0) );  // Position ready to write data
	
	vec_forp(TSTensorEntry, S->tensors, e, 0) e->offset += os_data;
	os_end += os_data;
	
	log_debug("safetensors write: sz_header:%"PRIu64"B sz_total:%"PRIu64"B",
		os_data, os_end);

end:
	if (R<0) log_error("safetensors write: %x", -R);
	dstr_free(tmps);
	return R;
}

int tstore_detect_safet(Stream* stm)
{
	uint8_t *end, *cur = stream_read_buffer(stm, &end);
	if (!(end-cur >= 9)) return 0;
	if (cur[8] != '{') return 0;
	uint64_t offset;
	memcpy(&offset, cur, 8);
	return (2 <= offset && offset <= 0xffffff);
}

const TensorStoreFormat ts_cls_safet = {
	"safetensor", "safetensor",
	tstore_detect_safet,
	tstore_read_safet,
	tstore_write_safet,
};
