/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: MIT
 */
#include "safetensors.h"
#include "ccommon/logging.h"
#include "ccommon/structio_json.h"
#include "ccommon/bisect.h"
#include "ids.h"
#include <assert.h>
#include <inttypes.h>

int safet_load_meta(TensorStore* S, StioStream *sio)
{
	int R=1, r;
	DynStr key=NULL;
	StioItem itm;

	TRY( stio_read(sio, &itm, 0) );
	TRY( stio_item_open_check(&itm, STIO_T_VALUE, ANY_T_MAP) );
	
	while (1) {
		TRY( r = stio_read(sio, &itm, 0) );
		if (r == STIO_R_CTX_END) break;
		TRY( stio_item_type_check(&itm, STIO_T_KEY, ANY_T_STRING) );

		dstr_copyz(key, itm.value.p.cp);
		
		TRY( r = stio_read(sio, &itm, 0) );
		TRY( stio_item_type_check(&itm, STIO_T_VALUE, ANY_T_STRING) );
		
		tstore_meta_add(S, key, itm.value.p.cp);
	}

	unsigned n = vec_count(S->meta);
	if (n) log_debug("safetensors metadata loaded: %u", n);

end:
	if (R<0) log_error("safetensors metadata");
	dstr_free(key);
	return R;
}

int safet_load_tensor_head(TensorStore* S, StioStream *sio, DynStr* key)
{
	int R=1, r;
	StioItem itm;
	TSTensorEntry e={0};
	bool can_ignore=false;

	TRY( stio_read(sio, &itm, 0) );
	TRY( stio_item_open_check(&itm, STIO_T_VALUE, ANY_T_MAP) );
	
	while (1) {
		TRY( r = stio_read(sio, &itm, 0) );
		if (r == STIO_R_CTX_END) break;
		TRY( stio_item_type_check(&itm, STIO_T_KEY, ANY_T_STRING) );

		if (!strcmp(itm.value.p.cp, "dtype")) {
			TRY( r = stio_read(sio, &itm, 0) );
			TRY( stio_item_type_check(&itm, STIO_T_VALUE, ANY_T_STRING) );
			e.dtype = tstore_dtype_fromz(itm.value.p.cp);
		}
		else if (!strcmp(itm.value.p.cp, "shape")) {
			TRY( stio_read(sio, &itm, 0) );
			TRY( stio_item_open_check(&itm, STIO_T_VALUE, ANY_T_ARRAY) );
			
			unsigned i=0;
			while (1) {
				if (i == COUNTOF(e.shape))
					ERROR_LOG(-1, "tensor shape too large");
				TRY( r = stio_read(sio, &itm, 0) );
				if (r == STIO_R_CTX_END) break;
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
			ERROR_LOG(-1, "unknown tensor key '%s'", itm.value.p.cp);
		}
	}
	
	can_ignore = true;  //TODO: cfg

	TRY_LOG(e.dtype, "unknown dtype '%s'", itm.value.p.cp);

	if (!(e.size >= e.offset))
		ERROR_LOG(-1, "invalid offsets [%"PRIu64", %"PRIu64"]", e.offset, e.size);
	e.size -= e.offset;

	if (tstore_tensor_size(&e) != e.size)
		ERROR_LOG(-1, "invalid size %"PRIu64" for %s %ux%ux%ux%u",
			e.size,	tstore_dtype_str(e.dtype),
			e.shape[0], e.shape[1], e.shape[2], e.shape[3] );

	// Reverse shape and fill with ones
	for (unsigned i=0; i<e.shape_n/2; ++i)
		SWAPTg(unsigned, e.shape[i], e.shape[e.shape_n-1-i]);
	for (unsigned i=e.shape_n; i<COUNTOF(e.shape); ++i)
		e.shape[i] = 1;

	e.stm = sio->s;
	e.offset += S->os_data;
	MAXSET(S->os_end, e.offset + e.size);

	tstore_tensor_add(S, key, &e);

end:
	if (R<0) {
		if (can_ignore) {
			log_warning("safetensors ignoring invalid tensor '%s'", *key);
			R = 0;
		}
		else
			log_error("safetensors tensor '%s'", *key);
	}
	return R;
}

int safet_load_head(TensorStore* S, Stream* stm, const char* prefix)
{
	int R=1, r;
	StioStream sio={0};
	StioItem itm;
	DynStr key=NULL;
	char buffer[2048];

	if (stream_read_var(stm, S->os_data) < 0)
		ERROR_LOG(-1, "could not read");
	S->os_data += 8;

	if (stream_char_get(stm) != '{')
		ERROR_LOG(-1, "invalid file");
	stream_unget(stm, 1);
	
	if (S->os_data > 1024*1024)
		ERROR_LOG(-1, "header too big: %"PRIu64"", S->os_data);

	TRY( stio_init(&sio, stm, &stio_class_json, 0, sizeof(buffer), buffer) );
	
	TRY( stio_read(&sio, &itm, 0) );
	TRY( stio_item_open_check(&itm, STIO_T_VALUE, ANY_T_MAP) );

	while (1) {
		TRY( r = stio_read(&sio, &itm, 0) );
		if (r == STIO_R_CTX_END) break;
		TRY( stio_item_type_check(&itm, STIO_T_KEY, ANY_T_STRING) );

		if (!strcmp(itm.value.p.cp, "__metadata__"))
			TRY( safet_load_meta(S, &sio) );
		else {
			dstr_copyz(key, prefix);
			dstr_appendz(key, itm.value.p.cp);
			TRY( safet_load_tensor_head(S, &sio, &key) );
		}
	}

	unsigned n = vec_count(S->tensors);
	if (n) log_debug("safetensors tensors loaded: %u", n);
	
	log_debug("safetensors read header size: %"PRIu64, S->os_data);
	log_debug("safetensors read total size: %"PRIu64, S->os_end);

end:
	if (R<0) log_error("safetensors header");
	dstr_free(key);
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

static inline
bool id_prefix_check(int id, const char* prefix)
{
	if (!(prefix && prefix[0])) return true;
	const char *str = id_str(id);
	unsigned i=0;
	while (str[i] == prefix[i] && str[i]) i++;
	return !prefix[i];
}

int safet_save_head(TensorStore* S, Stream* stm, const char* prefix)
{
	int R=0;
	uint64_t offset=0;

	if (stream_write_var(stm, offset) < 0)  //placeholder
		ERROR_LOG(-1, "safetensors could not write");
	
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
			stream_str_put_escape(stm, id_str(S->meta[i].value));
			stream_char_put(stm, '"');
		}
		stream_char_put(stm, '}');
		first=false;
	}

	// Tensors
	vec_forp(TSTensorEntry, S->tensors, e, 0) {
		if (!id_prefix_check(e->key, prefix)) continue;
		e->stm = stm;
		e->offset = offset;
		e->size = tstore_tensor_size(e);
		offset += e->size;

		if (first) first=false; else stream_char_put(stm, ',');
		stream_char_put(stm, '"');
		stream_str_put_escape(stm, id_str(e->key));
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
	S->os_end = offset;
		
	stream_char_put(stm, '}');

	// Header size / data offset
	S->os_data = stream_pos_get(stm);
	TRY( stream_seek(stm, 0, 0) );
	offset = S->os_data - 8;
	TRY( stream_write_var(stm, offset) );
	TRY( stream_seek(stm, S->os_data, 0) );
	
	vec_forp(TSTensorEntry, S->tensors, e, 0) e->offset += S->os_data;
	S->os_end += S->os_data;
	
	log_debug("safetensors write header size: %"PRIu64, S->os_data);
	log_debug("safetensors write total size: %"PRIu64, S->os_end);

end:
	return R;
}
