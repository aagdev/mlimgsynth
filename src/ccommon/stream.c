/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#include "stream.h"
#include "alloc.h"
#include <limits.h>

#ifdef STREAM_LOG_DEBUG
	#define DebugLog(...) (printf("STREAM " __VA_ARGS__), putchar('\n'))
#else
	#define DebugLog(...)
#endif

#ifndef STREAM_BUFFER_SIZE
#define STREAM_BUFFER_SIZE 4096
#endif

#ifndef STREAM_BUFFER_SIZE_MIN
#define STREAM_BUFFER_SIZE_MIN 32
#endif

#ifndef STREAM_ALLOCATOR
#define STREAM_ALLOCATOR  g_allocator
#endif

#define FLAG_SET(V,F,C) \
	((C) ? ((V) |= (F)) : ((V) &= ~(F)))

const StreamClass stream_class_memory = { .name = "memory" };

int stream_open_flags_normalize(int flags)
{
	if (flags & SOF_CREATE) {
		if ((flags & SOF_WRITE_MASK) == SOF_CREATE) flags |= SOF_TRUNCATE;
		flags |= SOF_WRITE;
	}
	else if (flags & SOF_WRITE) {
	}
	else {
		flags |= SOF_READ;
	}
	return flags;
}

static inline
int open_flags_to_flags(int oflags)
{
	int sflags=0;
	if (oflags & SOF_READ ) sflags |= SF_ALLOW_READ;
	if (oflags & SOF_WRITE) sflags |= SF_ALLOW_WRITE;
	if (!sflags) sflags |= SF_ALLOW_READ;
	return sflags;
}

static inline
void stream_clean(Stream* S)
{
	S->cursor = S->cursor_end = S->buffer;
	S->flags &= SF_BUFFER_MANAGE;  //keeps only this flag
	S->pos = 0;
	S->cls = NULL;
	S->internal = NULL;
}

int stream_ibuffer_set(Stream*restrict S, size_t size, void*restrict buffer)
{
	if (buffer)
	{
		if (size < STREAM_BUFFER_SIZE_MIN)
			return STREAM_E_GENERIC;

		if (S->buffer)  //TODO: what should be done with the current data?
			return STREAM_E_GENERIC;

		S->flags &= ~SF_BUFFER_MANAGE;

		S->cursor_end = S->cursor = S->buffer = buffer;
	}
	else
	{
		if (size < STREAM_BUFFER_SIZE_MIN)
			size = STREAM_BUFFER_SIZE_MIN;

		if (S->buffer) {
			// No downsizing
			if (S->buffer + size <= S->buffer_end)
				return 0;

			if (~S->flags & SF_BUFFER_MANAGE)
				return STREAM_E_GENERIC;
		}

		uint8_t * buf = alloc_realloc(STREAM_ALLOCATOR, S->buffer, size);
		size = alloc_size_opt(STREAM_ALLOCATOR, buf, size);
		S->flags |= SF_BUFFER_MANAGE;
		S->cursor = S->cursor - S->buffer + buf;
		S->cursor_end = S->cursor_end - S->buffer + buf;
		S->buffer = buf;
	}

	S->buffer_end = S->buffer + size;
	if (S->flags & SF_MODE_WRITE)
		S->cursor_end = S->buffer_end;

	return 0;
}

int stream_ibuffer_increase(Stream* S, size_t size)
{
	if (!S->buffer && size < STREAM_BUFFER_SIZE)
		size = STREAM_BUFFER_SIZE;  //first buffer allocation
	if (S->buffer + size <= S->buffer_end) return 0;
	return stream_ibuffer_set(S, size, NULL);
}

void stream_open_begin(Stream* S, int flags)
{
	stream_close(S, 0);  //SCF_KEEP_BUFFER);
}

int stream_open_end(Stream* S, int flags)
{
	flags = stream_open_flags_normalize(flags);  //reduntant?
	S->flags |= open_flags_to_flags(flags);

	//TODO SOF_COPY ?

	if ((S->flags & (SF_ALLOW_READ|SF_ALLOW_WRITE)) == SF_ALLOW_WRITE)
	{
		S->flags |= SF_MODE_WRITE;
		S->cursor_end = S->buffer_end;
	}

	return 0;
}

int stream_open_memory(Stream*restrict S, void*restrict buf, size_t size,
	int flags)
{
	stream_open_begin(S, flags);
	flags = stream_open_flags_normalize(flags);

	if (!size)
		return STREAM_E_GENERIC;

	if (!buf) {
		TRYR( stream_ibuffer_set(S, size+1, NULL) );
	}
	else if (flags & SOF_COPY) {
		flags &= ~SOF_COPY;
		TRYR( stream_ibuffer_set(S, size+1, NULL) );
		memcpy(S->buffer, buf, size);
		S->cursor = S->buffer;
		S->cursor_end = S->cursor + size;
	}
	else {
		S->cursor = S->buffer = buf;
		S->cursor_end = S->buffer_end = S->buffer + size;
	}

	S->cls = &stream_class_memory;
	return stream_open_end(S, flags);
}

void stream_buffer_free(Stream* S)
{
	if (S->flags & SF_BUFFER_MANAGE) {
		alloc_free(STREAM_ALLOCATOR, S->buffer);
		S->flags &= ~SF_BUFFER_MANAGE;
	}
	S->buffer = S->buffer_end = NULL;
}

int stream_close(Stream* S, int flags)
{
	int r=0;
	if (!S) return 0;

	if (S->cls) {
		if (S->flags & SF_MODE_WRITE && !(flags & SCF_NO_FLUSH)) {
			int rr = stream_flush(S);
			if (rr < 0) r = rr;
		}

		if (S->cls->close) {
			DebugLog("close");
			int rr = S->cls->close(S->internal);
			if (!r && rr < 0) r = rr;
		}
	}

	if (S->buffer && !(flags & SCF_KEEP_BUFFER))
		stream_buffer_free(S);

	stream_clean(S);

	return r;
}

void stream_read_buffer_reposition(Stream* S)
{
	assert( !(S->flags & SF_MODE_WRITE) );
	assert( S->cls->read );
	if (S->cursor > S->buffer) {
		size_t data_size = S->cursor_end - S->cursor;
		if (data_size > 0)
			memmove(S->buffer, S->cursor, data_size);
		S->pos += S->cursor - S->buffer;
		S->cursor = S->buffer;
		S->cursor_end = S->cursor + data_size;
	}
}

int stream_read_buffer_fill(Stream* S)
{
	assert( !(S->flags & SF_MODE_WRITE) );
	size_t n_free = S->buffer_end - S->cursor_end;
	if (n_free > 0) {
		DebugLog("read %zu", n_free);
		if (!S->cls->read) return STREAM_E_READ;
		long n = S->cls->read(S->internal, S->cursor_end, n_free);
		if (n < 0) return n;
		FLAG_SET(S->flags, SF_END, n<n_free);
		S->cursor_end += n;
	}
	return 0;
}

int stream_read_buffer_discard(Stream* S)
{
	assert( !(S->flags & SF_MODE_WRITE) );
	intptr_t n_read = S->cursor_end - S->cursor;
	if (n_read > 0) {
		DebugLog("seek %+zd", -n_read);
		if (!S->cls->seek) return STREAM_E_SEEK;
		int64_t r = S->cls->seek(S->internal, -n_read, SEEK_CUR);
		if (r < 0 && r != STREAM_E_TELL) return r;
		assert( r == STREAM_E_TELL || r == S->pos );
	}
	S->pos += S->cursor - S->buffer;
	S->cursor = S->cursor_end = S->buffer;
	return 0;
}

int stream_write_buffer_flush(Stream* S)
{
	assert( S->flags & SF_MODE_WRITE );
	size_t n_pending = S->cursor - S->buffer;
	if (n_pending > 0) {
		DebugLog("write %zu", n_pending);
		if (!S->cls->write) return STREAM_E_WRITE;
		long n = S->cls->write(S->internal, S->buffer, n_pending);
		if (n < 0) return n;
		if (n != n_pending) return STREAM_E_WRITE;  //TODO: loop?

		S->pos += S->cursor - S->buffer;
		S->cursor = S->buffer;
		S->cursor_end = S->buffer_end;
	}
	return 0;
}

int stream_read_prep(Stream* S, size_t nbytes)
{
	if (!(S->flags & SF_ALLOW_READ))
		return STREAM_E_NOT_ALLOW;

	// Switch mode
	if (S->flags & SF_MODE_WRITE) {
		if (S->cls->write)
			TRYR( stream_write_buffer_flush(S) );
		
		S->flags &= ~SF_MODE_WRITE;
		if (S->cls->read)
			S->cursor_end = S->cursor = S->buffer;

		if (S->cls->seek)
			S->cls->seek(S->internal, 0, SEEK_CUR);
	}

	size_t ncheck = nbytes;
	IFFALSESET(ncheck, (S->buffer_end - S->buffer)/2 );
	IFFALSESET(ncheck, STREAM_BUFFER_SIZE/2);

	// Read more data
	if (!(S->cursor+ncheck <= S->cursor_end) && S->cls->read)
	{
		TRYR( stream_ibuffer_increase(S, nbytes) );
		stream_read_buffer_reposition(S);
		TRYR( stream_read_buffer_fill(S) );
	}

	if (nbytes && !(S->cursor+nbytes <= S->cursor_end))
		return STREAM_E_READ;

	assert(S->cursor_end >= S->cursor);
	size_t sz = S->cursor_end - S->cursor;
	MINSET(sz, INT_MAX);
	return sz;
}

int stream_write_prep(Stream* S, size_t nbytes)
{
	if (!(S->flags & SF_ALLOW_WRITE))
		return STREAM_E_NOT_ALLOW;

	// Switch mode
	if (!(S->flags & SF_MODE_WRITE)) {
		if (S->cls->write)
			TRYR( stream_read_buffer_discard(S) );
		
		S->flags |= SF_MODE_WRITE;
		S->cursor_end = S->buffer_end;
		
		if (S->cls->seek)
			S->cls->seek(S->internal, 0, SEEK_CUR);
	}

	size_t ncheck = nbytes;
	IFFALSESET(ncheck, (S->buffer_end - S->buffer)/2 );
	IFFALSESET(ncheck, STREAM_BUFFER_SIZE/2);

	// Liberate space
	if (!(S->cursor+ncheck <= S->cursor_end))
	{
		if (S->cls->write)
			TRYR( stream_write_buffer_flush(S) );

		TRYR( stream_ibuffer_increase(S, nbytes) );
	}

	if (nbytes && !(S->cursor+nbytes <= S->cursor_end))
		return STREAM_E_GENERIC;  //should not happen

	assert(S->cursor_end >= S->cursor);
	size_t sz = S->cursor_end - S->cursor;
	MINSET(sz, INT_MAX);
	return sz;
}

int stream_flush(Stream* S)
{
	if (S->cls->write)
		TRYR( stream_write_buffer_flush(S) );
	if (S->cls->flush) {
		DebugLog("flush");
		TRYR( S->cls->flush(S->internal) );
	}
	return 0;
}

int stream_seek_i(Stream* S, int64_t offset, int origin)
{
	if (S->cls->seek)
	{
		if (S->flags & SF_MODE_WRITE && S->cls->write)
			TRYR( stream_write_buffer_flush(S) );

		int64_t r;
		switch (origin) {
		case SEEK_SET:
			DebugLog("seek %ld", (long)offset);
			r = S->cls->seek(S->internal, offset, SEEK_SET);
			if (r < 0 && r != STREAM_E_TELL) return r;
			S->pos = offset;
			S->cursor = S->cursor_end = S->buffer;
			break;

		case SEEK_CUR:
			offset -= S->cursor - S->buffer;
			DebugLog("seek %+ld", (long)offset);
			r = S->cls->seek(S->internal, offset, SEEK_CUR);
			if (r < 0 && r != STREAM_E_TELL) return r;
			S->pos += offset;
			S->cursor = S->cursor_end = S->buffer;
			break;

		case SEEK_END:
			DebugLog("seek end%+ld", (long)offset);
			r = S->cls->seek(S->internal, offset, SEEK_END);
			if (r < 0) return r;
			S->pos = r;
			S->cursor = S->cursor_end = S->buffer;
			break;

		default:
			assert(0);
		}

		assert(r == STREAM_E_TELL || r == S->pos);
		
		if (S->flags & SF_MODE_WRITE)
			S->cursor_end = S->buffer_end;
	}
	else
	{
		uint8_t* cur = S->cursor;
		switch (origin) {
		case SEEK_SET:
			cur = S->buffer + offset;
			break;
		case SEEK_CUR:
			cur = S->cursor + offset;
			break;
		case SEEK_END:
			cur = S->cursor_end + offset;
			break;
		}

		if (cur < S->buffer) return STREAM_E_SEEK;
		if (cur > S->cursor_end) return STREAM_E_SEEK;
		S->cursor = cur;
	}

	return 0;
}

long stream_read_i(Stream*restrict S, size_t nbyte, void*restrict buf)
{
	assert((long)nbyte >= 0);
	if (S->flags & SF_MODE_WRITE || nbyte < STREAM_BUFFER_SIZE/2)
		TRYR( stream_read_prep(S, 0) );

	uint8_t *bcur = buf;

	if (S->cursor < S->cursor_end) {
		size_t n = MINg(nbyte, S->cursor_end - S->cursor);
		memcpy(bcur, S->cursor, n);
		S->cursor += n;
		bcur += n;
		nbyte -= n;
	}

	if (nbyte && S->cls->read)
	{	//pass through
		S->pos += S->cursor - S->buffer;
		S->cursor = S->cursor_end = S->buffer;

		DebugLog("read through %zu", nbyte);	
		long n = S->cls->read(S->internal, bcur, nbyte);
		if (n < 0) return n;
		FLAG_SET(S->flags, SF_END, n<nbyte);
		S->pos += n;
		bcur += n;
		//nbyte -= n;
	}

	return bcur - (uint8_t*)buf;
}

long stream_write_i(Stream*restrict S, size_t nbyte, const void*restrict buf)
{
	assert((long)nbyte >= 0);
	if (!(S->flags & SF_MODE_WRITE) || nbyte < STREAM_BUFFER_SIZE/2)
		TRYR( stream_write_prep(S, 0) );
	
	if (S->cursor+nbyte <= S->cursor_end) {
		memcpy(S->cursor, buf, nbyte);
		S->cursor += nbyte;
		return nbyte;
	}
	else if (S->cls->write) {
		TRYR( stream_write_buffer_flush(S) );

		DebugLog("write through %zu", nbyte);
		long n = S->cls->write(S->internal, buf, nbyte);
		if (n < 0) return n;
		S->pos += n;
		return n;
	}
	else
		return 0;  //should it be an error?
}

#if __STDC_HOSTED__
int stream_vprintf(Stream*restrict S, const char*restrict format, va_list ap)
{
	char *end, *cur = stream_write_buffer(S, &end);
	assert(cur < end);
	int nbyte = vsnprintf(cur, end-cur, format, ap);
	if (nbyte > 0) stream_commit(S, cur+nbyte);
	return nbyte;
}

int stream_printf(Stream*restrict S, const char*restrict format, ...)
{
	va_list ap;
	va_start(ap, format);
	int nbytes = stream_vprintf(S, format, ap);
	va_end(ap);
	return nbytes;
}
#endif  //__STDC_HOSTED__

int stream_open_argv(Stream* S, int argc, char* argv[], int sep)
{
	int R=0;

	// Calculates the total size
	unsigned size = 1 + (sep>=0 ? argc : 0); 
	for (int i=1; i<argc; ++i) size += strlen(argv[i]);
	assert(size < 0xFFFFFF);

	// Prepare a new memory stream
	TRY( stream_open_memory(S, NULL, size, SOF_READ|SOF_WRITE) );

	// Copy the arguments
	char *bcur = stream_write_buffer(S, NULL);
	for (int i=1; i<argc; ++i) {
		if (i>1 && sep >= 0) *bcur++ = sep;
		for (char *acur=argv[i]; *acur; ++acur) 
			*bcur++ = *acur;
	}
	stream_commit(S, bcur);

	// Ready to read
	TRY( stream_seek(S, 0, 0) );
	TRY( stream_read_prep(S, 0) );

end:
	if (R<0) stream_close(S, 0);
	return size;
}

int stream_full_file_load(Stream*restrict S, const char*restrict path)
{
	int R=0;
	
	TRY( stream_open_file(S, path, SOF_READ) );
	TRY( stream_seek(S, 0, SEEK_END) );
	uint64_t sz = stream_pos_get(S);
	if (sz > (1u<<30)) RETURN(-1);  //sanity check
	TRY( stream_ibuffer_set(S, sz, NULL) );
	TRY( stream_seek(S, 0, SEEK_SET) );
	TRY( stream_read_prep(S, sz) );
	assert( S->cursor_end - S->cursor == sz );
	stream_close(S, SCF_KEEP_BUFFER);
	assert( S->buffer_end - S->buffer >= sz );

	//TODO: operation: convert any stream to a memory stream
	S->cls = &stream_class_memory;
	S->flags |= SF_ALLOW_READ | SF_ALLOW_WRITE;
	S->cursor = S->buffer;
	S->cursor_end = S->buffer + sz;
	
end:
	if (R < 0) stream_close(S, 0);
	return R;
}

const char* stream_error_desc_get(int error)
{
	switch ((StreamError)error) {
	case STREAM_E_GENERIC:			return "unknown";
	case STREAM_E_EOF:				return "end of file";
	case STREAM_E_MEM_ALLOC:		return "out of memory";
	case STREAM_E_NOT_ALLOW:		return "op. not allowed";
	case STREAM_E_NOT_SUPPORTED:	return "op. not supported";
	case STREAM_E_OPEN:				return "open";
	case STREAM_E_READ:				return "read";
	case STREAM_E_WRITE:			return "write";
	case STREAM_E_CLOSE:			return "close";
	case STREAM_E_SEEK:				return "seek";
	case STREAM_E_TELL:				return "tell";
	case STREAM_E_FLUSH:			return "flush";
	case STREAM_E_DATA:				return "data";
	}
	return "???";
}

/*
	C standard library I/O interface
*/
#if __STDC_HOSTED__
const char* stream_open_flags_to_stdio_mode(int flags)
{
	if (flags & SOF_CREATE) {
		if (flags & SOF_READ) {
			if (flags & SOF_APPEND) return "a+b";
			else return "w+b";
		} else {
			if (flags & SOF_APPEND) return "ab";
			else return "wb";
		}
	}
	else if (flags & SOF_WRITE) {
		if (flags & SOF_READ) return "r+b";
		else return "wb";
	}
	else if (flags & SOF_READ) return "rb";
	else return "rb";
}

int stream_stdio_open_handle(Stream*restrict s, FILE*restrict f, int flags)
{
	// In case we are switching from reading to writing or vise versa
	fseek(f, 0, SEEK_CUR);

	stream_open_begin(s, flags);
	s->internal = (void*)f;
	s->cls = &stream_class_stdio;
	return stream_open_end(s, flags);
}

int stream_stdio_open_file(Stream*restrict s, const char*restrict path,
	int flags)
{
	flags = stream_open_flags_normalize(flags);
	const char* mode = stream_open_flags_to_stdio_mode(flags);
	FILE* f = fopen(path, mode);
	if (!f) return STREAM_E_OPEN;
	//setbuf(f, 0);

	return stream_stdio_open_handle(s, f, flags);
}

int stream_stdio_close(void* p)
{
	FILE* f = p;
	int r = 0;
	if (f) {
		r = fclose(f);
		if (r == EOF)
			r = STREAM_E_CLOSE;
	}
	return r;
}

long stream_stdio_read(void*restrict p, void*restrict buffer, size_t size)
{
	FILE* f = p;
	size_t n_done = fread(buffer, 1, size, f);
	if (!n_done) {
		if (ferror(f)) return STREAM_E_READ;
	}
	//if (feof(f)) s->flags |= SF_EOF;
	return n_done;
}

long stream_stdio_write(void*restrict p, const void*restrict buffer,
	size_t size)
{
	FILE* f = p;
	size_t n_done = fwrite(buffer, 1, size, f);
	if (n_done != size) {
		if (ferror(f)) return STREAM_E_WRITE;
	}
	return n_done;
}

int64_t stream_stdio_seek(void* p, int64_t offset, int origin)
{
	FILE* f = p;

	long os = offset;
	if (os != offset) return STREAM_E_NOT_SUPPORTED;

	if (fseek(f, os, origin))
		return STREAM_E_SEEK;

	os = ftell(f);
	if (os == EOF) return STREAM_E_TELL;

	return os;
}

int stream_stdio_flush(void* p)
{
	return fflush((FILE*)p) == 0 ? 0 : STREAM_E_FLUSH;
}

const StreamClass stream_class_stdio = {
	stream_stdio_read,
	stream_stdio_write,
	stream_stdio_close,
	stream_stdio_seek,
	stream_stdio_flush,
	"stdio"
};
#endif  //__STDC_HOSTED__

/*
	Windows interface
*/
#ifdef __WIN32__
#define WIN32_LEAN_AND_MEAN
#define WIN32_EXTRA_LEAN
#include <windows.h>

struct StreamWinapiMmap {
	HANDLE hf, hm;
	void *addr;
};

int stream_winapi_mmap_open_handle(Stream*restrict S, HANDLE hf, int oflags)
{
	if (oflags & SOF_WRITE) return -1;

	size_t size;
	DWORD sh, sl = GetFileSize(hf, &sh);
	size = (size_t)sl + (((size_t)sh) << 32);

	HANDLE hm = CreateFileMappingA(hf, NULL, PAGE_READONLY, 0, 0, NULL);
	if (!hm) return -1;

	uint8_t *p = MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
	if (!p) {
		CloseHandle(hm);
		return -1;
	}
	
	if (S->buffer) stream_buffer_free(S);
	S->buffer = p;
	S->buffer_end = p + size;
	S->cursor = S->buffer;
	S->cursor_end = S->buffer_end;
	
	struct StreamWinapiMmap *state;
	state = alloc_alloc(STREAM_ALLOCATOR, sizeof(*state));
	state->hf = hf;
	state->hm = hm;
	state->addr = p;
	
	S->internal = (void*)state;
	S->cls = &stream_class_winapi_mmap;
	return stream_open_end(S, oflags);
}

int stream_winapi_mmap_close(void* p)
{
	bool ok=true;
	struct StreamWinapiMmap *s = p;
	if (!UnmapViewOfFile(s->addr)) ok = false;
	if (!CloseHandle(s->hm)) ok = false;
	if (!CloseHandle(s->hf)) ok = false;
	*s = (struct StreamWinapiMmap){0};
	alloc_free(STREAM_ALLOCATOR, s);
	return ok ? 0 : STREAM_E_CLOSE;
}

const StreamClass stream_class_winapi_mmap = {
	NULL,
	NULL,
	stream_winapi_mmap_close,
	NULL,
	NULL,
	"winapi-mmap"
};

int stream_winapi_open_handle(Stream*restrict S, HANDLE h, int oflags)
{
	stream_open_begin(S, oflags);
	if (oflags & SOF_MMAP && !(oflags & SOF_WRITE)) {  //default?
		int r = stream_winapi_mmap_open_handle(S, h, oflags);
		if (r >= 0) return 1;
		// If it fails, fall back to a normal stream
	}
	S->internal = (void*)h;
	S->cls = &stream_class_winapi;
	return stream_open_end(S, oflags);
}

int stream_winapi_open_file(Stream*restrict s, const char*restrict path,
	int flags)
{
	flags = stream_open_flags_normalize(flags);

	DWORD dwDesiredAccess=0;
	if (flags & SOF_READ)  dwDesiredAccess |= GENERIC_READ;
	if (flags & SOF_WRITE) dwDesiredAccess |= GENERIC_WRITE;

	DWORD dwCreationDisposition=0;
	if (flags & SOF_WRITE) {
		if (flags & SOF_CREATE) {
			if (flags & SOF_TRUNCATE)
				dwCreationDisposition = CREATE_ALWAYS;
			else
				dwCreationDisposition = OPEN_ALWAYS;
		}
		else if (flags & SOF_TRUNCATE)
			dwCreationDisposition = TRUNCATE_EXISTING;
		else
			dwCreationDisposition = OPEN_EXISTING;
	} else
		dwCreationDisposition = OPEN_EXISTING;

	//TODO: CreateFileW
	HANDLE h = CreateFileA(path, dwDesiredAccess,
		FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
		NULL, dwCreationDisposition, FILE_ATTRIBUTE_NORMAL, NULL);
	if (h == INVALID_HANDLE_VALUE) return STREAM_E_OPEN;

	if (flags & SOF_APPEND) {
		DWORD r = SetFilePointer(h, 0, NULL, FILE_END);
		if (r == 0xFFFFFFFF) {
			CloseHandle(h);
			return STREAM_E_OPEN;
		}
	}

	return stream_winapi_open_handle(s, h, flags);
}

int stream_winapi_close(void* p)
{
	HANDLE h = (HANDLE)p;
	int r = 0;
	if (h) {
		if (!CloseHandle(h))
			r = STREAM_E_CLOSE;
	}
	return r;
}

long stream_winapi_read(void*restrict p, void*restrict buffer, size_t size)
{
	HANDLE h = (HANDLE)p;
	DWORD n=0;
	if (!ReadFile(h, buffer, size, &n, NULL))
		return STREAM_E_READ;
	//if (GetLastError() == ERROR_HANDLE_EOF)
	return n;
}

long stream_winapi_write(void*restrict p, const void*restrict buffer,
	size_t size)
{
	HANDLE h = (HANDLE)p;
	DWORD n=0;
	if (!WriteFile(h, buffer, size, &n, NULL))
		return STREAM_E_WRITE;
	return n;
}

int64_t stream_winapi_seek(void* p, int64_t offset, int origin)
{
	HANDLE hfile = (HANDLE)p;

	DWORD dwMoveMethod=0;
	switch (origin) {
	case SEEK_SET:  dwMoveMethod = FILE_BEGIN;  break;
	case SEEK_CUR:  dwMoveMethod = FILE_CURRENT;  break;
	case SEEK_END:  dwMoveMethod = FILE_END;  break;
	}

	LONG l=offset&0xFFFFFFFF, h=(offset>>32)&0xFFFFFFFF;
	l = SetFilePointer(hfile, l, &h, dwMoveMethod);
	if (l == 0xFFFFFFFF) {
		DWORD r = GetLastError();
		if (r != NO_ERROR) return STREAM_E_SEEK;
	}

	return l | ((int64_t)h<<32);
}

int stream_winapi_flush(void* p)
{
	return FlushFileBuffers((HANDLE)p) ? 0 : STREAM_E_FLUSH;
}

const StreamClass stream_class_winapi = {
	stream_winapi_read,
	stream_winapi_write,
	stream_winapi_close,
	stream_winapi_seek,
	stream_winapi_flush,
	"winapi"
};
#endif //__WIN32__

/*
	POSIX interface
*/
#ifdef __unix__
#define _FILE_OFFSET_BITS 64
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>  //mmap
#include <fcntl.h>
#include <unistd.h>

struct StreamPosixMmap {
	void *addr;
	size_t size;
};

int stream_posix_mmap_open_handle(Stream* S, int fd, int oflags)
{
	off_t size = lseek(fd, 0, SEEK_END);
	if (size < 0) return -1;
	if (lseek(fd, 0, SEEK_SET) != 0) return -1;

	int prot = 0;
	if (oflags & SOF_READ)  prot |= PROT_READ;
	if (oflags & SOF_WRITE) prot |= PROT_WRITE;  //TODO: testing pending
	uint8_t *p = mmap(NULL, size, prot, MAP_PRIVATE, fd, 0);
	if (p == MAP_FAILED) return -1;

	if (S->buffer) stream_buffer_free(S);
	S->buffer = p;
	S->buffer_end = p + size;
	S->cursor = S->buffer;
	S->cursor_end = S->buffer_end;

	struct StreamPosixMmap *state;
	state = alloc_new(STREAM_ALLOCATOR, struct StreamPosixMmap, 1);
	state->addr = p;
	state->size = size;
	S->internal = state;
	S->cls = &stream_class_posix_mmap;

	close(fd);  //TODO: optional (eg std*)

	return stream_open_end(S, oflags);
}

int stream_posix_mmap_close(void* p)
{
	struct StreamPosixMmap *s = p;
	int r = munmap(s->addr, s->size);
	s->addr = NULL;
	alloc_free(STREAM_ALLOCATOR, s);
	return r<0 ? STREAM_E_CLOSE : 0;
}

int stream_posix_mmap_flush(void* p)
{
	struct StreamPosixMmap *s = p;
	int r = msync(s->addr, s->size, MS_SYNC);
	return r<0 ? STREAM_E_FLUSH : 0;
}

const StreamClass stream_class_posix_mmap = {
	NULL,
	NULL,
	stream_posix_mmap_close,
	NULL,
	stream_posix_mmap_flush,
	"posix-mmap"
};

int stream_posix_open_handle(Stream* S, int fd, int flags)
{
	stream_open_begin(S, flags);
	if (flags & SOF_MMAP && !(flags & SOF_WRITE)) {  //default?
		int r = stream_posix_mmap_open_handle(S, fd, flags);
		if (r >= 0) return 1;
		// If it fails, fall back to a normal stream
	}
	S->internal = (void*)(uintptr_t)fd;
	S->cls = &stream_class_posix;
	return stream_open_end(S, flags);
}

int stream_posix_open_file(Stream*restrict s, const char*restrict path,
	int flags)
{
	int f=0;
	flags = stream_open_flags_normalize(flags);
	if (flags & SOF_WRITE) {
		if (flags & SOF_READ) f |= O_RDWR;
		else f |= O_WRONLY;
		if (flags & SOF_CREATE) f |= O_CREAT;
		if (flags & SOF_APPEND) f |= O_APPEND;
		if (flags & SOF_TRUNCATE) f |= O_TRUNC;
	} else
		f |= O_RDONLY;

	int fd = open(path, f, S_IRUSR|S_IWUSR);
	if (fd < 0) return STREAM_E_OPEN;

	return stream_posix_open_handle(s, fd, flags);
}

int stream_posix_close(void* p)
{
	int fd = (int)(uintptr_t)p;
	int r = 0;
	if (fd) {
		r = close(fd);
		if (r < 0)
			r = STREAM_E_CLOSE;
	}
	return r;
}

long stream_posix_read(void*restrict p, void*restrict buffer,
	size_t size)
{
	int fd = (int)(uintptr_t)p;
	ssize_t n_done = read(fd, buffer, size);
	if (n_done < 0) return STREAM_E_READ;
	//if (n_done == 0) s->flags |= SF_EOF;
	return n_done;
}

long stream_posix_write(void*restrict p, const void*restrict buffer,
	size_t size)
{
	int fd = (int)(uintptr_t)p;
	ssize_t n_done = write(fd, buffer, size);
	if (n_done < 0) return STREAM_E_WRITE;
	return n_done;
}

int64_t stream_posix_seek(void* p, int64_t offset, int origin)
{
	int fd = (int)(uintptr_t)p;
	off_t os = lseek(fd, offset, origin);
	if (os < 0) return STREAM_E_SEEK;
	return os;
}

int stream_posix_flush(void* p)
{
	//fdatasync?
	int r = fsync((int)(uintptr_t)p);
	if (r == -22) r = 0;  //EINVAL: ok for pipes
	return (r < 0) ? STREAM_E_FLUSH : 0;
}

const StreamClass stream_class_posix = {
	stream_posix_read,
	stream_posix_write,
	stream_posix_close,
	stream_posix_seek,
	stream_posix_flush,
	"posix"
};
#endif  //__unix__

/* Cross-platform standard stream access
 */
int stream_open_std(Stream* s, int fd, int flags)
{
	if (!flags) {
		switch (fd) {
		case STREAM_STD_IN:
			flags = SOF_READ;
			break;
		case STREAM_STD_OUT:
		case STREAM_STD_ERR:
			flags = SOF_WRITE;
			break;
		}
	}

#if defined(__unix__)
	return stream_posix_open_handle(s, fd, flags);	

#elif defined(__WIN32__)
	int n=0;
	switch (fd) {
	case STREAM_STD_IN:
		n = STD_INPUT_HANDLE;
		break;
	case STREAM_STD_OUT:
		n = STD_OUTPUT_HANDLE;
		break;
	case STREAM_STD_ERR:
		n = STD_ERROR_HANDLE;
		break;
	}
	HANDLE h = GetStdHandle(n);
	return stream_winapi_open_handle(s, h, flags);

#else
	FILE * h=NULL;
	switch (fd) {
	case STREAM_STD_IN:
		h = stdin;
		break;
	case STREAM_STD_OUT:
		h = stdout;
		break;
	case STREAM_STD_ERR:
		h = stderr;
		break;
	}
	return stream_stdio_open_handle(s, h, flags);
#endif
}
