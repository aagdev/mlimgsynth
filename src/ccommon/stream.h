/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 *
 * Streams
 *
 * Example:
 *
Stream stm={0};  //initialize to zero
TRY( stream_open_file(&stm, "test.txt", SOF_CREATE) );
TRY( stream_str_put("Hello world!\n") );

TRY( stream_seek(&stm, 0, 0) );

char *end, *cur = stream_read_buffer(&stm, &end);
assert( memcmp(cur, "Hello world!\n", end-cur) == 0 );
stream_commit(&stm, end);

stream_close(&stm, 0);
 */
#pragma once
#include "ccommon.h"
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdbool.h>

//TODO: test mode switch with all functions !
//TODO: memory stream like a file: with EOF position and extensible
//TODO: configurable allocator?

typedef enum StreamError {
	STREAM_E_GENERIC		= -0x101,
	STREAM_E_EOF			= -0x102,
	STREAM_E_MEM_ALLOC		= -0x103,
	STREAM_E_NOT_ALLOW		= -0x104,
	STREAM_E_NOT_SUPPORTED	= -0x105,
	STREAM_E_OPEN			= -0x106,
	STREAM_E_READ			= -0x107,
	STREAM_E_WRITE			= -0x108,
	STREAM_E_CLOSE			= -0x109,
	STREAM_E_SEEK			= -0x10a,
	STREAM_E_TELL			= -0x10b,
	STREAM_E_FLUSH			= -0x10c,
	STREAM_E_DATA			= -0x10d,
} StreamError;

enum StreamFlag {
	SF_MODE_WRITE		=  1,
	SF_ALLOW_READ		=  2,
	SF_ALLOW_WRITE		=  4,
	SF_BUFFER_MANAGE	=  8,
	SF_END				= 16,  //end of stream reached
	//SF_MMAP				= 32,  //it's memory mapped
	//non-blocking
	//sync
	//eof? error?
	SF_CUSTOM_MASK		= 0x7FFF0000,
};

enum StreamOpenFlag {
	SOF_READ			= 1,
	SOF_WRITE			= 2,
	SOF_CREATE			= 4,
	SOF_TRUNCATE		= 8,
	SOF_APPEND			= 16,
	/// Copy all the data to memory before proceding
	SOF_COPY			= 32,
	/// Attempts to memory map the source if supported
	SOF_MMAP			= 64,
	
	SOF_WRITE_MASK		= SOF_WRITE|SOF_CREATE|SOF_TRUNCATE|SOF_APPEND,
	SOF_CUSTOM_MASK		= 0x7FFF0000,
};

enum StreamCloseFlag {
	SCF_NO_FLUSH		= 1,
	SCF_KEEP_BUFFER		= 2,
};

//enum StreamSeekOrigin {  //collision with windows.h
//	STREAM_SEEK_SET = 0,
//	STREAM_SEEK_CUR = 1,
//	STREAM_SEEK_END = 2,
//};

typedef struct Stream Stream;

typedef struct StreamClass {
	long (*read)(void*restrict, void*restrict, size_t);
	long (*write)(void*restrict, const void*restrict, size_t);
	int (*close)(void*);
	int64_t (*seek)(void*, int64_t, int);
	int (*flush)(void*);  //fsync
	const char * name;
} StreamClass;

struct Stream {
	uint8_t *cursor, *cursor_end;  // Data range
	uint8_t *buffer, *buffer_end;  // Internal buffer
	uint64_t			pos;
	int32_t				flags;
	const StreamClass *	cls;
	void *				internal;
};

/* Check if the stream is ready to be used.
 */
static inline
bool stream_good(const Stream* S) { return S && S->cls; }

/* Opens an stream to memory location with a fixed size.
 * <buffer> may be NULL, then an internal buffer is allocated.
 * Returns 0 on success, a negative error code on failure.
 */
int stream_open_memory(Stream*restrict S, void*restrict buffer, size_t size,
	int flags);

/* Opens an stream to file.
 * Returns 0 on success, a negative error code on failure.
 */
//int stream_open_file(Stream* S, const char* path, int flags);

/* Closes the stream and free all related resouces.
 * After this it can be initialized again.
 * Returns 0 on success, a negative error code on failure.
 */
int stream_close(Stream* S, int flags);

/* Returns a buffer from where data can be read.
 * Returns NULL if no data can be read.
 * Use stream_commit to advance the stream cursor.
 */
static inline
void* stream_read_buffer_(Stream* S, void** end);
#define stream_read_buffer(S,E)  stream_read_buffer_((S), (void**)(E))

/* Returns a buffer to where data can be written.
 * Returns NULL if no data can be written.
 * Use stream_commit to advance the stream cursor.
 */
static inline
void* stream_write_buffer_(Stream* S, void** end);
#define stream_write_buffer(S,E)  stream_write_buffer_((S), (void**)(E))

/* Prepares to read at least nbytes.
 * If the amount of bytes available is less, returns an error.
 * Returns the read buffer size on succeed, a negative error code on failure.
 */
int stream_read_prep(Stream* S, size_t nbytes);

/* Prepares to write at least nbytes.
 * nbytes may be zero, then it will attempt to have half of the internal buffer
 * ready, but will not return an error for a smaller amount.
 * Returns the write buffer size on succeed, a negative error code on failure.
 */
int stream_write_prep(Stream* S, size_t nbytes);

/* Returns whatever the stream cursor is at the end.
 * In read mode, this means that the last read reached the end of the stream.
 * In write mode, writing more will increase the stream length.
 */
static inline
bool stream_end_is(Stream* S) { return S->flags & SF_END; }

/* Returns the current buffer available size.
 * Valid both for reading and writing.
 */
static inline
size_t stream_buffer_size(Stream* S) { return S->cursor_end - S->cursor; }

/* Returns the current buffer start and end.
 */
static inline
void* stream_buffer_get_(Stream* S, void** end);
#define stream_buffer_get(S,E)  stream_buffer_get_((S), (void**)(E))

/* Advances the stream cursor.
 * Confirms the read or written bytes in the buffer.
 */
static inline
void stream_commit(Stream*restrict S, const void*restrict cursor);

/* Flushes the internal buffers.
 * In write mode, Sends any pending writes to the OS.
 * In read mode, discards any data cached.
 */
int stream_flush(Stream* S);

/* Forces writting the buffer to the underling support (i.e. disk).
 * Implies a flush.
 */
int stream_sync(Stream* S);

/* Reads and get a buffer with nbytes.
 * Advances the cursor.
 * Returns NULL on error.
 */
static inline
void* stream_get(Stream* S, size_t nbytes);

/* Peeks nbytes into the stream.
 * Synonimous of stream_read_buffer.
 */
static inline
void* stream_peek(Stream* S, size_t nbytes);

/* Moves the cursor nbytes back.
 * Return the number of bytes unget'ed.
 * Equivalent to stream_seek(S, -nbytes, SEEK_CUR).
 */
static inline
int stream_unget(Stream* S, size_t nbytes);

/* Reads nbytes to buffer.
 * May skips the internal buffer for large sizes.
 * Returns 0 on success, negative error code on failure.
 */
static inline
long stream_read(Stream*restrict S, size_t nbytes, void*restrict buffer);

/* Like stream_read, but returns an error if less than nbytes were read.
 */
static inline
int stream_read_chk(Stream*restrict S, size_t nbytes, void*restrict buffer);

/* Writes nbytes from buffer.
 * May skips the internal buffer for large sizes.
 * Returns the number of bytes written on success, negative error code on failure.
 */
static inline
long stream_write(Stream*restrict S, size_t nbytes, const void*restrict buffer);

/* Like stream_write, but returns an error if less than nbytes were written.
 * Returns 0 on success.
 */
static inline
int stream_write_chk(Stream*restrict S, size_t nbytes, const void*restrict buffer);

/* Changes the position of the stream cursor if posible.
 *
 * Origin: SEEK_SET (0), SEEK_CUR (1) or SEEK_END (2).
 * Offset: number of byte to move from the origin.
 * Returns 0 on success, negative error code on failure.
 */
static inline
int stream_seek(Stream* S, int64_t offset, int origin);

/* Get the current stream position.
 */
static inline
uint64_t stream_pos_get(Stream* S) { return S->pos + (S->cursor - S->buffer); }

/* Read one byte from the stream.
 * Returns the byte read on success, negative error code on failure.
 */
static inline
int stream_char_get(Stream* S);

/* Reads into the variable V.
 * Returns 0 on success, negative error code on failure.
 */
#define stream_read_var(S, V) \
	stream_read_chk((S), sizeof(V), &(V))

/* Writes the variable V.
 * Returns 0 on success, negative error code on failure.
 */
#define stream_write_var(S, V) \
	stream_write_chk((S), sizeof(V), &(V))

/* Writes one byte..
 * Returns 0 on success, negative error code on failure.
 */
static inline
int stream_char_put(Stream* S, int c);

/* Writes a zero-terminated string.
 * Returns the number of bytes written, negative error code on failure.
 */
static inline
int stream_str_put(Stream*restrict S, const char*restrict str);

/* Writes formatted text.
 * Returns the number of bytes written, negative error code on failure.
 */
#ifdef __GNUC__
__attribute__((format(printf , 2, 3)))
#endif
int stream_printf(Stream*restrict S, const char*restrict format, ...);

int stream_vprintf(Stream*restrict S, const char*restrict format, va_list ap);

/* Changes the internal buffer.
 * If buffer is NULL, then memory is allocated.
 */
int stream_ibuffer_set(Stream*restrict S, size_t size, void*restrict buffer);

/* Increase to size of the internal buffer to at least nbytes.
 * If the buffer has already enough space, no change is done (returns 0).
 * If the buffer is no allocated, an error is returned (< 0).
 */
int stream_ibuffer_increase(Stream* S, size_t nbytes);

/* Returns the usable space of the internal buffer.
 */
static inline
size_t stream_ibuffer_size(Stream* S) { return S->buffer_end - S->buffer; }

/* Returns a static error description in english.
 */
const char* stream_error_desc_get(int error);

/* Convert a program's C arguments to a memory stream.
 *
 * <sep> is a character to insert between arguments (but not at the end),
 * use negative for none. Usually ' ', '\t', '\n' or 0.
 * Returns the number of bytes in the stream on success,
 * a negative error code on failure.
 */
int stream_open_argv(Stream* s, int argc, char* argv[], int sep);

/* Initialize an stream to access a standard stream.
 * If flags are zero, flags appropiate to the choosen stream are used.
 * fd: STREAM_STD_*
 */
int stream_open_std(Stream* s, int fd, int flags);

enum {
	STREAM_STD_IN  = 0,
	STREAM_STD_OUT = 1,
	STREAM_STD_ERR = 2,
};

/* Return true if the stream is fully memory mapped.
 * In that case, the buffers returned are not invalidated by any operation
 * except closing the stream.
 */
static inline
bool stream_mmap_is(Stream* S) {
	return !S->cls->read && !S->cls->write && !S->cls->seek;
}

/* Loads a file completely in a memory stream.
 * Returns 0 on success, a negative error code on failure.
 */
int stream_full_file_load(Stream*restrict S, const char*restrict path);

//
#define TRY_with_file_stream(PATH, FLAGS, SVAR, EXPR) do { \
	Stream SVAR={0}; \
	const char *stm_path_ = (PATH); \
	int stm_r_ = stream_open_file(&SVAR, stm_path_, (FLAGS)); \
	if (stm_r_ < 0) { \
		ERROR_LOG(stm_r_, "could not open '%s': %d", stm_path_, stm_r_); \
	} else { \
		EXPR; \
		stream_close(&SVAR, 0); \
	} \
} while(0)

#define with_file_stream(RVAR, SVAR, PATH, FLAGS) \
	for (Stream SVAR={0}; \
		SVAR.cls != (void*)0xffffffff && \
			((RVAR) = stream_open_file(&SVAR, (PATH), (FLAGS))) == 0; \
		stream_close(&SVAR, 0), SVAR.cls=(void*)0xffffffff)

/*
	Classes
*/
extern const StreamClass stream_class_memory;

#if __STDC_HOSTED__
	extern const StreamClass stream_class_stdio;
	int stream_stdio_open_handle(Stream*restrict s, FILE*restrict f, int flags);
	int stream_stdio_open_file(Stream*restrict s, const char*restrict path, int flags);
#endif

#ifdef __WIN32__
	extern const StreamClass stream_class_winapi;
	extern const StreamClass stream_class_winapi_mmap;
	int stream_winapi_open_handle(Stream* S, void* h, int flags);
	int stream_winapi_open_file(Stream*restrict S, const char*restrict path, int flags);
	#define stream_open_file stream_winapi_open_file
#elif defined(__unix__)
	extern const StreamClass stream_class_posix;
	extern const StreamClass stream_class_posix_mmap;
	int stream_posix_open_handle(Stream* S, int fd, int flags);
	int stream_posix_open_file(Stream*restrict S, const char*restrict path, int flags);
	#define stream_open_file stream_posix_open_file
#else
	#define stream_open_file stream_stdio_open_file
#endif

/*
	Internal-use functions
*/
long stream_read_i(Stream*restrict S, size_t nbytes, void*restrict buffer);
long stream_write_i(Stream*restrict S, size_t nbytes, const void*restrict buffer);
int stream_seek_i(Stream* S, int64_t offset, int origin);

/*
	Inline implementations
*/
static inline
void* stream_read_buffer_(Stream* S, void** end)
{
	if (stream_read_prep(S, 0) < 0) {
		if (end) *end = NULL;
		return NULL;
	}
	return stream_buffer_get(S, end);
}

static inline
void* stream_write_buffer_(Stream* S, void** end)
{
	if (stream_write_prep(S, 0) < 0) {
		if (end) *end = NULL;
		return NULL;
	}
	return stream_buffer_get(S, end);
}

static inline
void* stream_buffer_get_(Stream* S, void** end)
{
	if (end) *end = S->cursor_end;
	return S->cursor;
}

static inline
void stream_commit(Stream*restrict S, const void*restrict cursor)
{
	assert( S->cursor <= (uint8_t*)cursor && (uint8_t*)cursor <= S->cursor_end );
	S->cursor = (void*)cursor;
}

static inline
void* stream_get(Stream* S, size_t nbytes)
{
	if (S->flags & SF_MODE_WRITE || !(S->cursor+nbytes <= S->cursor_end))
		if (stream_read_prep(S, nbytes) < 0) return NULL;
	uint8_t *cur = stream_buffer_get(S, NULL);
	if (cur) stream_commit(S, cur + nbytes);
	return cur;
}

static inline
void* stream_peek(Stream* S, size_t nbytes)
{
	if (S->flags & SF_MODE_WRITE || !(S->cursor+nbytes <= S->cursor_end))
		if (stream_read_prep(S, nbytes) < 0) return NULL;
	return stream_buffer_get(S, NULL);
}

static inline
int stream_unget(Stream* S, size_t nbytes) {
	return stream_seek(S, -(int64_t)nbytes, SEEK_CUR);
}

static inline
long stream_read(Stream*restrict S, size_t nbytes, void*restrict buffer)
{
	assert((long)nbytes >= 0);
	if (!(S->flags & SF_MODE_WRITE) && S->cursor+nbytes <= S->cursor_end)
	{
		memcpy(buffer, S->cursor, nbytes);
		S->cursor += nbytes;
		return nbytes;
	}
	else return stream_read_i(S, nbytes, buffer);
}

static inline
int stream_read_chk(Stream*restrict S, size_t nbytes, void*restrict buffer)
{
	long r = stream_read(S, nbytes, buffer);
	if (r < 0) return r;
	if (r < nbytes) return STREAM_E_READ;
	return 0;
}

static inline
long stream_write(Stream*restrict S, size_t nbytes, const void*restrict buffer)
{
	assert((long)nbytes >= 0);
	if (S->flags & SF_MODE_WRITE && S->cursor+nbytes <= S->cursor_end)
	{
		memcpy(S->cursor, buffer, nbytes);
		S->cursor += nbytes;
		return nbytes;
	}
	else return stream_write_i(S, nbytes, buffer);
}

static inline
int stream_write_chk(Stream*restrict S, size_t nbytes, const void*restrict buffer)
{
	long r = stream_write(S, nbytes, buffer);
	if (r < 0) return r;
	if (r < nbytes) return STREAM_E_WRITE;
	return 0;
}

static inline
int stream_seek(Stream* S, int64_t offset, int origin)
{
	switch (origin) {
	case SEEK_SET:
		if (offset >= S->pos) {
			uint8_t* cur = S->buffer + (offset - S->pos);
			if (cur <= S->cursor_end) {
				S->cursor = cur;
				return 0;
			}
		}
		break;
	case SEEK_CUR: {
		uint8_t* cur = S->cursor + offset;
		if (S->buffer <= cur && cur <= S->cursor_end) {
			S->cursor = cur;
			return 0;
		}
		} break;
	}
	return stream_seek_i(S, offset, origin);
}

static inline
int stream_char_get(Stream* S)
{
	if (S->flags & SF_MODE_WRITE || !(S->cursor < S->cursor_end))
		TRYR( stream_read_prep(S, 1) );
	return *(S->cursor++);

}

static inline
int stream_char_put(Stream* S, int c)
{
	if (!(S->flags & SF_MODE_WRITE) || !(S->cursor < S->cursor_end))
		TRYR( stream_write_prep(S, 1) );
	*S->cursor++ = c;
	return 0;	
}

static inline
int stream_str_put(Stream*restrict S, const char*restrict str) {
	size_t n = strlen(str);
	TRYR( stream_write(S, n, str) );
	return n;
}
