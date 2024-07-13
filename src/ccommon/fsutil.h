/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 *
 * File system utility functions
 */
#pragma once
#include <string.h>
#include <stdbool.h>

// Returns the last part of a path.
// Example: "dir/name.ext" -> "name.ext"
static inline
char* path_tail(const char* path);

// Returns the file name extension.
// Examples: "dir/name.ext" -> "ext", "name" -> "" (pointer to end)
static inline
char* path_ext(const char* path);

static inline
bool path_abs_is(const char* path);

static inline
bool path_sep_is(int c);

// Returns 1 if it exists, 0 otherwise
int file_exists(const char* path);

// Returns 1 on creation, 0 if already exists and <0 on error
int directory_make(const char* path);

enum FsDirType {
	FS_DIR_TEMP = 1,
	FS_DIR_USER_CONFIG,
	FS_DIR_USER_CACHE,
	FS_DIR_USER_DATA,
};
// Writes to out the path to choosen system directory.
// Returns the number of bytes written, <0 on error
int fs_dir_get(size_t maxlen, char* out, enum FsDirType type);

/* Inline implementations */
static inline
char* path_tail(const char* path)
{
	int i = strlen(path);
	for (i--; i>=0; --i) if (path_sep_is(path[i])) return (char*)(path+i+1);
	return (char*)path;
}

static inline
char* path_ext(const char* path)
{
	int n = strlen(path);
	for (int i=n-1; i>=0; --i) if (path[i] == '.') return (char*)(path+i+1);
	return (char*)path+n;  //empty
}

static inline
bool path_abs_is(const char* path)
{
	if (path[0] == '/') return true;
#ifdef __WIN32__
	if (path[0] == '\\') return true;
	if (path[0] && path[1] == ':') return true;
#endif
	return false;
}

static inline
bool path_sep_is(int c)
{
	if (c == '/') return true;
#ifdef __WIN32__
	if (c == '\\') return true;
#endif
	return false;
}
