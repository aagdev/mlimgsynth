/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#include "fsutil.h"
#include <stdlib.h>

static inline
size_t str_copy(size_t maxlen, char* dst, const char* src) {
	if (!maxlen || !dst || !src) return 0;
	char * dst0 = dst;
	maxlen--; //null terminator
	while (maxlen-- > 1 && *src) *dst++ = *src++;
	*dst = 0;
	return dst - dst0;
}

// -----------------------------------------------------------------------------
#if defined(__unix__)
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

int file_exists(const char* path)
{
    return (access(path, F_OK) != -1);
}

int directory_make(const char* path)
{
	if (mkdir(path, 0777) < 0) {
		if (errno == EEXIST) return 0;
		return -1;
	}
	return 1;
}

static
int user_dir_get(size_t maxlen, char* out, const char* var, const char* hdir)
{
	const char * dir = getenv(var);
	if (dir) {
		return str_copy(maxlen, out, dir);
	}
	else if ((dir = getenv("HOME"))) {
		size_t i = str_copy(maxlen, out, dir), ip=i;
		i += str_copy(maxlen-i, out+i, hdir);
		if (!file_exists(out)) { out[ip] = 0; i=ip; }
		return i;
	}
	return -1;
}

int fs_dir_get(size_t maxlen, char* out, enum FsDirType type)
{
	switch (type) {
	case FS_DIR_TEMP: {
		const char * dir = getenv("TMPDIR");
		if (!dir) dir = "/tmp";
		return str_copy(maxlen, out, dir);
	}
	case FS_DIR_USER_CONFIG:
		return user_dir_get(maxlen, out, "XDG_CONFIG_HOME", "/.config");
	case FS_DIR_USER_CACHE:
		return user_dir_get(maxlen, out, "XDG_CACHE_HOME", "/.cache");
	case FS_DIR_USER_DATA:
		return user_dir_get(maxlen, out, "XDG_DATA_HOME", "/.local/.cache");
	}
	return -1;
}

// -----------------------------------------------------------------------------
#elif defined(__WIN32__)
#define WIN32_LEAN_AND_MEAN
#define WIN32_EXTRA_LEAN
#include <windows.h>

int file_exists(const char* path)
{
    DWORD dwAttrib = GetFileAttributesA(path);
    return (dwAttrib != INVALID_FILE_ATTRIBUTES);
    //return (dwAttrib != INVALID_FILE_ATTRIBUTES &&
    //        !(dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
}

int directory_make(const char* path)
{
	if (CreateDirectoryA(path, NULL) == 0) {
		if (GetLastError() == ERROR_ALREADY_EXISTS) return 0;
		return -1;
	}
	return 1;
}

static
int user_dir_get(size_t maxlen, char* out, const char* var)
{
	const char * dir;
	if (var && (dir = getenv(var))) ;
	else if ((dir = getenv("APPDATA"))) ;
	else if ((dir = getenv("USERPROFILE"))) ;
	else return -1;
	return str_copy(maxlen, out, dir);
}

int fs_dir_get(size_t maxlen, char* out, enum FsDirType type)
{
	switch (type) {
	case FS_DIR_TEMP: {
		const char * dir = getenv("TEMP");
		if (!dir) return -1;
		return str_copy(maxlen, out, dir);
	}
	case FS_DIR_USER_CONFIG:
		return user_dir_get(maxlen, out, NULL);
	case FS_DIR_USER_CACHE:
		return user_dir_get(maxlen, out, "LOCALAPPDATA");
	case FS_DIR_USER_DATA:
		return user_dir_get(maxlen, out, "LOCALAPPDATA");
	}
	return -1;
}

// -----------------------------------------------------------------------------
#else
#include <stdio.h>

int file_exists(const char* path)
{
    FILE * f = fopen(path, "r");
    if (!f) return 0;
    fclose(f);
    return 1;
}

int fs_dir_get(size_t maxlen, char* out, enum FsDirType type)
{
	return -1;
}

#endif
