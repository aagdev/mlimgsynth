/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 *
 * Logging interface
 */
#pragma once
#include "stream.h"
#include <stdarg.h>
#include <stdbool.h>

//TODO: interface to use other loggers

// Levels
#define LOG_LVL_STEP 10

enum LoggingLevel {
	LOG_LVL_NONE	= 0,
	LOG_LVL_ERROR	= LOG_LVL_STEP,
	LOG_LVL_WARNING	= LOG_LVL_STEP*2,
	LOG_LVL_INFO	= LOG_LVL_STEP*3, //normal
	LOG_LVL_INFO2	= LOG_LVL_STEP*4, //verbose
	LOG_LVL_DEBUG	= LOG_LVL_STEP*5,
	LOG_LVL_DEBUG2	= LOG_LVL_STEP*6,
	LOG_LVL_MAX		= 255
};

// Utility macros, use mostly these
#define log_error(...)		log_log(LOG_LVL_ERROR, __VA_ARGS__)
#define log_warning(...)	log_log(LOG_LVL_WARNING, __VA_ARGS__)
#define log_info(...)		log_log(LOG_LVL_INFO, __VA_ARGS__)
#define log_info2(...)		log_log(LOG_LVL_INFO2, __VA_ARGS__)
#define log_debug(...)		log_log(LOG_LVL_DEBUG, __VA_ARGS__)
#define log_debug2(...)		log_log(LOG_LVL_DEBUG2, __VA_ARGS__)

#define log_log(LVL, ...) do {\
	if (log_level_check((LVL))) \
		log_logf((LVL), __VA_ARGS__); \
} while (0)
	
#define log_log_str(LVL, STR) do {\
	if (log_level_check((LVL))) \
		log_logs((LVL), (STR)); \
} while (0)

// Interface
struct Logger {
	int			level;
	Stream *	stm;
};

extern struct Logger g_logger;

static inline
bool log_level_check(int level)
{
	return level <= g_logger.level;
}

static inline
int log_level_set(int level)
{
	int oldval = g_logger.level;
	g_logger.level = level;
	return oldval;
}

static inline
int log_level_inc(int change)
{
	int oldval = g_logger.level;
	g_logger.level += change;
	return oldval;
}

void log_logs(int level, const char* text);

#ifdef __GNUC__
__attribute__((format(printf, 2, 0)))
#endif
void log_logv(int level, const char format[], va_list ap);

#ifdef __GNUC__
__attribute__((format(printf, 2, 3)))
#endif
void log_logf(int level, const char format[], ...);

// Low level interface, no checking
//TODO: join line_begin and level_check
void log_line_begin(int level);

void log_line_str(const char* str);

#ifdef __GNUC__
__attribute__((format(printf, 1, 0)))
#endif
void log_line_strv(const char format[], va_list ap);

#ifdef __GNUC__
__attribute__((format(printf, 1, 2)))
#endif
void log_line_strf(const char format[], ...);

void log_line_end();
