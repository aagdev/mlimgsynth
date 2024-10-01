/* Copyright 2024, Alejandro A. García <aag@zorzal.net>
 * SPDX-License-Identifier: Zlib
 */
#include "logging.h"
//#include <stdio.h>

struct Logger g_logger = {
#ifdef DEBUG
	.level	= LOG_LVL_DEBUG,
#else
	.level	= LOG_LVL_INFO,
#endif
};

Stream g_logger_stream;

void log_line_begin_raw(int level)
{
	if (!g_logger.stm) {
		stream_open_std(&g_logger_stream, STREAM_STD_ERR, 0);		
		g_logger.stm = &g_logger_stream;
	}

	const char * lvl_prefix = 0;
	if      (level >= LOG_LVL_DEBUG)   lvl_prefix = "DEBUG ";
	else if (level >= LOG_LVL_INFO)    ;
	else if (level >= LOG_LVL_WARNING) lvl_prefix = "WARN  ";
	else                               lvl_prefix = "ERROR ";

	//TODO time

	if (lvl_prefix)
		stream_str_put(g_logger.stm, lvl_prefix);
}

void log_line_str(const char* str)
{
	stream_str_put(g_logger.stm, str);
}

#if __STDC_HOSTED__
void log_line_strv(const char format[], va_list ap)
{
	stream_vprintf(g_logger.stm, format, ap);
}

void log_line_strf(const char format[], ...)
{
	va_list ap;
	va_start(ap, format);
	log_line_strv(format, ap);
	va_end(ap);
}
#endif

void log_line_end()
{
	stream_char_put(g_logger.stm, '\n');
	stream_flush(g_logger.stm);
}

void log_logs(int level, const char* text)
{
	if (!text) return;
	if (!log_level_check(level)) return;
	log_line_begin(level);
	log_line_str(text);
	log_line_end();
}

#if __STDC_HOSTED__
void log_logv(int level, const char format[], va_list ap)
{
	if (!format) return;
	if (!log_level_check(level)) return;
	log_line_begin(level);
	log_line_strv(format, ap);
	log_line_end();
}

void log_logf(int level, const char format[], ...)
{
	if (!format) return;
	if (!log_level_check(level)) return;
	log_line_begin(level);

	va_list ap;
	va_start(ap, format);
	log_line_strv(format, ap);
	va_end(ap);

	log_line_end();
}
#endif
