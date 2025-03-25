# Makefile
# Copyright 2024, Alejandro A. García <aag@zorzal.net>
# SPDX-License-Identifier: Zlib

# Common make definitions and rules for use in multiple projects.
# 
# Use VPATH to configure the source directories
# e.g. VPATH = src:src/ccommon
#
# Append to cppflags to add include directories
# e.g. cppflags += -Isrc/ccommon

objdir = obj
depdir = .d

flags    = $(FLAGS)
cppflags = $(CPPFLAGS)
cflags   = -std=c99 -Wall -pedantic $(CFLAGS)
cxxflags = $(CXXFLAGS)
ldlibs   = $(LDLIBS)
ldflags  = $(LDFLAGS)

depflags = -MT $@ -MMD -MP -MF $(depdir)/$*.d

# Options for dynamic libraries
flags += -fPIC -fvisibility=hidden

### Compilation options
ifndef nonative
cppflags += -march=native
endif

ifdef debug
objdir = obj_dbg
depdir = .d_dbg
#cppflags += -g -DDEBUG
cppflags += -ggdb -g3 -DDEBUG
else ifdef debugo
objdir = obj_dbg
depdir = .d_dbg
cppflags += -ggdb -g3 -DDEBUG
flags += -Og
else ifdef small
cppflags += -DNDEBUG
ldflags += -Wl,--strip-all
flags += -Os
else ifdef fast
cppflags += -DNDEBUG
ldflags += -Wl,--strip-all
flags += -O3
flags += -flto -fwhole-program -fuse-linker-plugin
else
cppflags += -DNDEBUG
ldflags += -Wl,--strip-all
flags += -O2
endif

ifdef profile
flags += -pg
$(info gprof CMD gmon.out | less)
endif

###
.PHONY: all clean

# Disable implicit rules
.SUFFIXES:
#.SUFFIXES: .c .o

# Do not remove intermediate files
.SECONDARY:

### OS specifics
ifeq ($(OS),Windows_NT)
EXEC_EXT=.exe
DLIB_EXT=.dll
RUN_PRE=
targets_bin = $(addsuffix $(EXEC_EXT),$(targets)) $(addsuffix $(DLIB_EXT),$(targets_dlib))
targets_bin2 = $(targets)
else
EXEC_EXT=
DLIB_EXT=.so
RUN_PRE=./
targets_bin = $(targets) $(addsuffix $(DLIB_EXT),$(targets_dlib))
targets_bin2 = 
endif

### Commands
COMPILE_C   = $(CC)  $(depflags) $(flags) $(cppflags) $(cflags)   -c -o $@ $<
COMPILE_CXX = $(CXX) $(depflags) $(flags) $(cppflags) $(cxxflags) -c -o $@ $<
LINK_EXEC = $(CC) $(flags) $(ldflags) -o $@$(EXEC_EXT) \
	$(addprefix $(objdir)/,$(filter %.o,$^)) $(ldlibs)
LINK_DLIB = $(CC) $(flags) $(ldflags) -shared -o $@$(DLIB_EXT) \
	$(addprefix $(objdir)/,$(filter %.o,$^)) $(ldlibs)

### Some commonly used dependencies
#$(info OS=$(OS))
ifeq ($(OS),Windows_NT)
socket_libs = -lws2_32
sdl_libs = -lmingw32 -lSDL2main -lSDL2
else
#socket_libs =
sdl_libs = -lSDL2main -lSDL2
endif
sdl_objs += image_sdl.o

### Rules
all: $(targets_dlib) $(targets)

$(targets): | $(objdir) $(depdir)
ifdef verbose
	$(LINK_EXEC)
else
	@echo "LINK $@"
	@$(LINK_EXEC)
endif
ifdef run
	$(RUN_PRE)$@
endif
ifdef gdb
	gdb $@
endif

$(targets_dlib): | $(objdir) $(depdir)
ifdef verbose
	$(LINK_DLIB)
else
	@echo "LINK $@"
	@$(LINK_DLIB)
endif

$(objdir):
	mkdir -p $(objdir)

%.o: $(objdir)/%.o ;

$(objdir)/%.o: %.c
ifdef verbose
	$(COMPILE_C)
else
	@echo "CC $@"
	@$(COMPILE_C)
endif

$(objdir)/%.o: %.cpp
ifdef verbose
	$(COMPILE_CXX)
else
	@echo "CXX $@"
	@$(COMPILE_CXX)
endif

###
$(depdir):
	mkdir -p $(depdir)

$(depdir)/%.d: ;

.PRECIOUS: $(depdir)/%.d

include $(wildcard $(depdir)/*.d)

### Clean-up rules
cleanbin:
	rm -f $(targets_bin) $(targets_bin2)

clean: cleanbin
	rm -f $(objdir)/* $(depdir)/*
	-rm -f gmon.out *.gcov

distclean: cleanbin
	rm -fr obj obj_dbg .d .d_dbg
	-rm -f gmon.out *.gcov

### Some shorthands
run_%: %
	@echo ""
	$(RUN_PRE)$<

test: $(addprefix run_,$(tests))
