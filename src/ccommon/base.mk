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
flags += -Os
else ifdef fast
cppflags += -DNDEBUG
flags += -O3
flags += -flto -fwhole-program -fuse-linker-plugin
else
cppflags += -DNDEBUG
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

### Commands
COMPILE_C   = $(CC)  $(depflags) $(flags) $(cppflags) $(cflags)   -c -o $@ $<
COMPILE_CXX = $(CXX) $(depflags) $(flags) $(cppflags) $(cxxflags) -c -o $@ $<
#LINK = $(CC) $(flags) $(ldflags) -o $@ $^ $(ldlibs)
LINK = $(CC) $(flags) $(ldflags) -o $@ \
	$(addprefix $(objdir)/,$(filter %.o,$^)) $(ldlibs)

### OS specifics
ifeq ($(OS),Windows_NT)
BIN_EXT=.exe
RUN_PRE=
targets_bin = $(addsuffix $(BIN_EXT),$(targets))
targets_bin2 = $(targets)
else
BIN_EXT=
RUN_PRE=./
targets_bin = $(targets)
targets_bin2 = 
endif

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
all: $(targets)

remake: clean all

$(targets): | $(objdir) $(depdir)
ifdef verbose
	$(LINK)
else
	@echo "LINK $@"
	@$(LINK)
endif
ifdef run
	$(RUN_PRE)$@
endif
ifdef gdb
	gdb $@
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
