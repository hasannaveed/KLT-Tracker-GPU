# =========================
# Compilers and tools
# =========================
CC       = gcc
NVC      = nvc
AR       = ar

# =========================
# Flags
# =========================
CFLAGS   = -O3 -DNDEBUG -fPIC -Wall
NVCFLAGS = -O3 -acc -gpu=cc75 -DNDEBUG -fPIC -Minfo=accel
LIBS     = -lm -lcudart

# =========================
# Source files
# =========================
CPU_SRC  =  error.c pnmio.c pyramid.c storeFeatures.c klt.c klt_util.c writeFeatures.c trackFeatures.c
GPU_SRC  = convolve_opt.c selectGoodFeatures_opt.c 
EXAMPLES := $(wildcard example*.c)

OBJ_CPU  = $(CPU_SRC:.c=.o)
OBJ_GPU  = $(GPU_SRC:.c=.o)
GPU_LIB  = libklt_gpu.a
EXECS    = $(EXAMPLES:.c=_GPU)

# =========================
# Default target
# =========================
all: $(EXECS)

# =========================
# Build GPU library
# =========================
$(GPU_LIB): $(OBJ_GPU)
	$(AR) rcs $@ $^

# =========================
# Compile CPU objects
$(OBJ_CPU): %.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compile GPU objects
$(OBJ_GPU): %.o: %.c
	$(NVC) $(NVCFLAGS) -c $< -o $@

# =========================
# Build example executables
%_GPU: %.c $(OBJ_CPU) $(GPU_LIB)
	$(NVC) $(NVCFLAGS) -o $@ $< $(OBJ_CPU) -L. -lklt_gpu $(LIBS)

# =========================
# Clean
.PHONY: clean
clean:
	rm -f $(OBJ_CPU) $(OBJ_GPU) $(GPU_LIB) $(EXECS)