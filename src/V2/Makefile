######################################################################
# Dual CPU + GPU Makefile for KLT
######################################################################

CC   = gcc
NVCC = nvcc
AR   = ar

CFLAGS    = -O3 -DNDEBUG -fPIC -Wall
NVCCFLAGS = -O3 -std=c++14 -arch=sm_75 -DNDEBUG -Xcompiler -fPIC
LIBS_CPU  = -lm
LIBS_GPU  = -lm -lcudart

COMMON_C = error.c pnmio.c pyramid.c storeFeatures.c klt.c klt_util.c writeFeatures.c
EXAMPLES = example1.c example2.c example3.c example4.c example5.c

CPU_SRC = convolve.c selectGoodFeatures.c trackFeatures.c $(COMMON_C)
GPU_CU  = convolve.cu selectGoodFeatures.cu trackFeatures.cu

# Only include .c files that don't have .cu counterparts
GPU_C = $(filter-out convolve.c selectGoodFeatures.c trackFeatures.c, $(COMMON_C))

OBJDIR_CPU = obj_cpu
OBJDIR_GPU = obj_gpu

######################################################################
# CPU Build
######################################################################
CPU_OBJS = $(addprefix $(OBJDIR_CPU)/,$(CPU_SRC:.c=.o))
CPU_LIB  = libklt_cpu.a
CPU_EXES = $(EXAMPLES:.c=_CPU)

.PHONY: cpu
cpu: $(CPU_LIB) $(CPU_EXES)
	@echo "✅ CPU build complete!"

$(CPU_LIB): $(CPU_OBJS)
	$(AR) rcs $@ $^

%_CPU: %.c $(CPU_LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lklt_cpu $(LIBS_CPU)

######################################################################
# GPU Build
######################################################################
GPU_OBJS = $(addprefix $(OBJDIR_GPU)/,$(GPU_CU:.cu=.o)) \
           $(addprefix $(OBJDIR_GPU)/,$(GPU_C:.c=.o))
GPU_LIB  = libklt_gpu.a
GPU_EXES = $(EXAMPLES:.c=_GPU)

.PHONY: gpu
gpu: $(GPU_LIB) $(GPU_EXES)
	@echo "🚀 GPU build complete!"

$(GPU_LIB): $(GPU_OBJS)
	$(AR) rcs $@ $^

%_GPU: %.c $(GPU_LIB)
	$(NVCC) $(NVCCFLAGS) -DUSE_GPU -o $@ $< -L. -lklt_gpu $(LIBS_GPU)

######################################################################
# Compilation Rules
######################################################################
.SUFFIXES: .cu .c .o

# --- CUDA compilation rule (must come first!) ---
$(OBJDIR_GPU)/%.o: %.cu
	@mkdir -p $(OBJDIR_GPU)
	$(NVCC) $(NVCCFLAGS) -DUSE_GPU -c $< -o $@

# --- GPU C fallback (for .c files without .cu) ---
$(OBJDIR_GPU)/%.o: %.c
	@mkdir -p $(OBJDIR_GPU)
	$(CC) $(CFLAGS) -DUSE_GPU -c $< -o $@

# --- CPU C compilation ---
$(OBJDIR_CPU)/%.o: %.c
	@mkdir -p $(OBJDIR_CPU)
	$(CC) $(CFLAGS) -c $< -o $@

######################################################################
# Clean
######################################################################
.PHONY: clean
clean:
	rm -rf $(OBJDIR_CPU) $(OBJDIR_GPU)
	rm -f *.a *_CPU *_GPU feat.ppm features.* *.tar *.tar.gz
	@echo "🧹 Cleaned all build files."
