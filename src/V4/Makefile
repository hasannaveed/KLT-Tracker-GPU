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
# Use optimized OpenACC C version instead of CUDA version
GPU_CU  = convolve.cu selectGoodFeatures.cu
GPU_OPT = trackFeatures_opt.c

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
GPU_OBJS = \
    $(addprefix $(OBJDIR_GPU)/,$(GPU_CU:.cu=.o)) \
    $(addprefix $(OBJDIR_GPU)/,$(GPU_OPT:.c=.o)) \
    $(addprefix $(OBJDIR_GPU)/,$(GPU_C:.c=.o))

GPU_LIB  = libklt_gpu.a
GPU_EXES = $(EXAMPLES:.c=_GPU)

.PHONY: gpu
gpu: $(GPU_LIB) $(GPU_EXES)
	@echo "✅ GPU build complete!"

$(GPU_LIB): $(GPU_OBJS)
	$(AR) rcs $@ $^

%_GPU: %.c $(GPU_LIB)
	$(NVCC) $(NVCCFLAGS) -DUSE_GPU -o $@ $< -L. -lklt_gpu -lgomp $(LIBS_GPU)

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
	$(CC) $(CFLAGS) -DUSE_GPU -fopenacc -c $< -o $@

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

######################################################################
# Profiling (CPU = gprof, GPU = nvprof)
######################################################################
PROFILES_DIR = profiles
PROFILE_FLAGS = -pg -g

.PHONY: cpu_profile gpu_profile profile gpu_profile_run all-graph clean-profile

# --------------------------------------------------------------------
# CPU profiling build (adds -pg and -g flags)
# --------------------------------------------------------------------
cpu_profile:
	$(MAKE) clean
	$(MAKE) CFLAGS="$(CFLAGS) $(PROFILE_FLAGS)" cpu
	@echo "🧭 CPU profiling build ready. Run ./exampleX_CPU to collect gmon.out data."

# --------------------------------------------------------------------
# GPU profiling build (adds debug symbols)
# --------------------------------------------------------------------
gpu_profile:
	$(MAKE) clean
	$(MAKE) NVCCFLAGS="$(NVCCFLAGS) $(PROFILE_FLAGS)" gpu
	@echo "🧭 GPU profiling build ready for nvprof."

# --------------------------------------------------------------------
# Run CPU examples under gprof and save reports to profiles/
# --------------------------------------------------------------------
profile:
	mkdir -p $(PROFILES_DIR)
	@for exe in $(EXAMPLES:.c=); do \
	    if [ -f "$${exe}_CPU" ]; then \
	        echo "=============================================="; \
	        echo "Running $${exe}_CPU for profiling..."; \
	        ./$$exe"_CPU"; \
	        echo "Generating gprof report for $${exe}_CPU..."; \
	        gprof $${exe}_CPU gmon.out > $(PROFILES_DIR)/gprof_$${exe}_CPU.txt; \
	        echo "Saved: $(PROFILES_DIR)/gprof_$${exe}_CPU.txt"; \
	    fi; \
	done
	@echo "✅ CPU profiling reports saved in $(PROFILES_DIR)/"

# --------------------------------------------------------------------
# Run GPU examples under nvprof and save reports to profiles/
# --------------------------------------------------------------------
gpu_profile_run:
	mkdir -p $(PROFILES_DIR)
	@for exe in $(EXAMPLES:.c=); do \
	    if [ -f "$${exe}_GPU" ]; then \
	        echo "=============================================="; \
	        echo "Profiling $${exe}_GPU with nvprof..."; \
	        nvprof --print-summary --log-file $(PROFILES_DIR)/nvprof_$${exe}_GPU.txt ./$$exe"_GPU"; \
	        echo "Saved: $(PROFILES_DIR)/nvprof_$${exe}_GPU.txt"; \
	    fi; \
	done
	@echo "✅ GPU profiling reports saved in $(PROFILES_DIR)/"

# --------------------------------------------------------------------
# Generate call graphs for CPU profiles
# --------------------------------------------------------------------
all-graph:
	mkdir -p $(PROFILES_DIR)
	@for txt in $(PROFILES_DIR)/gprof_*.txt; do \
	    exe=$$(basename $$txt .txt | sed 's/^gprof_//'); \
	    echo "Generating call graph for $$exe..."; \
	    gprof2dot -f prof $$txt | dot -Tpng -o $(PROFILES_DIR)/profile_$${exe}.png; \
	    echo "Saved: $(PROFILES_DIR)/profile_$${exe}.png"; \
	done

# --------------------------------------------------------------------
# Clean up profiling outputs
# --------------------------------------------------------------------
clean-profile:
	rm -rf $(PROFILES_DIR) gmon.out
	@echo "🧹 Cleaned profiling data."
