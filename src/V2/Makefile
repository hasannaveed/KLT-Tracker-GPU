######################################################################
# CUDA-ready Makefile for KLT (with CPU & GPU builds, profiling, call graphs)
######################################################################

# Compilers
CC   = gcc
NVCC = nvcc

# Flags
CFLAGS   = -O3 -pg -DNDEBUG -fPIC
NVCCFLAGS = -O3 -std=c++14 -arch=sm_75 -DNDEBUG -pg -Xcompiler -fPIC -DUSE_GPU

# Libraries
LIBS_CPU = -lm
LIBS_GPU = -lm -lcudart

######################################################################
# Source files
######################################################################

# GPU-specific CUDA code
SRC_GPU = convolve.cu computeGradientSum.cu selectGoodFeatures_GPU.cu

# Common CPU sources (used for both CPU & GPU)
SRC_CPU = error.c pnmio.c pyramid.c storeFeatures.c trackFeatures.c \
           klt.c klt_util.c writeFeatures.c selectGoodFeatures.c convolve.c

# Example programs
SRC_EXAMPLES = example1.c example2.c example3.c example4.c example5.c

# Objects
OBJ_GPU = $(SRC_GPU:.cu=.o)
OBJ_CPU = $(SRC_CPU:.c=.o)
OBJ_EXAMPLES = $(SRC_EXAMPLES:.c=.o)

# Libraries
LIB_CPU = libklt_CPU.a
LIB_GPU = libklt_GPU.a

######################################################################
# Targets
######################################################################

all: cpu gpu

######################################################################
# CPU BUILD
######################################################################

cpu: $(LIB_CPU) $(SRC_EXAMPLES:.c=_CPU)
	@echo "✅ CPU build complete."

$(LIB_CPU): $(OBJ_CPU)
	ar rcs $@ $^
	@echo "Built CPU library: $@"

$(SRC_EXAMPLES:.c=_CPU): %_CPU: %.o $(LIB_CPU)
	$(CC) $(CFLAGS) -o $@ $< $(LIB_CPU) $(LIBS_CPU)
	@echo "Built CPU example: $@"

######################################################################
# GPU BUILD
######################################################################

gpu: $(LIB_GPU) $(SRC_EXAMPLES:.c=_GPU)
	@echo "✅ GPU build complete."

$(LIB_GPU): $(OBJ_CPU) $(OBJ_GPU)
	ar rcs $@ $^
	@echo "Built GPU library: $@"

$(SRC_EXAMPLES:.c=_GPU): %_GPU: %.o $(LIB_GPU)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LIB_GPU) $(LIBS_GPU)
	@echo "Built GPU example: $@"

######################################################################
# Compilation Rules
######################################################################

# Compile C files with gcc (CPU)
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compile CUDA files with nvcc (GPU)
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

######################################################################
# Profiling and Call Graphs
######################################################################

PROFILES_DIR = profiles

profile: $(SRC_EXAMPLES:.c=_CPU)
	mkdir -p $(PROFILES_DIR)
	@for exe in $(SRC_EXAMPLES:.c=_CPU); do \
	    echo "=============================================="; \
	    echo "Profiling $$exe..."; \
	    ./$$exe; \
	    gprof $$exe gmon.out > $(PROFILES_DIR)/gprof_$${exe}.txt; \
	    echo "Saved: $(PROFILES_DIR)/gprof_$${exe}.txt"; \
	done
	@echo "All profiling reports saved in $(PROFILES_DIR)/"

all-graph:
	mkdir -p $(PROFILES_DIR)
	@for txt in $(PROFILES_DIR)/gprof_*.txt; do \
	    exe=$$(basename $$txt .txt | sed 's/^gprof_//'); \
	    echo "Generating call graph for $$exe..."; \
	    gprof2dot -f prof $$txt | dot -Tpng -o $(PROFILES_DIR)/profile_$${exe}.png; \
	    echo "Saved: $(PROFILES_DIR)/profile_$${exe}.png"; \
	done

clean-profile:
	@echo "Cleaning profiling files..."
	rm -rf $(PROFILES_DIR) gmon.out
	@echo "Profiling data cleaned successfully."

######################################################################
# Cleanup
######################################################################

clean:
	rm -f .o *.a libklt.a \
	      example*_CPU example*_GPU example1 example2 example3 example4 example5 \
	      feat*.ppm features.ft features.txt *.tar *.tar.gz
	@echo "Cleaned all build files."

######################################################################
# End of Makefile
#####################################################################