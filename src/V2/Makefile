######################################################################
# CUDA-ready Makefile for KLT (CPU/GPU variants)
######################################################################

# Compilers
CC   = gcc
NVCC = nvcc

# Flags
CFLAGS    = -O3 -DNDEBUG -pg -fPIC
NVCCFLAGS = -O3 -std=c++14 -arch=sm_75 -DNDEBUG -pg -Xcompiler -fPIC

# Libraries
LIBS      = -lm
NVLIBS    = -lm -lcudart

# Sources
SRC_LIB_CPU = convolve.c selectGoodFeatures.c error.c pnmio.c pyramid.c \
              storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c

SRC_LIB_GPU = convolve.cu computeGradientSum.cu selectGoodFeatures_GPU.cu error.c pnmio.c pyramid.c \
              storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c

SRC_EXAMPLES = example1.c example2.c example3.c

# Object files
OBJ_LIB_CPU = $(SRC_LIB_CPU:.c=.o)
OBJ_LIB_GPU = $(SRC_LIB_GPU:.c=.o)
OBJ_EXAMPLES = $(SRC_EXAMPLES:.c=.o)

# Library names
LIB_CPU = libklt_CPU.a
LIB_GPU = libklt_GPU.a

######################################################################
# Default target
######################################################################
all: cpu gpu

######################################################################
# CPU Build
######################################################################
cpu: $(SRC_EXAMPLES:.c=_CPU)

# Compile CPU objects
%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

# Build CPU library
$(LIB_CPU): $(OBJ_LIB_CPU)
	ar ruv $(LIB_CPU) $(OBJ_LIB_CPU)
	@echo "CPU library $(LIB_CPU) built."

# Build CPU examples
%_CPU: %.o $(LIB_CPU)
	$(CC) $(CFLAGS) -o $@ $< $(LIB_CPU) $(LIBS)
	@echo "Built CPU example: $@"

######################################################################
# GPU Build
######################################################################
gpu: $(SRC_EXAMPLES:.c=_GPU)

# Compile GPU objects
%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

# Build GPU library
$(LIB_GPU): $(OBJ_LIB_GPU)
	ar ruv $(LIB_GPU) $(OBJ_LIB_GPU)
	@echo "GPU library $(LIB_GPU) built."

# Build GPU examples
%_GPU: %.o $(LIB_GPU)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LIB_GPU) $(NVLIBS)
	@echo "Built GPU example: $@"

######################################################################
# Profiling & Call Graphs (CPU only)
######################################################################
PROFILES_DIR = profiles

profile: cpu
	mkdir -p $(PROFILES_DIR)
	@for exe in $(SRC_EXAMPLES:.c=_CPU); do \
	    echo "=============================================="; \
	    echo "Profiling $$exe..."; \
	    ./$$exe; \
	    gprof $$exe gmon.out > $(PROFILES_DIR)/gprof_$$exe.txt; \
	    echo "Saved: $(PROFILES_DIR)/gprof_$$exe.txt"; \
	done

all-graph: profile
	@for txt in $(PROFILES_DIR)/gprof_*.txt; do \
	    exe=$$(basename $$txt .txt | sed 's/^gprof_//'); \
	    echo "Generating call graph for $$exe..."; \
	    gprof2dot -f prof $$txt | dot -Tpng -o $(PROFILES_DIR)/profile_$$exe.png; \
	    echo "Saved: $(PROFILES_DIR)/profile_$$exe.png"; \
	done

######################################################################
# Clean
######################################################################
clean:
	rm -f *.o *.a $(EXAMPLES:.c=) *.tar *.tar.gz libklt.a \
	      feat*.ppm features.ft features.txt
	@echo "Cleaned all build files."

clean-profile:
	rm -rf $(PROFILES_DIR) gmon.out
	@echo "Cleaned profiling data."