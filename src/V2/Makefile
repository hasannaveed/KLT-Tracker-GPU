######################################################################
# CUDA-ready Makefile for KLT (with computeGradientSum.cu)
######################################################################

# CUDA compiler
NVCC = nvcc

# Compiler flags
NVCCFLAGS = -O3 -Xcompiler -fPIC -DNDEBUG -pg

# Libraries
LIBS = -lm -lcudart

# Source files for the library
SRC_LIB = convolve.cu computeGradientSum.cu selectGoodFeatures_GPU.cu error.c pnmio.c pyramid.c \
           storeFeatures.c trackFeatures.c klt.c \
          klt_util.c writeFeatures.c

# Source files for examples
SRC_EXAMPLES = example1.c example2.c example3.c example4.c example5.c

# Object files
OBJ_LIB = $(SRC_LIB:.c=.o)
OBJ_LIB := $(OBJ_LIB:.cu=.o)

OBJ_EXAMPLES = $(SRC_EXAMPLES:.c=.o)

# Library name
LIB = libklt.a

######################################################################
# Default target
######################################################################

all: $(SRC_EXAMPLES:.c=)

######################################################################
# Compilation rules
######################################################################

# Compile CUDA files
%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

# Compile C files
%.o: %.c
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

######################################################################
# Build static library
######################################################################

$(LIB): $(OBJ_LIB)
	ar ruv $(LIB) $(OBJ_LIB)
	@echo "Static library $(LIB) built."

######################################################################
# Build examples
######################################################################

$(SRC_EXAMPLES:.c=): %: %.o $(LIB)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LIB) $(LIBS)
	@echo "Built example: $@"

######################################################################
# Dependency (optional)
######################################################################

depend:
	makedepend $(SRC_LIB) $(SRC_EXAMPLES)

######################################################################
# Clean up
######################################################################

clean:
	rm -f $(OBJ_LIB) $(OBJ_EXAMPLES) $(SRC_EXAMPLES:.c=) $(LIB)
	@echo "Cleaned all build files."

######################################################################
# Profiling (optional)
######################################################################

PROFILES_DIR = profiles

profile: $(SRC_EXAMPLES:.c=)
	mkdir -p $(PROFILES_DIR)
	@for exe in $(SRC_EXAMPLES:.c=); do \
	    echo "=============================================="; \
	    echo "Running $$exe for profiling..."; \
	    ./$$exe; \
	    echo "Generating gprof report for $$exe..."; \
	    gprof $$exe gmon.out > $(PROFILES_DIR)/gprof_$$exe.txt; \
	    echo "Saved: $(PROFILES_DIR)/gprof_$$exe.txt"; \
	done
	@echo "All profiling reports saved in $(PROFILES_DIR)/"

all-graph:
	mkdir -p $(PROFILES_DIR)
	@for txt in $(PROFILES_DIR)/gprof_*.txt; do \
	    exe=$$(basename $$txt .txt | sed 's/^gprof_//'); \
	    echo "Generating call graph for $$exe..."; \
	    gprof2dot -f prof $$txt | dot -Tpng -o $(PROFILES_DIR)/profile_$$exe.png; \
	    echo "Saved: $(PROFILES_DIR)/profile_$$exe.png"; \
	done


clean-profile:
	rm -rf $(PROFILES_DIR) gmon.out
	@echo "Cleaned profiling data."
