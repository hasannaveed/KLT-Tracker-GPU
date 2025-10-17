######################################################################
# Choose your compilers
######################################################################
CC   = gcc
NVCC = nvcc

######################################################################
# Compilation Flags
######################################################################
FLAG1  = -DNDEBUG
CFLAGS = -O3 -pg $(FLAG1)
NVFLAGS = -std=c++14 -arch=sm_75

######################################################################
# Source Files
######################################################################
EXAMPLES = example1.c example2.c example3.c example4.c example5.c
ARCH = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
       storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c

######################################################################
# Libraries
######################################################################
LIB = -L/usr/local/lib -L/usr/lib -lm -lcudart
INCLUDE = -I/usr/local/cuda/include

######################################################################
# Object Files
######################################################################
OBJS = $(ARCH:.c=.o) computeGradientSum.o

######################################################################
# Default Target
######################################################################
all: libgpu $(EXAMPLES:.c=)

######################################################################
# Compile Rules
######################################################################
.c.o:
	$(CC) -c $(CFLAGS) $< -o $@

computeGradientSum.o: computeGradientSum.cu
	$(NVCC) -c $(NVFLAGS) computeGradientSum.cu -o computeGradientSum.o

######################################################################
# Library Build
######################################################################
libgpu: $(OBJS)
	rm -f libklt_gpu.a
	ar ruv libklt_gpu.a $(OBJS)
	rm -f $(ARCH:.c=.o)

######################################################################
# Example Builds (GPU linked)
######################################################################
example1: libklt_gpu.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt_gpu $(LIB)

example2: libklt_gpu.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt_gpu $(LIB)

example3: libklt_gpu.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt_gpu $(LIB)

example4: libklt_gpu.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt_gpu $(LIB)

example5: libklt_gpu.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt_gpu $(LIB)

######################################################################
# Profiling Targets
######################################################################
PROFILES_DIR = profiles

# Generate gprof text reports for all examples
profile: $(EXAMPLES:.c=)
	mkdir -p $(PROFILES_DIR)
	@for exe in $(EXAMPLES:.c=); do \
	    echo "=============================================="; \
	    echo "Running $$exe for profiling..."; \
	    ./$$exe; \
	    echo "Generating gprof report for $$exe..."; \
	    gprof $$exe gmon.out > $(PROFILES_DIR)/gprof_$$exe.txt; \
	    echo "Saved: $(PROFILES_DIR)/gprof_$$exe.txt"; \
	done
	@echo "=============================================="
	@echo "All gprof text reports generated successfully in $(PROFILES_DIR)/"

# Generate call graph PNGs from gprof text reports
all-graph:
	mkdir -p $(PROFILES_DIR)
	@for txt in $(PROFILES_DIR)/gprof_*.txt; do \
	    exe=$$(basename $$txt .txt | sed 's/^gprof_//'); \
	    echo "Generating call graph for $$exe..."; \
	    gprof2dot -f prof $$txt | dot -Tpng -o $(PROFILES_DIR)/profile_$$exe.png; \
	    echo "Saved: $(PROFILES_DIR)/profile_$$exe.png"; \
	done
	@echo "=============================================="
	@echo "All call graph images generated successfully in $(PROFILES_DIR)/"

# Clean all profiling-related files (safe to use)
clean-profile:
	@echo "Cleaning profiling files..."
	rm -rf $(PROFILES_DIR) gmon.out
	@echo "Profiling data cleaned successfully."

######################################################################
# Cleanup
######################################################################
clean:
	rm -f *.o *.a $(EXAMPLES:.c=) *.ppm *.txt *.fl *.pgm libklt_gpu.a
