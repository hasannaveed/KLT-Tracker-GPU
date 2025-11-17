# =========================
# Compilers and tools
# =========================
CC       = gcc
NVC      = nvc
AR       = ar

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


