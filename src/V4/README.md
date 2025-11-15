KLT Dual CPU + GPU Build and Profiling Guide

This project provides a unified Makefile to build and profile both CPU and GPU implementations of the KLT (Kanade-Lucas-Tomasi) feature tracker. It supports compiling, profiling, and generating reports for CPU (GCC) and GPU (CUDA/NVCC) versions, as well as managing clean builds and automated profiling pipelines.

1. Overview

The Makefile handles compilation for two separate targets:

CPU Build: Uses gcc to compile .c source files into a static library libklt_cpu.a and a set of example executables (e.g., example1_CPU).

GPU Build: Uses nvcc to compile .cu CUDA source files into libklt_gpu.a and GPU-enabled executables (e.g., example1_GPU).

The structure supports mixed codebases, automatically separating .c and .cu files while managing shared code between both versions.

2. Directory and File Layout
src/
│
├── convolve.c            # CPU convolution
├── convolve.cu           # Optimized GPU convolution
├── selectGoodFeatures.c  # CPU feature selection
├── selectGoodFeatures.cu # Optimized GPU feature selection
├── trackFeatures.c       # CPU feature tracking
├── trackFeatures.cu      # Optimized GPU feature tracking
├── error.c, pnmio.c, pyramid.c, etc. # Common shared sources
│
├── example1.c, example2.c, ...       # Example programs
│
├── obj_cpu/              # Object files for CPU build
├── obj_gpu/              # Object files for GPU build
└── profiles/             # Generated profiling data and reports

3. Building Targets
CPU Build

Compile all CPU-only files and link them into libklt_cpu.a:
```bash
make cpu
```


This produces:

libklt_cpu.a (static library)

Example executables: example1_CPU, example2_CPU, etc.

When finished, you’ll see:

✅ CPU build complete!

GPU Build

Compile all GPU and shared source files with CUDA enabled:

```bash
make gpu
```


This produces:

libklt_gpu.a

Example executables: example1_GPU, example2_GPU, etc.

When finished, you’ll see:

🚀 GPU build complete!

4. Cleaning Builds

To remove all object files, libraries, executables, and profiling outputs:

```bash
make clean
```


You’ll see:

🧹 Cleaned all build files.

5. Profiling
CPU Profiling (with gprof)

Rebuild the CPU version with profiling flags:

```bash
make cpu_profile
```

Then run an example, for example:

./example3_CPU


After running, a gmon.out file is generated. You can automatically produce reports for all CPU examples using:

```bash
make profile
```

This saves individual profiling reports (e.g., profiles/gprof_example3_CPU.txt).

To visualize call graphs:

```bash
make all-graph
```

This converts reports into PNG graphs (e.g., profiles/profile_example3_CPU.png).

GPU Profiling (with Nsight Systems)

Rebuild the GPU version with debug symbols:

```bash
make gpu_profile
```

Then use Nsight Systems to profile the desired example manually:

```bash
nsys profile -t cuda --force-overwrite=true -o profiles/report_example3_GPU ./example3_GPU
```

This generates a .qdrep report (e.g., profiles/report_example3_GPU.qdrep).

Convert this binary report into a human-readable text summary:

```bash
nsys stats report_example3_GPU.qdrep > profiles/report_example3_GPU.txt
```

You can then open or analyze profiles/report_example3_GPU.txt for CUDA kernel timings and performance summaries.

Alternatively, to automatically profile all GPU examples and export text summaries:

```bash
make gpu_nsys
```

This saves reports like:

profiles/nsys_example1_GPU.txt
profiles/nsys_example2_GPU.txt
...

6. Cleaning Profiling Data

To remove all generated profiling reports and logs:

``` bash
make clean-profile
```

You’ll see:

🧹 Cleaned profiling data.

7. Summary of Key Commands
Task	Command
Build CPU version	make cpu
Build GPU version	make gpu
Clean all builds	make clean
Build CPU with profiling	make cpu_profile
Build GPU with profiling	make gpu_profile
Run CPU profiling and generate reports	make profile
Generate CPU call graphs	make all-graph
Profile GPU example manually	nsys profile -t cuda --force-overwrite=true -o profiles/report_example3_GPU ./example3_GPU
Convert GPU profiling report to text	nsys stats report_example3_GPU.qdrep > profiles/report_example3_GPU.txt
Remove profiling outputs	make clean-profile