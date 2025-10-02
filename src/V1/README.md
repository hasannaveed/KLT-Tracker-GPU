# CCP HPC Project – Deliverable 1 (V1 Baseline)

This repository contains the **baseline sequential implementation (V1)** of the Kanade–Lucas–Tomasi (KLT) Tracker, developed as part of the CCP High Performance Computing course project.

---

## 📂 Repository Structure
CCP-HPC-KLT/
├── src/V1/ # Baseline source code and Makefile
├── profiles/ # Profiling outputs (gprof results)
├── report/ # Project reports (Deliverables)
├── scripts/ # Scripts/config files (if needed later)
├── bin/ # (Optional) compiled binaries
└── README.md # Project documentation

## ⚙️ Build Instructions
1. Open Ubuntu (or SSH to the university server).  
2. Navigate to the V1 folder:
   ```bash
   cd src/V1

Build everything:
make all


This generates the following executables:
example1  example2  example3  example4  example5

Clean build files:
make clean


▶️ Running Examples
Each example demonstrates the KLT tracker on test image sequences.

Run an example:
./example1
This produces:

feat*.ppm → images with tracked features drawn

features.txt → feature coordinates

You can also run:
./example2
./example3
./example4
./example5


📊 Profiling (for Deliverable 1)
We profiled the baseline (V1) to identify computational hotspots.

Steps:

Rebuild with profiling enabled (-pg is already added to CFLAGS):
make clean
make all
Run an example (e.g., small workload):


./example1
This generates gmon.out.

Produce profiling report:
gprof example1 gmon.out > profiles/gprof_example1.txt

Repeat with a larger workload (e.g., example3 or example4):
./example3
gprof example3 gmon.out > profiles/gprof_example3.txt

Profiling Outputs:
Saved in profiles/ folder (e.g., gprof_example1.txt, gprof_example3.txt).