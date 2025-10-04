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

## Insatallation guidelines:

🧱 1️⃣ System Update
```bash
sudo apt update && sudo apt upgrade -y
```

🧰 2️⃣ Install GCC (C Compiler)
```bash
sudo apt install build-essential -y
```

🧪 3️⃣ Install gprof (GNU Profiler)
```bash
sudo apt install gprof -y
```

🧮 4️⃣ Install Graphviz (for PNG call graphs)
```bash
sudo apt install graphviz -y
```

🐍 5️⃣ Install Python & pipx (for gprof2dot)
```bash
sudo apt install python3 python3-pip pipx -y
```

🧠 6️⃣ Install gprof2dot (Python-based graph generator)
```bash
pipx install gprof2dot
```

## ⚙️ Build Instructions
1. Open Ubuntu (or SSH to the university server).  
2. Navigate to the V1 folder:
   ```bash
   cd src/V1

Step 1: Clean
   ```bash
   make clean
   ```

Step 2: Build everything
```bash
   make all
```

This generates the following executables:
example1  example2  example3  example4  example5


▶️ Running Examples
Each example demonstrates the KLT tracker on test image sequences.

Run an example:
```bash
./example1
```
This produces:
feat*.ppm → images with tracked features drawn

features.txt → feature coordinates

You can also run:
```bash
./example2
./example3
./example4
./example5
```


🧩 Profiling

To generate profiling data using gprof:
```bash
make profile-all
```

This will:

Run all example programs.
Generate profiling reports inside the profiles/ directory:
```bash
profiles/gprof_example1.txt
profiles/gprof_example2.txt
```

Run profiling for a single example (optional):
```bash
./example1
gprof example1 gmon.out > profiles/gprof_example1.txt
```
🕸 Call Graph Visualization

To generate function call graphs from the profiling data:
```bash
make all-graphs
```

This will produce .png files in the same profiles/ folder:

profiles/callgraph_example1.png
profiles/callgraph_example2.png

For single file run this:
```bash
gprof example1 gmon.out > profiles/gprof_example1.txt
gprof2dot -f prof profiles/gprof_example1.txt | dot -Tpng -o profiles/callgraph_example1.png
```

Each image shows a visual representation of function calls and their relative execution costs.

🧹 Cleaning Profiles

To remove all build and profiling files:
```bash
make clean
```
This also deletes the profiles/ directory.


## Sum-up of all instructions
```bash
make clean
make all
make profile
make all-graph
make clean-profile
```

All profiling builds include the -pg flag for gprof.
Each example runs independently, generating its own gmon.out.
The Makefile automatically handles creating and cleaning the profiles/ directory.



