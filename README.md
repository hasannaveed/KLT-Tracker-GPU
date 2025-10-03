# KLT Feature Tracker Acceleration on GPUs
**CS 4110 - High Performance Computing with GPUs**  
Complex Computing Problem (CCP)  

This project accelerates the Kanade–Lucas–Tomasi (KLT) Feature Tracker using GPU programming with CUDA and OpenACC.  
The work is divided into multiple versions (V1–V4), each demonstrating incremental improvements in performance and optimization.

---

## 📂 Repository Structure
src/
├── V1/ 
# Baseline sequential CPU implementation
├── V2/ 
# Naive GPU implementation (CUDA)
├── V3/ 
# Optimized GPU implementation (CUDA + memory/launch optimizations)
├── V4/ 
# OpenACC pragma-based optimized implementation
data/ 
# Dataset (images/sequences for testing)
report/ 
# Reports (D1–D4 deliverables)
slides/ 
# Presentation slides
README.md 
# Project overview and usage instructions


---

## 🚀 Project Versions
- **V1 (Baseline CPU)**  
  Sequential KLT feature tracker provided as the starting point. Used for profiling and identifying hotspots.  

- **V2 (Naive GPU)**  
  First CUDA implementation. Direct mapping of computation to GPU threads with minimal optimizations.  

- **V3 (Optimized GPU)**  
  Highly optimized CUDA implementation. Improvements include:  
  - Custom launch configuration  
  - Occupancy tuning  
  - Communication optimizations (minimizing CPU–GPU transfers)  
  - Memory hierarchy optimizations (shared memory, coalesced access)  

- **V4 (OpenACC)**  
  Directive-based GPU acceleration using OpenACC for easier portability and maintainability.  

---

