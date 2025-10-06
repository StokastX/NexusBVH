# Nexus BVH
![teaser9](https://github.com/user-attachments/assets/a56284f9-bfe7-49d1-b83a-6374537d7e9b)

**NexusBVH** is a fast and high-quality GPU BVH builder written in C++ and CUDA.

It implements the H-PLOC algorithm proposed by [Benthin et al. 2024](https://dl.acm.org/doi/10.1145/3675377), a high-performance BVH construction method designed for GPUs. H-PLOC constructs high-quality BVHs through hierarchical clustering of spatially nearby primitives in parallel, making it well-suited for real-time ray tracing applications.


> 📝 *This project was originally developed as part of the GPU Computing course at Ensimag. The full report is available [here](https://patrick-attimont.com/assets/documents/NXB_report.pdf).*


## BVH Construction Benchmark

All times are in milliseconds and represent kernel execution times measured on the CPU side. Benchmarked on a **Ryzen 7 5700X, RTX 3070 8 Go.** 

BVH2 refers to the H-PLOC kernel with a search radius of 8. Radix sort is performed using 32-bit Morton codes. When using 64-bit Morton codes, sorting time is approximately **3x slower**.

| Scene (Triangles)      | Scene Bounds | Morton Codes | Radix Sort | BVH2  | BVH8  | Total |
|------------------------|--------------|--------------|------------|-------|-------|--------|
| **Sponza (0.3M)**      | 0.03         | 0.01         | 0.06       | 0.20  | 0.14  | 0.44   |
| **Buddha (1.1M)**      | 0.11         | 0.02         | 0.10       | 0.47  | 0.42  | 1.12   |
| **Hairball (2.9M)**    | 0.29         | 0.19         | 0.20       | 0.90  | 1.36  | 2.95   |
| **Bistro (3.9M)**      | 0.40         | 0.26         | 0.25       | 1.35  | 1.84  | 4.10   |
| **Powerplant (12.7M)** | 1.34         | 0.84         | 1.33       | 3.67  | 5.90  | 13.09  |
| **Lucy (28.1M)**       | 2.98         | 1.79         | 3.11       | 9.75  | 15.36 | 33.00  |

## Prerequisites
NexusBVH is a CMake-based project and requires the following dependencies:

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (from NVIDIA)
- [CMake](https://cmake.org/download/) version 3.22 or higher

The project has been tested on both **Windows** (with Visual Studio) and **Ubuntu**.

## Build

1. **Clone the repository**:

   ```sh
   git clone https://github.com/StokastX/NexusBVH
   ```

2. **Generate the solution via cmake**:

   ``` sh
   mkdir build
   cd build
   cmake ..
   ```

3. **Build the project**:
- On Linux: Use ```make``` on your preferred build system:

   ``` sh
   make -j
   ```
- On Windows (Visual Studio): Open the generated solution file in Visual Studio, and press F5 to build and run.

## Resources

- H-PLOC: [Benthin et al. 2024](https://dl.acm.org/doi/10.1145/3675377)
- PLOC++: [Benthin et al. 2022](https://dl.acm.org/doi/10.1145/3543867)
- PLOC: [Meister and Bittner 2018](https://ieeexplore.ieee.org/document/7857089)
- Bottom-up LBVH traversal: [Apetrei 2014](https://doi.org/10.2312/cgvc.20141206)
