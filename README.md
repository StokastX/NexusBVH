# Nexus BVH

NexusBVH is a fast and high-quality GPU BVH builder written in C++ and CUDA.
It implements H-PLOC [\[Benthin et al. 2024\]](https://dl.acm.org/doi/10.1145/3675377) algorithm, focusing on high-performance and high-quality hierarchy generation.

## BVH Construction Benchmark

All times are in milliseconds and represent kernel execution times measured on the CPU side. Benchmarked on a **Ryzen 9 8945HS, RTX 4070 Laptop (90W, 8GB VRAM).** 

BVH2 refers to the H-PLOC kernel with a search radius of 8. Radix sort is performed using 32-bit Morton codes. When using 64-bit Morton codes, sorting time is approximately **4× slower**.

| Scene (Triangles)      | Scene Bounds | Morton Codes | Radix Sort (32-bit) | BVH2  | Total  |
|------------------------|--------------|--------------|----------------------|------|--------|
| **Sponza (0.3M)**      | 0.11         | 0.01         | 0.15                 | 0.37 | 0.64   |
| **Buddha (1.1M)**      | 0.37         | 0.20         | 0.28                 | 1.02 | 1.88   |
| **Hairball (2.9M)**    | 1.21         | 0.49         | 0.83                 | 2.03 | 4.56   |
| **Bistro (3.8M)**      | 0.99         | 0.48         | 0.81                 | 2.33 | 4.61   |
| **Powerplant (12.7M)** | 3.64         | 2.04         | 5.34                 | 8.59 | 19.60  |


## Prerequisites

- Microsoft Visual Studio 2022
- Nvidia [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [CMake](https://cmake.org/download/) 3.22 or higher

## Build
- Clone the repository
   ```sh
   git clone https://github.com/StokastX/NexusBVH
   ```
- Run setup.bat to automatically generate a Visual Studio solution in the build/ directory.

  Alternatively, you can generate the solution via cmake:
  ```sh
  mkdir build
  cd build
  cmake ..
  ```
- Open the Visual Studio solution and build the project

## Resources

- H-PLOC: [\[Benthin et al. 2024\]](https://dl.acm.org/doi/10.1145/3675377)
- PLOC++: [\[Benthin et al. 2022\]](https://dl.acm.org/doi/10.1145/3543867)
- PLOC: [\[Meister and Bittner 2018\]](https://ieeexplore.ieee.org/document/7857089)
- Bottom-up LBVH traversal: [\[Apetrei 2014\]](https://doi.org/10.2312/cgvc.20141206)
