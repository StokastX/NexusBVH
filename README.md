# Nexus BVH

NexusBVH is a fast and high-quality GPU BVH builder written in C++ and CUDA
It implements H-PLOC [\[Benthin et al. 2024\]](https://dl.acm.org/doi/10.1145/3675377) algorithm, focusing on high-performance and high-quality hierarchy generation.

## BVH Construction Benchmark

All times are in milliseconds. Benchmarked on a **Ryzen 9 8945HS, RTX 4070 Laptop (90W, 8GB VRAM).**

| Scene (Triangles)      | Triangle Bounds | Scene Bounds | Morton Codes | Radix Sort (64-bit) | Cluster Init | BVH2  | Total  |
|------------------------|----------------|--------------|--------------|----------------------|-------------|------|--------|
| **Sponza (0.3M)**      | 0.07           | 0.08         | 0.01         | 0.29                 | 0.05        | 0.34 | 0.84   |
| **Buddha (1.1M)**      | 0.26           | 0.30         | 0.10         | 0.67                 | 0.32        | 1.03 | 2.68   |
| **Hairball (2.9M)**    | 0.71           | 1.09         | 0.36         | 3.31                 | 0.78        | 2.21 | 8.46   |
| **Bistro (3.8M)**      | 0.73           | 0.86         | 0.36         | 3.30                 | 0.77        | 2.49 | 8.52   |
| **Powerplant (12.7M)** | 3.23           | 3.04         | 1.87         | 17.80                | 3.42        | 8.93 | 38.25  |


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
