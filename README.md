# Nexus BVH
![teaser9](https://github.com/user-attachments/assets/a56284f9-bfe7-49d1-b83a-6374537d7e9b)

This project was developed as part of the GPU Computing course at Ensimag. The full report can be found [here](https://patrick-attimont.com/assets/documents/NXB_report.pdf).

NexusBVH is a fast and high-quality GPU BVH builder written in C++ and CUDA.
It implements H-PLOC [\[Benthin et al. 2024\]](https://dl.acm.org/doi/10.1145/3675377) algorithm, focusing on high-performance and high-quality hierarchy generation.

## BVH Construction Benchmark

All times are in milliseconds and represent kernel execution times measured on the CPU side. Benchmarked on a **Ryzen 7 5700X, RTX 3070 8 Go.** 

BVH2 refers to the H-PLOC kernel with a search radius of 8. Radix sort is performed using 32-bit Morton codes. When using 64-bit Morton codes, sorting time is approximately **3x slower**.

| Scene (Triangles)      | Scene Bounds | Morton Codes | Radix Sort           | BVH2  | Total  |
|------------------------|--------------|--------------|----------------------|------|--------|
| **Sponza (0.3M)**      | 0.06         | 0.04         | 0.22                 | 0.37 | 0.68   |
| **Buddha (1.1M)**      | 0.22         | 0.14         | 0.37                 | 1.05 | 1.78   |
| **Hairball (2.9M)**    | 0.55         | 0.31         | 0.89                 | 2.01 | 3.86   |
| **Bistro (3.8M)**      | 0.59         | 0.31         | 1.02                 | 2.66 | 4.58   |
| **Powerplant (12.7M)** | 2.52         | 1.31         | 3.59                 | 8.85 | 16.27  |
| **Lucy (28.1M)**       | 5.78         | 3.07         | 7.98                 | 22.2 | 39.03  |

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
