# Nexus BVH

Fast and high quality GPU BVH builder written in C++ and CUDA.
It implements H-PLOC [[Benthin et al. 2024](https://dl.acm.org/doi/10.1145/3675377)] algorithm.

## Prerequisites

- Microsoft Visual Studio 2022
- Nvidia [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [CMake](https://cmake.org/download/) 3.22 or higher

## Build
- Clone the repository
   ```sh
   git clone https://github.com/StokastX/NexusBVH
   ```
- Launch the setup.bat script. It will generate a Visual Studio solution in the build folder

  Alternatively, you can generate the solution via cmake:
  ```sh
  mkdir build
  cd build
  cmake ..
  ```
- Open the Visual Studio solution and build the project

## Resources

- H-PLOC: [[Benthin et al. 2024](https://dl.acm.org/doi/10.1145/3675377)]
- PLOC++: [[Benthin et al. 2022](https://dl.acm.org/doi/10.1145/3543867)]
- PLOC: [[Meister and Bittner 2018](https://ieeexplore.ieee.org/document/7857089)]
- Bottom-up LBVH traversal: [[Apetrei 2014]([https://ieeexplore.ieee.org/document/7857089](https://doi.org/10.2312/cgvc.20141206))]
