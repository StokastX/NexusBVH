# Nexus BVH
![teaser9](https://github.com/user-attachments/assets/a56284f9-bfe7-49d1-b83a-6374537d7e9b)

**NexusBVH** is a fast and high-quality GPU BVH builder written in C++17 and CUDA.

It implements the H-PLOC algorithm proposed by [Benthin et al. 2024](https://dl.acm.org/doi/10.1145/3675377), a high-performance BVH construction method designed for GPUs. H-PLOC constructs high-quality BVHs through hierarchical clustering of spatially nearby primitives in parallel, making it well-suited for real-time ray tracing applications.

On top of the binary BVH, NexusBVH can collapse the hierarchy into a **compressed wide BVH8** in the style of [Ylitie et al. 2017](https://research.nvidia.com/publication/2017-07_efficient-incoherent-ray-traversal-gpus-through-compressed-wide-bvhs) — 8 children per node, quantized down to 80 bytes.

> 📝 *This project was originally developed as part of the GPU Computing course at Ensimag. The full report is available [here](https://patrick-attimont.com/assets/documents/NXB_report.pdf).*


## At a glance

- **BVH2** via H-PLOC, and a **compressed BVH8** collapse on top of it
- Builds from **triangles or AABBs**, on the device from end to end
- 32-bit or 64-bit Morton codes, chosen per build (speed vs. quality)
- The returned BVH **owns its device memory**, so there is no free function to call
- Runs on **your stream**, allocates from **your memory pool**
- Failures are **exceptions**, and a failed build leaks nothing
- Optional **per-step kernel timings** that do not serialize the build
- No dependencies beyond the CUDA Toolkit


## BVH Construction Benchmark

All times are in milliseconds and represent kernel execution times measured on the CPU side. Benchmarked on an **Intel Core i9-14900K, RTX 4080 SUPER.**

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

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 12 or 13
- [CMake](https://cmake.org/download/) 3.24 or higher
- A C++17 host compiler (MSVC 2022 on Windows)

The project has been tested on both **Windows** (with Visual Studio) and **Ubuntu**.


## Build

```sh
git clone https://github.com/StokastX/NexusBVH
cd NexusBVH
cmake -S . -B build
cmake --build build --config Release
```

On Windows this generates a Visual Studio solution in `build/`, which you can open and
build with F5 instead.

The build targets the architecture of the GPU in the machine by default; pass
`-DCMAKE_CUDA_ARCHITECTURES=86` to override it.

| Option | Default | |
|---|---|---|
| `NEXUSBVH_BUILD_TESTS` | `ON` when top level | Build the doctest suite |
| `NEXUSBVH_INSTALL` | `ON` when top level | Generate install and package rules |
| `NEXUSBVH_TEST_PRIM_COUNT` | `200000` | Primitive count used by the larger test scenes |


## Using NexusBVH in your project

NexusBVH builds as a **static library**. Its public surface is `include/NXB/`, and the
target to link is `NXB::NexusBVH` — the same name whichever way you consume it.

**As a subdirectory**, if you vendor the source:

``` cmake
add_subdirectory(external/NexusBVH)
target_link_libraries(myapp PRIVATE NXB::NexusBVH)
```

**As an installed package**, if you would rather build it once:

``` sh
cmake --install build --config Release --prefix /path/to/prefix
```

``` cmake
find_package(NexusBVH REQUIRED)   # -DCMAKE_PREFIX_PATH=/path/to/prefix
target_link_libraries(myapp PRIVATE NXB::NexusBVH)
```

Either way the target carries the include directory, C++17 and the CUDA runtime with it,
and **your project does not need the CUDA language enabled** to use the library. Device
symbols are resolved when the archive is built, so the API is host-side: your own kernels
cannot call NexusBVH `__device__` functions directly.


## Quick start

``` cpp
#include <NXB/BVHBuilder.h>
#include <NXB/DeviceBuffer.h>

std::vector<NXB::Triangle> triangles = LoadMesh();

// Upload. The constructor synchronizes, so the vector may be freed right after.
NXB::DeviceBuffer<NXB::Triangle> devicePrims(triangles);

NXB::BVH8 bvh = NXB::BuildBVH8(devicePrims.Get(), (uint32_t)triangles.size());

// View() is what a kernel receives
TraceKernel<<<grid, block>>>(bvh.View(), rays, hits);

// Nothing to free: bvh releases its device memory when it goes out of scope
```

`NXB::BuildBVH2` has the same signature and returns an `NXB::BVH2`.


## The API

`NexusBVH/include/NXB/` is the entire public surface; `src/` is private.

### Building

``` cpp
template <typename PrimT>   // PrimT is NXB::Triangle or NXB::AABB
NXB::BVH2 NXB::BuildBVH2(PrimT* devicePrims, uint32_t primCount,
                         BuildConfig = {}, BVHBuildMetrics* = nullptr);

template <typename PrimT>
NXB::BVH8 NXB::BuildBVH8(PrimT* devicePrims, uint32_t primCount,
                         BuildConfig = {}, BVHBuildMetrics* = nullptr);
```

`devicePrims` is a **device** pointer. `NXB::DeviceBuffer<T>` (`NXB/DeviceBuffer.h`) is the
RAII way to produce one, but `NXB::Triangle` is just three `float3` vertices and
`NXB::AABB` a min/max pair, so a renderer that already holds either on the device can pass
its own pointer. A `primCount` of 0 returns an empty BVH.

`BuildBVH8` builds a BVH2 internally and releases it before returning.

### What a build returns

`BVH2` and `BVH8` own their device memory and release it in the destructor, on the stream
they were built on and back into the pool they came from. They are move-only.

``` cpp
NXB::BVH2 bvh = NXB::BuildBVH2(devicePrims.Get(), primCount);

bvh.View();                       // BVH2::DeviceView, what a kernel takes
bvh.ToHost();                     // BVH2::Host, std::vector backed
bvh.Release();                    // give up ownership; the caller then owes a cudaFreeAsync
NXB::BVH2::Adopt(view, stream);   // the inverse: take a view over

bvh.NodeCount();  bvh.PrimCount();  bvh.Bounds();  bvh.Empty();
```

`BVH8` is the same over its two arrays (the nodes and `primIdx`), plus
`AverageChildPerNode()`.

A `DeviceView` owns nothing: its pointers belong to the BVH that handed it out. And since
the destructor's free is stream-ordered, **the stream a BVH was built on must outlive the
BVH**.

### Build configuration

``` cpp
struct NXB::BuildConfig
{
    bool          prioritizeSpeed = false;   // 32-bit Morton codes instead of 64-bit
    cudaMemPool_t pool            = nullptr; // where the build allocates from
    cudaStream_t  stream          = nullptr; // where the build runs
};
```

- **`prioritizeSpeed`** selects 32-bit Morton codes (10 bits per axis): the radix sort is
  roughly 3x faster, at the cost of positional accuracy and therefore BVH quality. The
  default is 64-bit (21 bits per axis).
- **`stream`** carries every allocation, copy and kernel of the build, and the build
  synchronizes against it alone. It has drained by the time the build returns.
- **`pool`** is the one that matters if you build more than once — see below.

### Reusing memory across builds

CUDA's default memory pool hands every free byte back to the driver at each
synchronization, so back-to-back builds re-acquire all of their memory every time.
`NXB::MemoryPool` (`NXB/MemoryPool.h`) is an RAII `cudaMemPool_t` that keeps what it holds:

``` cpp
NXB::MemoryPool pool;

NXB::BuildConfig config;
config.pool = pool.Handle();

for (const Mesh& mesh : meshes)
    NXB::BVH8 bvh = NXB::BuildBVH8(mesh.devicePrims, mesh.primCount, config);

pool.TrimTo();   // hand the VRAM back when you are done
```

Measured on a 4070 Laptop, this is worth roughly **2x on rebuilds** (BVH8 at 1M
primitives: 8.0 → 3.5 ms; at 5M: 42.6 → 19.6 ms). Below ~200k primitives it is within
noise. The pool is private to the object, and the BVH produced is identical either way.

### Errors

Every entry point reports a CUDA failure by throwing **`NXB::CudaError`**, which carries
the `cudaError_t` in `.code`. Nothing in the library calls `exit` or writes to a stream.

``` cpp
try
{
    NXB::BVH8 bvh = NXB::BuildBVH8(devicePrims.Get(), primCount);
}
catch (const NXB::CudaError& e)
{
    // A build that throws has already released everything it allocated,
    // so retrying with fewer primitives after an OOM is safe.
    std::cerr << e.what() << "\n";
}
```

### Timings

Pass a `BVHBuildMetrics*` to get per-step kernel times, in milliseconds:

``` cpp
NXB::BVHBuildMetrics metrics;
NXB::BVH8 bvh = NXB::BuildBVH8(devicePrims.Get(), primCount, config, &metrics);

// computeSceneBoundsTime, computeMortonCodesTime, radixSortTime,
// bvhBuildTime, bvh8ConversionTime, totalTime
```

Asking for them does not serialize the build. `totalTime` has its own pair of events rather
than being summed from the steps, so it also covers the allocations and the gaps between
launches.

For repeated measurement:

``` cpp
#include <NXB/BenchmarkReport.h>   // opt-in: the only public header that does stream I/O

auto samples = NXB::BenchmarkBuild(
    [&](NXB::BuildConfig cfg, NXB::BVHBuildMetrics* m) {
        return NXB::BuildBVH8(devicePrims.Get(), primCount, cfg, m);
    },
    /* warmup */ 500, /* measured */ 1000, config);

NXB::PrintReport(std::cout, samples);

// or reduce the samples yourself
NXB::BVHBuildMetrics median = NXB::Median(samples);
```

### SAH cost

``` cpp
#include <NXB/BVHCost.h>

float cost = NXB::ComputeSAHCost(bvh);   // BVH2 or BVH8, with stream and pool overloads
```

A property of a finished tree rather than a build output, so it is a call you make when you
want the number: it launches one kernel over every node and synchronizes. `C_T = 3`,
`C_I = 2`.


## Testing

The suite is doctest-based and built by default:

``` sh
ctest -C Release --test-dir build           # everything, ~2 s
ctest -C Release --test-dir build -L fast   # tiny scenes only, ~0.6 s
```

It walks the hierarchies the builder produces and checks their invariants: indices in
range, parent boxes containing their children, every primitive referenced exactly once,
scene bounds matching a host-computed reference, and for the BVH8 the quantization contract
on top of that.


## Resources

- H-PLOC: [Benthin et al. 2024](https://dl.acm.org/doi/10.1145/3675377)
- PLOC++: [Benthin et al. 2022](https://dl.acm.org/doi/10.1145/3543867)
- PLOC: [Meister and Bittner 2018](https://ieeexplore.ieee.org/document/7857089)
- Compressed wide BVHs: [Ylitie et al. 2017](https://research.nvidia.com/publication/2017-07_efficient-incoherent-ray-traversal-gpus-through-compressed-wide-bvhs)
- Bottom-up LBVH traversal: [Apetrei 2014](https://doi.org/10.2312/cgvc.20141206)
