#include "Setup.h"

#include <cub/device/device_radix_sort.cuh>
#include <device_launch_parameters.h>

#include "CudaUtils.h"
#include "BuilderUtils.h"


namespace NXB
{
	template <typename PrimT>
	__global__ void ComputeSceneBoundsKernel(BVH2BuildState buildState, PrimT* primitives)
	{
		uint32_t primIdx = blockDim.x * blockIdx.x + threadIdx.x;
		uint32_t laneId = threadIdx.x & (WARP_SIZE - 1);
		uint32_t threadCount = blockDim.x * gridDim.x;

		BVH2::Node node;
		AABB bounds;
		bounds.Clear();

		for (uint32_t i = primIdx; i < buildState.primCount; i += threadCount)
		{
			node.bounds = GetBounds(primitives[i]);
			node.leftChild = INVALID_IDX;
			node.rightChild = i;
			buildState.nodes[i] = node;

			bounds.Grow(node.bounds);
		}

		// Perform warp-level grow
		bounds = WarpReduceGrow(FULL_MASK, bounds);

		// Scene bounds update
		if (laneId == 0)
			AtomicGrow(buildState.sceneBounds, bounds);
	}


	template <typename McT>
	__global__ void ComputeMortonCodesKernel(BVH2BuildState buildState, McT* mortonCodes)
	{
		uint32_t primIdx = blockDim.x * blockIdx.x + threadIdx.x;
		uint32_t threadCount = blockDim.x * gridDim.x;

		for (uint32_t i = primIdx; i < buildState.primCount; i += threadCount)
		{
			AABB primBounds = buildState.nodes[i].bounds;
			AABB* sceneBounds = buildState.sceneBounds;
			float3 centroid = primBounds.Centroid();

			mortonCodes[i] = MortonCode<McT>((centroid - sceneBounds->bMin) / (sceneBounds->bMax - sceneBounds->bMin));

			// Initialize cluster indices as well
			buildState.clusterIdx[i] = i;
		}
	}

	void RadixSort(BVH2BuildState& buildState, uint32_t*& mortonCodes, BVHBuildMetrics* buildMetrics)
	{
		using byte = unsigned char;

		size_t tempStorageBytes = 0;
		void* tempStorage = nullptr;

		uint32_t* mortonCodesSorted = CudaMemory::AllocAsync<uint32_t>(buildState.primCount);
		uint32_t* clusteridxSorted = CudaMemory::AllocAsync<uint32_t>(buildState.primCount);

		cub::DoubleBuffer<uint32_t> keysBuffer(mortonCodes, mortonCodesSorted);
		cub::DoubleBuffer<uint32_t> valuesBuffer(buildState.clusterIdx, clusteridxSorted);

		// Get the temporary storage size necessary to perform radix sorting
		cub::DeviceRadixSort::SortPairs(tempStorage, tempStorageBytes, keysBuffer, valuesBuffer, buildState.primCount, 2, 32);

		tempStorage = CudaMemory::AllocAsync<byte>(tempStorageBytes);

		cudaEvent_t start, stop;
		if (buildMetrics)
		{
			CUDA_CHECK(cudaEventCreate(&start));
			CUDA_CHECK(cudaEventCreate(&stop));
			CUDA_CHECK(cudaEventRecord(start));
		}

		// Perform radix sorting
		cub::DeviceRadixSort::SortPairs(tempStorage, tempStorageBytes, keysBuffer, valuesBuffer, buildState.primCount, 2, 32);

		if (buildMetrics)
		{
			CUDA_CHECK(cudaEventRecord(stop));
			CUDA_CHECK(cudaEventSynchronize(stop));
			CUDA_CHECK(cudaEventElapsedTime(&buildMetrics->radixSortTime, start, stop));
			CUDA_CHECK(cudaEventDestroy(start));
			CUDA_CHECK(cudaEventDestroy(stop));
		}

		mortonCodes = keysBuffer.Current();
		buildState.clusterIdx = valuesBuffer.Current();

		CudaMemory::FreeAsync(tempStorage);

		CudaMemory::FreeAsync(keysBuffer.Alternate());
		CudaMemory::FreeAsync(valuesBuffer.Alternate());

	}

	void RadixSort(BVH2BuildState& buildState, uint64_t*& mortonCodes, BVHBuildMetrics* buildMetrics)
	{
		using byte = unsigned char;

		size_t tempStorageBytes = 0;
		void* tempStorage = nullptr;

		uint64_t* mortonCodesSorted = CudaMemory::AllocAsync<uint64_t>(buildState.primCount);
		uint32_t* clusteridxSorted = CudaMemory::AllocAsync<uint32_t>(buildState.primCount);

		cub::DoubleBuffer<uint64_t> keysBuffer(mortonCodes, mortonCodesSorted);
		cub::DoubleBuffer<uint32_t> valuesBuffer(buildState.clusterIdx, clusteridxSorted);

		// Get the temporary storage size necessary to perform radix sorting
		cub::DeviceRadixSort::SortPairs(tempStorage, tempStorageBytes, keysBuffer, valuesBuffer, buildState.primCount, 1, 64);

		tempStorage = CudaMemory::AllocAsync<byte>(tempStorageBytes);

		cudaEvent_t start, stop;
		if (buildMetrics)
		{
			CUDA_CHECK(cudaEventCreate(&start));
			CUDA_CHECK(cudaEventCreate(&stop));
			CUDA_CHECK(cudaEventRecord(start));
		}

		// Perform radix sorting
		cub::DeviceRadixSort::SortPairs(tempStorage, tempStorageBytes, keysBuffer, valuesBuffer, buildState.primCount, 1, 64);

		if (buildMetrics)
		{
			CUDA_CHECK(cudaEventRecord(stop));
			CUDA_CHECK(cudaEventSynchronize(stop));
			CUDA_CHECK(cudaEventElapsedTime(&buildMetrics->radixSortTime, start, stop));
			CUDA_CHECK(cudaEventDestroy(start));
			CUDA_CHECK(cudaEventDestroy(stop));
		}

		mortonCodes = keysBuffer.Current();
		buildState.clusterIdx = valuesBuffer.Current();

		CudaMemory::FreeAsync(tempStorage);

		CudaMemory::FreeAsync(keysBuffer.Alternate());
		CudaMemory::FreeAsync(valuesBuffer.Alternate());
	}

	template __global__ void ComputeSceneBoundsKernel<Triangle>(BVH2BuildState buildState, Triangle* primitives);
	template __global__ void ComputeSceneBoundsKernel<AABB>(BVH2BuildState buildState, AABB* primitives);

	template __global__ void ComputeMortonCodesKernel<uint32_t>(BVH2BuildState buildState, uint32_t* mortonCodes);
	template __global__ void ComputeMortonCodesKernel<uint64_t>(BVH2BuildState buildState, uint64_t* mortonCodes);
}
