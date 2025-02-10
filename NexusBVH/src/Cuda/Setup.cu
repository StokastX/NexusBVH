#include "Setup.h"

#include <cub/device/device_radix_sort.cuh>
#include <device_launch_parameters.h>

#include "CudaUtils.h"
#include "BuilderUtils.h"


namespace NXB
{
	template <typename PrimT>
	__global__ void ComputeSceneBounds(BuildState buildState, PrimT* primitives)
	{
		uint32_t primIdx = blockDim.x * blockIdx.x + threadIdx.x;

		BVH2::Node node;
		AABB bounds;
		bounds.Clear();

		for (uint32_t i = primIdx; i < buildState.primCount; i += blockDim.x * gridDim.x)
		{
			node.bounds = GetBounds(primitives[i]);
			node.leftChild = INVALID_IDX;
			node.rightChild = i;
			buildState.nodes[i] = node;

			bounds.Grow(node.bounds);
		}

		// Perform block-level grow
		bounds = BlockReduceGrow(bounds);

		// Scene bounds update
		if (threadIdx.x == 0)
			AtomicGrow(buildState.sceneBounds, bounds);
	}


	template <typename McT>
	__global__ void ComputeMortonCodes(BuildState buildState, McT* mortonCodes)
	{
		uint32_t primIdx = blockDim.x * blockIdx.x + threadIdx.x;

		if (primIdx >= buildState.primCount)
			return;

		AABB primBounds = buildState.nodes[primIdx].bounds;
		AABB* sceneBounds = buildState.sceneBounds;
		float3 centroid = primBounds.Centroid();

		McT mortonCode = MortonCode<McT>((centroid - sceneBounds->bMin) / (sceneBounds->bMax - sceneBounds->bMin));
		mortonCodes[primIdx] = mortonCode;

		// Initialize cluster indices as well
		buildState.clusterIdx[primIdx] = primIdx;
	}

	void RadixSort(BuildState& buildState, uint32_t*& mortonCodes, BVHBuildMetrics* buildMetrics)
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
			CUDA_CHECK(cudaEventCreate(&start);
			CUDA_CHECK(cudaEventCreate(&stop);
			CUDA_CHECK(cudaDeviceSynchronize()));
			CUDA_CHECK(cudaEventRecord(start)));
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

	void RadixSort(BuildState& buildState, uint64_t*& mortonCodes, BVHBuildMetrics* buildMetrics)
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
			CUDA_CHECK(cudaEventCreate(&start);
			CUDA_CHECK(cudaEventCreate(&stop);
			CUDA_CHECK(cudaDeviceSynchronize()));
			CUDA_CHECK(cudaEventRecord(start)));
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

	template __global__ void ComputeSceneBounds<Triangle>(BuildState buildState, Triangle* primitives);
	template __global__ void ComputeSceneBounds<AABB>(BuildState buildState, AABB* primitives);

	template __global__ void ComputeMortonCodes<uint32_t>(BuildState buildState, uint32_t* mortonCodes);
	template __global__ void ComputeMortonCodes<uint64_t>(BuildState buildState, uint64_t* mortonCodes);
}
