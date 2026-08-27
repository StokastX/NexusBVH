#include "Setup.h"

#include <cub/device/device_radix_sort.cuh>
#include <device_launch_parameters.h>

#include "CudaUtils.h"
#include "Launch.h"
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


	template <typename McT>
	void RadixSort(BVH2BuildState& buildState, McT*& mortonCodes, cudaStream_t stream, BVHBuildMetrics* buildMetrics)
	{
		size_t tempStorageBytes = 0;
		void* tempStorage = nullptr;

		McT* mortonCodesSorted = CudaMemory::AllocAsync<McT>(buildState.primCount, stream);
		uint32_t* clusteridxSorted = CudaMemory::AllocAsync<uint32_t>(buildState.primCount, stream);

		cub::DoubleBuffer<McT> keysBuffer(mortonCodes, mortonCodesSorted);
		cub::DoubleBuffer<uint32_t> valuesBuffer(buildState.clusterIdx, clusteridxSorted);

		uint32_t startBit, endBit;
		if constexpr (std::is_same_v<McT, uint32_t>)
			startBit = 2, endBit = 32;
		else
			startBit = 1, endBit = 64;

		// Get the temporary storage size necessary to perform radix sorting
		cub::DeviceRadixSort::SortPairs(tempStorage, tempStorageBytes, keysBuffer, valuesBuffer, buildState.primCount, startBit, endBit, stream);

		tempStorage = CudaMemory::AllocAsync<uint8_t>(tempStorageBytes, stream);

		// Perform radix sorting
		{
			StepTimer timer(MetricPtr(buildMetrics, &BVHBuildMetrics::radixSortTime), stream);

			cub::DeviceRadixSort::SortPairs(tempStorage, tempStorageBytes, keysBuffer, valuesBuffer, buildState.primCount, startBit, endBit, stream);
		}

		mortonCodes = keysBuffer.Current();
		buildState.clusterIdx = valuesBuffer.Current();

		CudaMemory::FreeAsync(tempStorage, stream);

		CudaMemory::FreeAsync(keysBuffer.Alternate(), stream);
		CudaMemory::FreeAsync(valuesBuffer.Alternate(), stream);
	}

	template __global__ void ComputeSceneBoundsKernel<Triangle>(BVH2BuildState buildState, Triangle* primitives);
	template __global__ void ComputeSceneBoundsKernel<AABB>(BVH2BuildState buildState, AABB* primitives);

	template __global__ void ComputeMortonCodesKernel<uint32_t>(BVH2BuildState buildState, uint32_t* mortonCodes);
	template __global__ void ComputeMortonCodesKernel<uint64_t>(BVH2BuildState buildState, uint64_t* mortonCodes);

	template void RadixSort<uint32_t>(BVH2BuildState& buildState, uint32_t*& mortonCodes, cudaStream_t stream, BVHBuildMetrics* buildMetrics);
	template void RadixSort<uint64_t>(BVH2BuildState& buildState, uint64_t*& mortonCodes, cudaStream_t stream, BVHBuildMetrics* buildMetrics);
}
