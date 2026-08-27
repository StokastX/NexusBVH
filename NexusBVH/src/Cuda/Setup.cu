#include "Setup.h"

#include <cub/device/device_radix_sort.cuh>
#include <device_launch_parameters.h>

#include <utility>

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
			node.leftChild = InvalidIdx;
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
	void RadixSort(BVH2BuildState& buildState, DeviceBuffer<McT>& mortonCodes,
		DeviceBuffer<uint32_t>& clusterIdx, const BuildConfig& buildConfig, BVHBuildMetrics* buildMetrics)
	{
		cudaStream_t stream = buildConfig.stream;
		size_t tempStorageBytes = 0;

		DeviceBuffer<McT> mortonCodesSorted(buildState.primCount, stream, buildConfig.pool);
		DeviceBuffer<uint32_t> clusterIdxSorted(buildState.primCount, stream, buildConfig.pool);

		cub::DoubleBuffer<McT> keysBuffer(mortonCodes.Get(), mortonCodesSorted.Get());
		cub::DoubleBuffer<uint32_t> valuesBuffer(clusterIdx.Get(), clusterIdxSorted.Get());

		uint32_t startBit, endBit;
		if constexpr (std::is_same_v<McT, uint32_t>)
			startBit = 2, endBit = 32;
		else
			startBit = 1, endBit = 64;

		// Get the temporary storage size necessary to perform radix sorting
		NXB_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, tempStorageBytes, keysBuffer, valuesBuffer, buildState.primCount, startBit, endBit, stream));

		DeviceBuffer<uint8_t> tempStorage(tempStorageBytes, stream, buildConfig.pool);

		// Perform radix sorting
		{
			StepTimer timer(MetricPtr(buildMetrics, &BVHBuildMetrics::radixSortTime), stream);

			NXB_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(tempStorage.Get(), tempStorageBytes, keysBuffer, valuesBuffer, buildState.primCount, startBit, endBit, stream));
		}

		// cub may have left the sorted data in either half. Where it picked the scratch
		// half, swap the owners so the caller's buffer owns the result.
		if (keysBuffer.Current() != mortonCodes.Get())
			std::swap(mortonCodes, mortonCodesSorted);
		if (valuesBuffer.Current() != clusterIdx.Get())
			std::swap(clusterIdx, clusterIdxSorted);

		buildState.clusterIdx = clusterIdx.Get();
	}

	template __global__ void ComputeSceneBoundsKernel<Triangle>(BVH2BuildState buildState, Triangle* primitives);
	template __global__ void ComputeSceneBoundsKernel<AABB>(BVH2BuildState buildState, AABB* primitives);

	template __global__ void ComputeMortonCodesKernel<uint32_t>(BVH2BuildState buildState, uint32_t* mortonCodes);
	template __global__ void ComputeMortonCodesKernel<uint64_t>(BVH2BuildState buildState, uint64_t* mortonCodes);

	template void RadixSort<uint32_t>(BVH2BuildState& buildState, DeviceBuffer<uint32_t>& mortonCodes,
		DeviceBuffer<uint32_t>& clusterIdx, const BuildConfig& buildConfig, BVHBuildMetrics* buildMetrics);
	template void RadixSort<uint64_t>(BVH2BuildState& buildState, DeviceBuffer<uint64_t>& mortonCodes,
		DeviceBuffer<uint32_t>& clusterIdx, const BuildConfig& buildConfig, BVHBuildMetrics* buildMetrics);
}
