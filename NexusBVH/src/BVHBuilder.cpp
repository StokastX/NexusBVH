#include "NXB/BVHBuilder.h"

#include "Math/Math.h"
#include "Cuda/CudaUtils.h"
#include "Cuda/BinaryBuilder.h"
#include "Cuda/Setup.h"

namespace NXB
{
	BVH2* BuildBinary(BuildState buildState, BVHBuildMetrics* buildMetrics)
	{
		buildState.mortonCodes = CudaMemory::AllocAsync<uint64_t>(buildState.primCount);
		buildState.parentIdx = CudaMemory::AllocAsync<int32_t>(buildState.primCount);
		buildState.clusterIdx = CudaMemory::AllocAsync<uint32_t>(buildState.primCount);
		buildState.clusterCount = CudaMemory::AllocAsync<uint32_t>(1);

		// Init parent ids to -1
		CudaMemory::MemsetAsync(buildState.parentIdx, INVALID_IDX, sizeof(int32_t) * buildState.primCount);

		CudaMemory::CopyAsync(buildState.clusterCount, &buildState.primCount, 1, cudaMemcpyHostToDevice);

		const uint32_t blockSize = 64;
		const uint32_t gridSize = DivideRoundUp(buildState.primCount, blockSize);

		void* args[1] = { &buildState };

		cudaEvent_t start, stop;

		// Step 2: Compute morton codes
		// ===============================================================================
		if (buildMetrics)
		{
			CUDA_CHECK(cudaEventCreate(&start));
			CUDA_CHECK(cudaEventCreate(&stop));
			CUDA_CHECK(cudaEventRecord(start));

			CUDA_CHECK(cudaLaunchKernel(ComputeMortonCodes, gridSize, blockSize, args, 0, 0));

			CUDA_CHECK(cudaEventRecord(stop));
			CUDA_CHECK(cudaEventSynchronize(stop));
			CUDA_CHECK(cudaEventElapsedTime(&buildMetrics->computeMortonCodesTime, start, stop));
		}
		else
		{
			CUDA_CHECK(cudaLaunchKernel(ComputeMortonCodes, gridSize, blockSize, args, 0, 0));
		}
		// ===============================================================================


		// Step 3: One sweep radix sort for morton codes (keys) and primitive ids (values)
		// ===============================================================================
		RadixSort(buildState, buildMetrics);
		// ===============================================================================


		// Step 4: HPLOC binary BVH building
		// ===============================================================================
		if (buildMetrics)
		{
			CUDA_CHECK(cudaEventRecord(start));

			CUDA_CHECK(cudaLaunchKernel(BuildBinaryBVH, gridSize, blockSize, args, 0, 0));

			CUDA_CHECK(cudaEventRecord(stop));
			CUDA_CHECK(cudaEventSynchronize(stop));
			CUDA_CHECK(cudaEventElapsedTime(&buildMetrics->bvhBuildTime, start, stop));
			CUDA_CHECK(cudaEventDestroy(start));
			CUDA_CHECK(cudaEventDestroy(stop));

			buildMetrics->totalTime = buildMetrics->computeSceneBoundsTime + buildMetrics->computeMortonCodesTime
				+ buildMetrics->radixSortTime + buildMetrics->bvhBuildTime;
		}
		else
		{
			CUDA_CHECK(cudaLaunchKernel(BuildBinaryBVH, gridSize, blockSize, args, 0, 0));
		}
		// ===============================================================================

		// Create and return the binary BVH
		BVH2 hostBvh;
		hostBvh.primCount = buildState.primCount;
		hostBvh.nodeCount = buildState.primCount * 2 - 1;
		hostBvh.nodes = buildState.nodes;
		CudaMemory::CopyAsync<AABB>(&hostBvh.bounds, buildState.sceneBounds, 1, cudaMemcpyDeviceToHost);

		// Copy to device
		BVH2* deviceBvh = CudaMemory::AllocAsync<BVH2>(1);
		CudaMemory::CopyAsync<BVH2>(deviceBvh, &hostBvh, 1, cudaMemcpyHostToDevice);

		CudaMemory::FreeAsync(buildState.mortonCodes);
		CudaMemory::FreeAsync(buildState.parentIdx);
		CudaMemory::FreeAsync(buildState.clusterIdx);
		CudaMemory::FreeAsync(buildState.clusterCount);

		return deviceBvh;
	}


	template<typename PrimT>
	BVH2* BuildBinaryImpl(PrimT* primitives, uint32_t primCount, BVHBuildMetrics* buildMetrics)
	{
		uint32_t nodeCount = primCount * 2 - 1;
		BuildState buildState;
		buildState.primCount = primCount;
		buildState.sceneBounds = CudaMemory::AllocAsync<AABB>(1);
		buildState.nodes = CudaMemory::AllocAsync<BVH2::Node>(nodeCount);

		const uint32_t blockSize = 64;
		const uint32_t gridSize = DivideRoundUp(primCount, blockSize);

		void* args[2] = { &buildState, &primitives };

		auto sceneBoundsKernel = static_cast<void (*)(BuildState, PrimT*)>(&ComputeSceneBounds);

		// Step 1: Compute scene bounding box
		// ===============================================================================
		if (buildMetrics)
		{
			cudaEvent_t start, stop;
			CUDA_CHECK(cudaEventCreate(&start));
			CUDA_CHECK(cudaEventCreate(&stop));
			CUDA_CHECK(cudaEventRecord(start));

			CUDA_CHECK(cudaLaunchKernel(sceneBoundsKernel, gridSize, blockSize, args, 0, 0));

			CUDA_CHECK(cudaEventRecord(stop));
			CUDA_CHECK(cudaEventSynchronize(stop));
			CUDA_CHECK(cudaEventElapsedTime(&buildMetrics->computeSceneBoundsTime, start, stop));
			CUDA_CHECK(cudaEventDestroy(start));
			CUDA_CHECK(cudaEventDestroy(stop));
		}
		else
		{
			CUDA_CHECK(cudaLaunchKernel(sceneBoundsKernel, gridSize, blockSize, args, 0, 0));
		}
		// ===============================================================================


		// Step 2 - 4: Build BVH
		// ==============================================================================
		BVH2* bvh = BuildBinary(buildState, buildMetrics);
		// ==============================================================================

		CudaMemory::FreeAsync(buildState.sceneBounds);

		CUDA_CHECK(cudaDeviceSynchronize());

		return bvh;
	}

	BVH2* BuildBinary(AABB* primitives, uint32_t primCount, BVHBuildMetrics* buildMetrics)
	{
		return BuildBinaryImpl<AABB>(primitives, primCount, buildMetrics);
	}

	BVH2* BuildBinary(Triangle* primitives, uint32_t primCount, BVHBuildMetrics* buildMetrics)
	{
		return BuildBinaryImpl<Triangle>(primitives, primCount, buildMetrics);
	}

	BVH8* ConvertToWideBVH(BVH2* binaryBVH, BVHBuildMetrics* buildMetrics)
	{
		return nullptr;
	}

	void FreeBVH(BVH2* bvh2)
	{
		BVH2 hostBvh;
		CudaMemory::Copy<BVH2>(&hostBvh, bvh2, 1, cudaMemcpyDeviceToHost);
		CudaMemory::Free(hostBvh.nodes);
		CudaMemory::Free(bvh2);
	}

	void FreeBVH(BVH8* wideBVH)
	{

	}
}