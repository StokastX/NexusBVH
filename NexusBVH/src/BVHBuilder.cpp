#include "NXB/BVHBuilder.h"

#include "Math/Math.h"
#include "Cuda/CudaUtils.h"
#include "Cuda/BinaryBuilder.h"
#include "Cuda/Setup.h"
#include <chrono>

namespace NXB
{
	template <typename McT = uint64_t>
	BVH2* BuildBinary(BuildState buildState, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics)
	{
		buildState.parentIdx = CudaMemory::AllocAsync<int32_t>(buildState.primCount);
		buildState.clusterIdx = CudaMemory::AllocAsync<uint32_t>(buildState.primCount);
		buildState.clusterCount = CudaMemory::AllocAsync<uint32_t>(1);
		McT* mortonCodes = CudaMemory::AllocAsync<McT>(buildState.primCount);

		// Init parent ids to -1
		CudaMemory::MemsetAsync(buildState.parentIdx, INVALID_IDX, sizeof(int32_t) * buildState.primCount);
		CudaMemory::CopyAsync(buildState.clusterCount, &buildState.primCount, 1, cudaMemcpyHostToDevice);

		void* args[2] = { &buildState, &mortonCodes };

		const uint32_t blockSize = 64;
		const uint32_t gridSize = DivideRoundUp(buildState.primCount, blockSize);

		cudaEvent_t start, stop;
		CUDA_CHECK(cudaEventCreate(&start));
		CUDA_CHECK(cudaEventCreate(&stop));

		// Step 2: Compute morton codes
		// ===============================================================================
		if (buildMetrics)
		{
			CUDA_CHECK(cudaEventRecord(start));

			CUDA_CHECK(cudaLaunchKernel(ComputeMortonCodes<McT>, gridSize, blockSize, args, 0, 0));

			CUDA_CHECK(cudaEventRecord(stop));
			CUDA_CHECK(cudaEventSynchronize(stop));
			CUDA_CHECK(cudaEventElapsedTime(&buildMetrics->computeMortonCodesTime, start, stop));
		}
		else
		{
			CUDA_CHECK(cudaLaunchKernel(ComputeMortonCodes<McT>, gridSize, blockSize, args, 0, 0));
		}
		// ===============================================================================


		// Step 3: Sort morton codes
		// ===============================================================================
		RadixSort(buildState, mortonCodes, buildMetrics);
		// ===============================================================================


		// Step 3: HPLOC binary BVH building
		// ===============================================================================
		if (buildMetrics)
		{
			CUDA_CHECK(cudaDeviceSynchronize());
			CUDA_CHECK(cudaEventRecord(start));

			CUDA_CHECK(cudaLaunchKernel(BuildBinaryBVH<McT>, gridSize, blockSize, args, 0, 0));

			CUDA_CHECK(cudaEventRecord(stop));
			CUDA_CHECK(cudaEventSynchronize(stop));
			CUDA_CHECK(cudaEventElapsedTime(&buildMetrics->bvhBuildTime, start, stop));

			buildMetrics->totalTime = buildMetrics->computeSceneBoundsTime + buildMetrics->computeMortonCodesTime
				+ buildMetrics->radixSortTime + buildMetrics->bvhBuildTime;
		}
		else
		{
			CUDA_CHECK(cudaLaunchKernel(BuildBinaryBVH<McT>, gridSize, blockSize, args, 0, 0));
		}
		// ===============================================================================

		CUDA_CHECK(cudaEventDestroy(start));
		CUDA_CHECK(cudaEventDestroy(stop));

		// Create and return the binary BVH
		BVH2 hostBvh;
		hostBvh.primCount = buildState.primCount;
		hostBvh.nodeCount = buildState.primCount * 2 - 1;
		hostBvh.nodes = buildState.nodes;
		CudaMemory::CopyAsync<AABB>(&hostBvh.bounds, buildState.sceneBounds, 1, cudaMemcpyDeviceToHost);

		// Copy to device
		BVH2* deviceBvh = CudaMemory::AllocAsync<BVH2>(1);
		CudaMemory::CopyAsync<BVH2>(deviceBvh, &hostBvh, 1, cudaMemcpyHostToDevice);

		CudaMemory::FreeAsync(buildState.parentIdx);
		CudaMemory::FreeAsync(buildState.clusterIdx);
		CudaMemory::FreeAsync(buildState.clusterCount);
		CudaMemory::FreeAsync(mortonCodes);

		return deviceBvh;
	}


	template<typename PrimT>
	BVH2* BuildBinaryImpl(PrimT* primitives, uint32_t primCount, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics)
	{
		BVH2* bvh;
		uint32_t nodeCount = primCount * 2 - 1;
		BuildState buildState;
		buildState.primCount = primCount;
		buildState.sceneBounds = CudaMemory::AllocAsync<AABB>(1);
		buildState.nodes = CudaMemory::AllocAsync<BVH2::Node>(nodeCount);

		const uint32_t blockSize = 32;
		const uint32_t gridSize = DivideRoundUp(primCount, blockSize);

		void* args[2] = { &buildState, &primitives };

		// Step 1: Compute scene bounding box
		// ===============================================================================
		if (buildMetrics)
		{
			cudaEvent_t start, stop;
			CUDA_CHECK(cudaEventCreate(&start));
			CUDA_CHECK(cudaEventCreate(&stop));
			CUDA_CHECK(cudaEventRecord(start));

			CUDA_CHECK(cudaLaunchKernel(ComputeSceneBounds<PrimT>, gridSize, blockSize, args, 0, 0));

			CUDA_CHECK(cudaEventRecord(stop));
			CUDA_CHECK(cudaEventSynchronize(stop));
			CUDA_CHECK(cudaEventElapsedTime(&buildMetrics->computeSceneBoundsTime, start, stop));
			CUDA_CHECK(cudaEventDestroy(start));
			CUDA_CHECK(cudaEventDestroy(stop));
		}
		else
		{
			CUDA_CHECK(cudaLaunchKernel(ComputeSceneBounds<PrimT>, gridSize, blockSize, args, 0, 0));
		}
		// ===============================================================================


		// Step 2 - 4: Build BVH
		// ==============================================================================
		if (buildConfig.prioritizeSpeed)
			bvh = BuildBinary<uint32_t>(buildState, buildConfig, buildMetrics);
		else
			bvh = BuildBinary<uint64_t>(buildState, buildConfig, buildMetrics);
		// ==============================================================================

		CudaMemory::FreeAsync(buildState.sceneBounds);

		CUDA_CHECK(cudaDeviceSynchronize());

		return bvh;
	}

	BVH2* BuildBinary(AABB* primitives, uint32_t primCount, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics)
	{
		return BuildBinaryImpl<AABB>(primitives, primCount, buildConfig, buildMetrics);
	}

	BVH2* BuildBinary(Triangle* primitives, uint32_t primCount, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics)
	{
		return BuildBinaryImpl<Triangle>(primitives, primCount, buildConfig, buildMetrics);
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