#include "NXB/BVHBuilder.h"

#include "Math/Math.h"
#include "Cuda/CudaUtils.h"
#include "Cuda/BinaryBuilder.h"
#include "Cuda/Setup.h"
#include "Cuda/Eval.h"
#include <chrono>

namespace NXB
{
	template <typename McT = uint64_t>
	BVH2 BuildBinaryImpl(BuildState buildState, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics)
	{
		uint32_t nodeCount = buildState.primCount * 2 - 1;
		buildState.parentIdx = CudaMemory::AllocAsync<int32_t>(buildState.primCount);
		buildState.clusterIdx = CudaMemory::AllocAsync<uint32_t>(buildState.primCount);
		buildState.clusterCount = CudaMemory::AllocAsync<uint32_t>(1);
		McT* mortonCodes = CudaMemory::AllocAsync<McT>(buildState.primCount);

		// Init parent ids to -1
		CudaMemory::MemsetAsync(buildState.parentIdx, INVALID_IDX, sizeof(int32_t) * buildState.primCount);
		CudaMemory::CopyAsync(buildState.clusterCount, &buildState.primCount, 1, cudaMemcpyHostToDevice);

		void* args[2] = { &buildState, &mortonCodes };

		uint32_t blockSize = 64;
		uint32_t gridSize = CudaUtils::GetGridSizeFullOccupancy((void*)ComputeMortonCodes<McT>, blockSize);

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

		gridSize = DivideRoundUp(buildState.primCount, blockSize);

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
		BVH2 bvh;
		bvh.primCount = buildState.primCount;
		bvh.nodeCount = nodeCount;
		bvh.nodes = buildState.nodes;
		CudaMemory::CopyAsync<AABB>(&bvh.bounds, buildState.sceneBounds, 1, cudaMemcpyDeviceToHost);

		if (buildMetrics)
		{
			float* cost = CudaMemory::AllocAsync<float>(1);
			CudaMemory::MemsetAsync(cost, 0, 1);
			void* args[2] = { &bvh, &cost };
			uint32_t gridSize = DivideRoundUp(nodeCount, blockSize);

			CUDA_CHECK(cudaLaunchKernel(ComputeBVHCost, gridSize, blockSize, args, 0, 0));

			CudaMemory::CopyAsync(&buildMetrics->bvhCost, cost, 1, cudaMemcpyDeviceToHost);
		}

		CudaMemory::FreeAsync(buildState.parentIdx);
		CudaMemory::FreeAsync(buildState.clusterIdx);
		CudaMemory::FreeAsync(buildState.clusterCount);
		CudaMemory::FreeAsync(mortonCodes);

		return bvh;
	}


	template<typename PrimT>
	BVH2 BuildBinary(PrimT* primitives, uint32_t primCount, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics)
	{
		BVH2 bvh;
		uint32_t nodeCount = primCount * 2 - 1;
		BuildState buildState;
		buildState.primCount = primCount;
		buildState.sceneBounds = CudaMemory::AllocAsync<AABB>(1);
		buildState.nodes = CudaMemory::AllocAsync<BVH2::Node>(nodeCount);

		// Clear scene bounds
		AABB sceneBounds;
		sceneBounds.Clear();
		CudaMemory::CopyAsync(buildState.sceneBounds, &sceneBounds, 1, cudaMemcpyHostToDevice);

		uint32_t blockSize = 64;
		uint32_t gridSize = CudaUtils::GetGridSizeFullOccupancy((void*)ComputeSceneBounds<PrimT>, blockSize);

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
			bvh = BuildBinaryImpl<uint32_t>(buildState, buildConfig, buildMetrics);
		else
			bvh = BuildBinaryImpl<uint64_t>(buildState, buildConfig, buildMetrics);
		// ==============================================================================

		CudaMemory::FreeAsync(buildState.sceneBounds);

		CUDA_CHECK(cudaDeviceSynchronize());

		return bvh;
	}

	BVH8 ConvertToWideBVH(BVH2* binaryBVH, BVHBuildMetrics* buildMetrics)
	{
		return BVH8();
	}

	BVH2 ToHost(BVH2 deviceBvh)
	{
		BVH2 hostBVH;
		hostBVH.primCount = deviceBvh.primCount;
		hostBVH.nodeCount = deviceBvh.nodeCount;
		hostBVH.bounds = deviceBvh.bounds;
		hostBVH.nodes = new BVH2::Node[deviceBvh.nodeCount];
		CudaMemory::Copy(hostBVH.nodes, deviceBvh.nodes, deviceBvh.nodeCount, cudaMemcpyDeviceToHost);
		return hostBVH;
	}

	void FreeHostBVH(BVH2 hostBvh)
	{
		delete[] hostBvh.nodes;
	}

	void FreeDeviceBVH(BVH2 deviceBvh)
	{
		CudaMemory::Free(deviceBvh.nodes);
	}

	void FreeBVH(BVH8* wideBVH)
	{

	}

	template BVH2 BuildBinary<Triangle>(Triangle* primitives, uint32_t primCount, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics);
	template BVH2 BuildBinary<AABB>(AABB* primitives, uint32_t primCount, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics);
}