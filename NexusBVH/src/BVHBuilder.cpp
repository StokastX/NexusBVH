#include "NXB/BVHBuilder.h"

#include "Math/Math.h"
#include "Cuda/CudaUtils.h"
#include "Cuda/BinaryBuilder.h"
#include "Cuda/WideConverter.h"
#include "Cuda/Setup.h"
#include "Cuda/Eval.h"
#include <chrono>

namespace NXB
{
	template <typename McT = uint64_t>
	BVH2 BuildBVH2Impl(BVH2BuildState buildState, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics)
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
		uint32_t gridSize = CudaUtils::GetGridSizeFullOccupancy((void*)ComputeMortonCodesKernel<McT>, blockSize);

		cudaEvent_t start, stop;

		// Step 2: Compute morton codes
		// ===============================================================================
		if (buildMetrics)
		{
			CUDA_CHECK(cudaEventCreate(&start));
			CUDA_CHECK(cudaEventCreate(&stop));
			CUDA_CHECK(cudaEventRecord(start));

			CUDA_CHECK(cudaLaunchKernel(ComputeMortonCodesKernel<McT>, gridSize, blockSize, args, 0, 0));

			CUDA_CHECK(cudaEventRecord(stop));
			CUDA_CHECK(cudaEventSynchronize(stop));
			CUDA_CHECK(cudaEventElapsedTime(&buildMetrics->computeMortonCodesTime, start, stop));
		}
		else
		{
			CUDA_CHECK(cudaLaunchKernel(ComputeMortonCodesKernel<McT>, gridSize, blockSize, args, 0, 0));
		}
		// ===============================================================================


		// Step 3: Sort morton codes
		// ===============================================================================
		RadixSort<McT>(buildState, mortonCodes, buildMetrics);
		// ===============================================================================

		gridSize = DivideRoundUp(buildState.primCount, blockSize);

		// Step 3: HPLOC binary BVH building
		// ===============================================================================
		if (buildMetrics)
		{
			CUDA_CHECK(cudaEventRecord(start));

			CUDA_CHECK(cudaLaunchKernel(BuildBVH2Kernel<McT>, gridSize, blockSize, args, 0, 0));

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
			CUDA_CHECK(cudaLaunchKernel(BuildBVH2Kernel<McT>, gridSize, blockSize, args, 0, 0));
		}
		// ===============================================================================


		// Create and return the binary BVH
		BVH2 bvh;
		bvh.primCount = buildState.primCount;
		bvh.nodeCount = nodeCount;
		bvh.nodes = buildState.nodes;
		CudaMemory::CopyAsync<AABB>(&bvh.bounds, buildState.sceneBounds, 1, cudaMemcpyDeviceToHost);

		if (buildMetrics)
		{
			float* cost = CudaMemory::AllocAsync<float>(1);
			CudaMemory::MemsetAsync(cost, 0, sizeof(float));
			void* args[2] = { &bvh, &cost };
			uint32_t gridSize = DivideRoundUp(nodeCount, blockSize);

			CUDA_CHECK(cudaLaunchKernel(ComputeBVH2CostKernel, gridSize, blockSize, args, 0, 0));

			CudaMemory::CopyAsync(&buildMetrics->bvh2Cost, cost, 1, cudaMemcpyDeviceToHost);
			CudaMemory::FreeAsync(cost);
		}

		CudaMemory::FreeAsync(buildState.parentIdx);
		CudaMemory::FreeAsync(buildState.clusterIdx);
		CudaMemory::FreeAsync(buildState.clusterCount);
		CudaMemory::FreeAsync(mortonCodes);

		return bvh;
	}


	template<typename PrimT>
	BVH2 BuildBVH2(PrimT* primitives, uint32_t primCount, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics)
	{
		BVH2 bvh;
		uint32_t nodeCount = primCount * 2 - 1;
		BVH2BuildState buildState;
		buildState.primCount = primCount;
		buildState.sceneBounds = CudaMemory::AllocAsync<AABB>(1);
		buildState.nodes = CudaMemory::AllocAsync<BVH2::Node>(nodeCount);

		// Clear scene bounds
		AABB sceneBounds;
		sceneBounds.Clear();
		CudaMemory::CopyAsync(buildState.sceneBounds, &sceneBounds, 1, cudaMemcpyHostToDevice);

		uint32_t blockSize = 64;
		uint32_t gridSize = CudaUtils::GetGridSizeFullOccupancy((void*)ComputeSceneBoundsKernel<PrimT>, blockSize);

		void* args[2] = { &buildState, &primitives };

		// Step 1: Compute scene bounding box
		// ===============================================================================
		if (buildMetrics)
		{
			cudaEvent_t start, stop;
			CUDA_CHECK(cudaEventCreate(&start));
			CUDA_CHECK(cudaEventCreate(&stop));
			CUDA_CHECK(cudaEventRecord(start));

			CUDA_CHECK(cudaLaunchKernel(ComputeSceneBoundsKernel<PrimT>, gridSize, blockSize, args, 0, 0));

			CUDA_CHECK(cudaEventRecord(stop));
			CUDA_CHECK(cudaEventSynchronize(stop));
			CUDA_CHECK(cudaEventElapsedTime(&buildMetrics->computeSceneBoundsTime, start, stop));
			CUDA_CHECK(cudaEventDestroy(start));
			CUDA_CHECK(cudaEventDestroy(stop));
		}
		else
		{
			CUDA_CHECK(cudaLaunchKernel(ComputeSceneBoundsKernel<PrimT>, gridSize, blockSize, args, 0, 0));
		}
		// ===============================================================================


		// Step 2 - 4: Build BVH
		// ==============================================================================
		if (buildConfig.prioritizeSpeed)
			bvh = BuildBVH2Impl<uint32_t>(buildState, buildConfig, buildMetrics);
		else
			bvh = BuildBVH2Impl<uint64_t>(buildState, buildConfig, buildMetrics);
		// ==============================================================================

		CudaMemory::FreeAsync(buildState.sceneBounds);

		CUDA_CHECK(cudaDeviceSynchronize());

		return bvh;
	}

	template <typename PrimT>
	BVH8 BuildBVH8(PrimT* primitives, uint32_t primCount, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics)
	{
		BVH2 bvh2 = BuildBVH2<PrimT>(primitives, primCount, buildConfig, buildMetrics);

		BVH8BuildState buildState;
		buildState.bvh2Nodes = bvh2.nodes;
		buildState.primCount = bvh2.primCount;

		// Worst case senario for a BVH8 built with H-PLOC collapsing: node count = (4n - 1) / 7.
		// This occurs when each internal node in the level above the leaves contains only two leaf nodes
		buildState.bvh8Nodes = CudaMemory::AllocAsync<BVH8::Node>(DivideRoundUp(4 * buildState.primCount - 1, 7));
		buildState.primIdx = CudaMemory::AllocAsync<uint32_t>(buildState.primCount);
		buildState.nodeCounter = CudaMemory::AllocAsync<uint32_t>(1);
		buildState.leafCounter = CudaMemory::AllocAsync<uint32_t>(1);
		buildState.workCounter = CudaMemory::AllocAsync<uint32_t>(1);
		buildState.workAllocCounter = CudaMemory::AllocAsync<uint32_t>(1);
		buildState.indexPairs = CudaMemory::AllocAsync<uint64_t>(buildState.primCount);

		// Init index pairs
		CudaMemory::MemsetAsync(buildState.indexPairs, INVALID_IDX, sizeof(uint64_t) * buildState.primCount);
		// Set first index pair to root of bvh2 and root of bvh8
		uint64_t firstPair = ((uint64_t)bvh2.nodeCount - 1) << 32;
		CudaMemory::CopyAsync(buildState.indexPairs, &firstPair, 1, cudaMemcpyHostToDevice);
		uint32_t nodeCount = 1;
		CudaMemory::CopyAsync(buildState.nodeCounter, &nodeCount, 1, cudaMemcpyHostToDevice);
		CudaMemory::CopyAsync(buildState.workAllocCounter, &nodeCount, 1, cudaMemcpyHostToDevice);
		
		CudaMemory::MemsetAsync(buildState.workCounter, 0, sizeof(uint32_t));
		CudaMemory::MemsetAsync(buildState.leafCounter, 0, sizeof(uint32_t));

		uint32_t blockSize = 256;
		uint32_t gridSize = DivideRoundUp(buildState.primCount, blockSize);
		void* args[1] = { &buildState };

		if (buildMetrics)
		{
			cudaEvent_t start, stop;
			CUDA_CHECK(cudaEventCreate(&start));
			CUDA_CHECK(cudaEventCreate(&stop));
			CUDA_CHECK(cudaEventRecord(start));

			CUDA_CHECK(cudaLaunchKernel(BuildBVH8Kernel, gridSize, blockSize, args, 0, 0));

			CUDA_CHECK(cudaEventRecord(stop));
			CUDA_CHECK(cudaEventSynchronize(stop));
			CUDA_CHECK(cudaEventElapsedTime(&buildMetrics->bvh8ConversionTime, start, stop));
			CUDA_CHECK(cudaEventDestroy(start));
			CUDA_CHECK(cudaEventDestroy(stop));

			buildMetrics->totalTime += buildMetrics->bvh8ConversionTime;
		}
		else
		{
			CUDA_CHECK(cudaLaunchKernel(BuildBVH8Kernel, gridSize, blockSize, args, 0, 0));
		}

		BVH8 bvh8;
		bvh8.nodes = buildState.bvh8Nodes;
		bvh8.primIdx = buildState.primIdx;
		bvh8.bounds = bvh2.bounds;
		bvh8.primCount = buildState.primCount;
		CudaMemory::CopyAsync<uint32_t>(&bvh8.nodeCount, buildState.nodeCounter, 1, cudaMemcpyDeviceToHost);

		if (buildMetrics)
		{
			float* cost = CudaMemory::AllocAsync<float>(1);
			CudaMemory::MemsetAsync(cost, 0, sizeof(float));
			// Make sure we get the node count
			CUDA_CHECK(cudaDeviceSynchronize());

			void* args[2] = { &bvh8, &cost };
			uint32_t gridSize = DivideRoundUp(bvh8.nodeCount, blockSize);

			CUDA_CHECK(cudaLaunchKernel(ComputeBVH8CostKernel, gridSize, blockSize, args, 0, 0));

			CudaMemory::CopyAsync(&buildMetrics->bvh8Cost, cost, 1, cudaMemcpyDeviceToHost);
			CudaMemory::FreeAsync(cost);

			// Warning: this formula is only valid if a leaf node contains exactly one primitive
			// Should be (totalNodes - 1) / internalNodes
			buildMetrics->averageChildPerNode = (float)(bvh8.primCount + bvh8.nodeCount - 1) / bvh8.nodeCount;
		}

		CudaMemory::FreeAsync(bvh2.nodes);
		CudaMemory::FreeAsync(buildState.nodeCounter);
		CudaMemory::FreeAsync(buildState.leafCounter);
		CudaMemory::FreeAsync(buildState.workCounter);
		CudaMemory::FreeAsync(buildState.indexPairs);

		CUDA_CHECK(cudaDeviceSynchronize());

		return bvh8;
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

	void FreeDeviceBVH(BVH8 deviceBvh)
	{
		CudaMemory::Free(deviceBvh.nodes);
		CudaMemory::Free(deviceBvh.primIdx);
	}

	template BVH2 BuildBVH2<Triangle>(Triangle* primitives, uint32_t primCount, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics);
	template BVH2 BuildBVH2<AABB>(AABB* primitives, uint32_t primCount, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics);

	template BVH8 BuildBVH8<Triangle>(Triangle* primitives, uint32_t primCount, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics);
	template BVH8 BuildBVH8<AABB>(AABB* primitives, uint32_t primCount, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics);
}