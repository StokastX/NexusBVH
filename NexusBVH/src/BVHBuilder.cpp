#include "NXB/BVHBuilder.h"

#include "Math/Math.h"
#include "Cuda/CudaUtils.h"
#include "Cuda/BinaryBuilder.h"
#include "Cuda/Setup.h"

namespace NXB
{
	BVH2* BVHBuilder::BuildBinary(AABB* primitives, uint32_t primCount, BVHBuildMetrics* buildMetrics)
	{
		uint32_t nodeCount = primCount * 2 - 1;
		BuildState buildState;
		buildState.primCount = primCount;
		buildState.primBounds = primitives;
		buildState.sceneBounds = CudaMemory::AllocAsync<AABB>(1);
		buildState.mortonCodes = CudaMemory::AllocAsync<uint64_t>(primCount);
		buildState.nodes = CudaMemory::AllocAsync<BVH2::Node>(nodeCount);
		buildState.primIdx = CudaMemory::AllocAsync<uint32_t>(primCount);
		buildState.parentIdx = CudaMemory::AllocAsync<int32_t>(primCount);
		buildState.clusterIdx = CudaMemory::AllocAsync<uint32_t>(primCount);
		buildState.clusterCount = CudaMemory::AllocAsync<uint32_t>(1);

		// Init parent ids to -1
		CudaMemory::MemsetAsync(buildState.parentIdx, -1, sizeof(int32_t) * primCount);

		CudaMemory::CopyAsync(buildState.clusterCount, &primCount, 1, cudaMemcpyHostToDevice);

		const uint32_t blockSize = 64;
		const uint32_t gridSize = DivideRoundUp(primCount, blockSize);

		void* args[1] = { &buildState };

		cudaEvent_t start, stop;
		CUDA_CHECK(cudaEventCreate(&start));
		CUDA_CHECK(cudaEventCreate(&stop));

		// Step 1: Compute scene bounding box
		// ===============================================================================
		if (buildMetrics)
		{
			CUDA_CHECK(cudaEventRecord(start));
			CUDA_CHECK(cudaLaunchKernel(ComputeSceneBounds, gridSize, blockSize, args, 0, 0));
			CUDA_CHECK(cudaEventRecord(stop));
			CUDA_CHECK(cudaEventSynchronize(stop));
			CUDA_CHECK(cudaEventElapsedTime(&buildMetrics->computeSceneBoundsTime, start, stop));
		}
		else
		{
			CUDA_CHECK(cudaLaunchKernel(ComputeSceneBounds, gridSize, blockSize, args, 0, 0));
		}
		// ===============================================================================


		// Step 2: compute morton codes
		// ===============================================================================
		if (buildMetrics)
		{
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


		// Step 3: one sweep radix sort for morton codes (keys) and primitive ids (values)
		// ===============================================================================
		RadixSort(buildState, buildMetrics);
		// ===============================================================================


		// Step 4: Initialize clusters
		// ===============================================================================
		if (buildMetrics)
		{
			CUDA_CHECK(cudaEventRecord(start));
			CUDA_CHECK(cudaLaunchKernel(InitClusters, gridSize, blockSize, args, 0, 0));
			CUDA_CHECK(cudaEventRecord(stop));
			CUDA_CHECK(cudaEventSynchronize(stop));
			CUDA_CHECK(cudaEventElapsedTime(&buildMetrics->initClustersTime, start, stop));
		}
		else
		{
			CUDA_CHECK(cudaLaunchKernel(InitClusters, gridSize, blockSize, args, 0, 0));
		}
		// ===============================================================================


		// Step 5: HPLOC binary BVH building
		// ===============================================================================
		if (buildMetrics)
		{
			CUDA_CHECK(cudaEventRecord(start));
			CUDA_CHECK(cudaLaunchKernel(BuildBinaryBVH, gridSize, blockSize, args, 0, 0));
			CUDA_CHECK(cudaEventRecord(stop));
			CUDA_CHECK(cudaEventSynchronize(stop));
			CUDA_CHECK(cudaEventElapsedTime(&buildMetrics->bvhBuildTime, start, stop));
			buildMetrics->totalTime = buildMetrics->computeTriangleBoundsTime + buildMetrics->computeSceneBoundsTime
				+ buildMetrics->computeMortonCodesTime + buildMetrics->radixSortTime + buildMetrics->initClustersTime + buildMetrics->bvhBuildTime;
		}
		else
		{
			CUDA_CHECK(cudaLaunchKernel(BuildBinaryBVH, gridSize, blockSize, args, 0, 0));
		}
		// ===============================================================================

		CUDA_CHECK(cudaEventDestroy(start));
		CUDA_CHECK(cudaEventDestroy(stop));

		// Create and return the binary BVH
		BVH2 hostBvh;
		hostBvh.primCount = primCount;
		hostBvh.nodeCount = nodeCount;
		hostBvh.nodes = buildState.nodes;
		hostBvh.primIdx = buildState.primIdx;
		CudaMemory::CopyAsync<AABB>(&hostBvh.bounds, buildState.sceneBounds, 1, cudaMemcpyDeviceToHost);

		// Copy to device
		BVH2* deviceBvh = CudaMemory::AllocAsync<BVH2>(1);
		CudaMemory::CopyAsync<BVH2>(deviceBvh, &hostBvh, 1, cudaMemcpyHostToDevice);

		CudaMemory::FreeAsync(buildState.sceneBounds);
		CudaMemory::FreeAsync(buildState.mortonCodes);
		CudaMemory::FreeAsync(buildState.clusterIdx);
		CudaMemory::FreeAsync(buildState.parentIdx);
		CudaMemory::FreeAsync(buildState.clusterCount);

		CUDA_CHECK(cudaDeviceSynchronize());

		return deviceBvh;
	}

	BVH2* BVHBuilder::BuildBinary(Triangle* primitives, uint32_t primCount, BVHBuildMetrics* buildMetrics)
	{
		BuildState buildState;
		buildState.primCount = primCount;
		buildState.primBounds = CudaMemory::AllocAsync<AABB>(primCount);

		const uint32_t blockSize = 64;
		const uint32_t gridSize = DivideRoundUp(primCount, blockSize);

		void* args[2] = { &buildState, &primitives };

		cudaEvent_t start, stop;
		CUDA_CHECK(cudaEventCreate(&start));
		CUDA_CHECK(cudaEventCreate(&stop));

		// Step 0: compute triangle bounding boxes
		// ==============================================================================
		if (buildMetrics)
		{
			CUDA_CHECK(cudaEventRecord(start));
			CUDA_CHECK(cudaLaunchKernel(ComputePrimBounds, gridSize, blockSize, args, 0, 0));
			CUDA_CHECK(cudaEventRecord(stop));
			CUDA_CHECK(cudaEventSynchronize(stop));
			CUDA_CHECK(cudaEventElapsedTime(&buildMetrics->computeTriangleBoundsTime, start, stop));
		}
		else
		{
			CUDA_CHECK(cudaLaunchKernel(ComputePrimBounds, gridSize, blockSize, args, 0, 0));
		}
		// ==============================================================================
		

		// Step 1 - 5: build BVH
		// ==============================================================================
		BVH2* bvh = BuildBinary(buildState.primBounds, primCount, buildMetrics);
		// ==============================================================================


		CUDA_CHECK(cudaEventDestroy(start));
		CUDA_CHECK(cudaEventDestroy(stop));

		CudaMemory::FreeAsync(buildState.primBounds);

		CUDA_CHECK(cudaDeviceSynchronize());

		return bvh;
	}

	BVH8* BVHBuilder::ConvertToWideBVH(BVH2* binaryBVH, BVHBuildMetrics* buildMetrics)
	{
		return nullptr;
	}

	void FreeBVH(BVH2* bvh2)
	{
		BVH2 hostBvh;
		CudaMemory::Copy<BVH2>(&hostBvh, bvh2, 1, cudaMemcpyDeviceToHost);
		CudaMemory::Free(hostBvh.nodes);
		CudaMemory::Free(hostBvh.primIdx);
		CudaMemory::Free(bvh2);
	}

	void FreeBVH(BVH8* wideBVH)
	{

	}
}