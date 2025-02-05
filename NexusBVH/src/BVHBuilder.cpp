#include "BVHBuilder.h"

#include "Math/Math.h"
#include "Cuda/CudaUtils.h"
#include "Cuda/BinaryBuilder.h"
#include "Cuda/Setup.h"

namespace NXB
{
	BVH2* BVHBuilder::BuildBinary(AABB* primitives, uint32_t primCount)
	{
		uint32_t nodeCount = primCount * 2 - 1;
		BuildState buildState;
		buildState.primCount = primCount;
		buildState.primBounds = primitives;
		buildState.sceneBounds = CudaMemory::AllocAsync<AABB>(1);
		buildState.mortonCodes = CudaMemory::AllocAsync<uint64_t>(primCount);
		buildState.nodes = CudaMemory::AllocAsync<BVH2::Node>(nodeCount);
		buildState.primIdx = CudaMemory::AllocAsync<uint32_t>(primCount);
		buildState.clusterIdx = CudaMemory::AllocAsync<uint32_t>(primCount);
		buildState.parentIdx = CudaMemory::AllocAsync<int32_t>(primCount);
		buildState.clusterIdx = CudaMemory::AllocAsync<uint32_t>(primCount);
		buildState.clusterCount = CudaMemory::AllocAsync<uint32_t>(1);

		// Init parent ids to -1
		CudaMemory::MemsetAsync(buildState.parentIdx, -1, sizeof(int32_t) * primCount);

		CudaMemory::CopyAsync(buildState.clusterCount, &primCount, 1, cudaMemcpyHostToDevice);

		const uint32_t blockSize = 64;
		const uint32_t gridSize = DivideRoundUp(primCount, blockSize);

		void* args[1] = { &buildState };

		std::cout << std::endl << std::endl << "========== Building binary BVH ==========" << std::endl << std::endl;

		float elapsedTime[5];
		cudaEvent_t start, stop;
		CUDA_CHECK(cudaEventCreate(&start));
		CUDA_CHECK(cudaEventCreate(&stop));
		CUDA_CHECK(cudaEventRecord(start));

		// Step 1: Compute scene bounding box
		// ===============================================================================
		CUDA_CHECK(cudaLaunchKernel(ComputeSceneBounds, gridSize, blockSize, args, 0, 0));
		// ===============================================================================

		CUDA_CHECK(cudaEventRecord(stop));
		CUDA_CHECK(cudaEventSynchronize(stop));
		CUDA_CHECK(cudaEventElapsedTime(&elapsedTime[0], start, stop));
		CUDA_CHECK(cudaEventRecord(start));

		// Step 2: compute morton codes
		// ===============================================================================
		CUDA_CHECK(cudaLaunchKernel(ComputeMortonCodes, gridSize, blockSize, args, 0, 0));
		// ===============================================================================

		CUDA_CHECK(cudaEventRecord(stop));
		CUDA_CHECK(cudaEventSynchronize(stop));
		CUDA_CHECK(cudaEventElapsedTime(&elapsedTime[1], start, stop));

		// Step 3: one sweep radix sort for morton codes and primitive ids
		// ===============================================================================
		elapsedTime[2] = RadixSort(buildState);
		// ===============================================================================

		CUDA_CHECK(cudaEventRecord(start));

		// Step 4: Initialize clusters
		// ===============================================================================
		CUDA_CHECK(cudaLaunchKernel(InitClusters, gridSize, blockSize, args, 0, 0));
		// ===============================================================================

		CUDA_CHECK(cudaEventRecord(stop));
		CUDA_CHECK(cudaEventSynchronize(stop));
		CUDA_CHECK(cudaEventElapsedTime(&elapsedTime[3], start, stop));
		CUDA_CHECK(cudaEventRecord(start));

		// Step 5: HPLOC binary BVH building
		// ===============================================================================
		CUDA_CHECK(cudaLaunchKernel(BuildBinaryBVH, gridSize, blockSize, args, 0, 0));
		// ===============================================================================

		CUDA_CHECK(cudaEventRecord(stop));
		CUDA_CHECK(cudaEventSynchronize(stop));
		CUDA_CHECK(cudaEventElapsedTime(&elapsedTime[4], start, stop));
		CUDA_CHECK(cudaEventDestroy(start));
		CUDA_CHECK(cudaEventDestroy(stop));

		float buildTime = elapsedTime[0] + elapsedTime[1] + elapsedTime[2] + elapsedTime[3] + elapsedTime[4];

		std::cout << "Primitive count: " << primCount << std::endl;
		std::cout << "Node count: " << nodeCount << std::endl << std::endl;

		std::cout << "---------- TIMINGS ----------" << std::endl << std::endl;
		std::cout << "Mesh bounds: " << elapsedTime[0] << " ms" << std::endl;
		std::cout << "Morton codes: " << elapsedTime[1] << " ms" << std::endl;
		std::cout << "Radix sort: " << elapsedTime[2] << " ms" << std::endl;
		std::cout << "Clusters init: " << elapsedTime[3] << " ms" << std::endl;
		std::cout << "Binary BVH building: " << elapsedTime[4] << " ms" << std::endl;
		std::cout << "Total build time: " << buildTime << " ms" << std::endl << std::endl;

		std::cout << "========== Building done ==========" << std::endl << std::endl;

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

	BVH2* BVHBuilder::BuildBinary(Triangle* primitives, uint32_t primCount)
	{
		BuildState buildState;
		buildState.primCount = primCount;
		buildState.primBounds = CudaMemory::AllocAsync<AABB>(primCount);

		const uint32_t blockSize = 64;
		const uint32_t gridSize = DivideRoundUp(primCount, blockSize);

		void* args[2] = { &buildState, &primitives };

		// Step 0: compute triangle bounding boxes
		// ==============================================================================
		CUDA_CHECK(cudaLaunchKernel(ComputePrimBounds, gridSize, blockSize, args, 0, 0));
		// ==============================================================================

		BVH2* bvh = BuildBinary(buildState.primBounds, primCount);

		CudaMemory::FreeAsync(buildState.primBounds);

		CUDA_CHECK(cudaDeviceSynchronize());

		return bvh;
	}

	BVH8* BVHBuilder::ConvertToWideBVH(BVH2* binaryBVH)
	{
		return nullptr;
	}

	void BVHBuilder::FreeBVH(BVH2* bvh2)
	{
		BVH2 hostBvh;
		CudaMemory::Copy<BVH2>(&hostBvh, bvh2, 1, cudaMemcpyDeviceToHost);
		CudaMemory::Free(hostBvh.nodes);
		CudaMemory::Free(hostBvh.primIdx);
		CudaMemory::Free(bvh2);
	}

	void BVHBuilder::FreeBVH(BVH8* wideBVH)
	{

	}
}