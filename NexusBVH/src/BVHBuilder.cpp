#include "BVHBuilder.h"

#include "Math/Math.h"
#include "Cuda/CudaUtils.h"
#include "Cuda/BinaryBuilder.h"
#include "Cuda/Setup.h"

namespace NXB
{
	BVH2* BVHBuilder::BuildBinary(AABB* primitives, uint32_t primCount)
	{
		BuildState buildState;
		buildState.primCount = primCount;
		buildState.primBounds = primitives;
		buildState.sceneBounds = CudaMemory::AllocAsync<AABB>(1);
		buildState.mortonCodes = CudaMemory::AllocAsync<uint64_t>(primCount);
		buildState.nodes = CudaMemory::AllocAsync<BVH2::Node>(primCount * 2);
		buildState.primIdx = CudaMemory::AllocAsync<uint32_t>(primCount);
		buildState.clusterIdx = CudaMemory::AllocAsync<uint32_t>(primCount);
		buildState.parentIdx = CudaMemory::AllocAsync<int32_t>(primCount);
		buildState.clusterIdx = CudaMemory::AllocAsync<uint32_t>(primCount);
		buildState.clusterCount = CudaMemory::AllocAsync<uint32_t>(1);

		// Init parent ids to -1
		CudaMemory::MemsetAsync(buildState.parentIdx, -1, sizeof(int32_t) * primCount);

		CudaMemory::MemsetAsync(buildState.clusterCount, 0, sizeof(uint32_t));

		const uint32_t blockSize = 64;
		const uint32_t gridSize = DivideRoundUp(primCount, blockSize);

		void* args[1] = { &buildState };

		// Step 1: Compute scene bounding box
		// ===============================================================================
		CUDA_CHECK(cudaLaunchKernel(ComputeSceneBounds, gridSize, blockSize, args, 0, 0));
		// ===============================================================================

		// Step 2: compute morton codes
		// ===============================================================================
		CUDA_CHECK(cudaLaunchKernel(ComputeMortonCodes, gridSize, blockSize, args, 0, 0));
		// ===============================================================================

		// Step 3: one sweep radix sort for morton codes and primitive ids
		// ===============================================================================
		RadixSort(buildState);
		// ===============================================================================

		// Step 4: HPLOC binary BVH building
		// ===============================================================================
		CUDA_CHECK(cudaLaunchKernel(BuildBinaryBVH, gridSize, blockSize, args, 0, 0));
		// ===============================================================================

		CudaMemory::FreeAsync(buildState.sceneBounds);
		CudaMemory::FreeAsync(buildState.mortonCodes);
		CudaMemory::FreeAsync(buildState.clusterIdx);
		CudaMemory::FreeAsync(buildState.parentIdx);

		// Create and return the binary BVH
		BVH2 hostBvh;
		hostBvh.primCount = primCount;
		hostBvh.nodeCount = primCount * 2;
		hostBvh.nodes = buildState.nodes;
		hostBvh.primIdx = buildState.primIdx;

		// Copy to device
		BVH2* deviceBvh = CudaMemory::AllocAsync<BVH2>(1);
		CudaMemory::Copy<BVH2>(deviceBvh, &hostBvh, 1, cudaMemcpyHostToDevice);

		CUDA_CHECK(cudaDeviceSynchronize());

		return deviceBvh;
	}

	BVH2* BVHBuilder::BuildBinary(Triangle* primitives, uint32_t primCount)
	{
		BuildState buildState;
		buildState.primCount = primCount;
		buildState.primBounds = CudaMemory::AllocAsync<AABB>(primCount);

		const uint32_t blockSize = 1024;
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