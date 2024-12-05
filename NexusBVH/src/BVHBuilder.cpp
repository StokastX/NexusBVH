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
		buildState.primIdx = CudaMemory::AllocAsync<uint32_t>(primCount);

		const uint32_t blockSize = 1024;
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
		// TODO
		// ===============================================================================

		CudaMemory::FreeAsync(buildState.sceneBounds);
		CudaMemory::FreeAsync(buildState.mortonCodes);
		CudaMemory::FreeAsync(buildState.primIdx);

		CUDA_CHECK(cudaDeviceSynchronize());

		return nullptr;
	}

	BVH2* BVHBuilder::BuildBinary(Triangle* primitives, uint32_t primCount)
	{
		BuildState buildState;
		buildState.primCount = primCount;
		buildState.primBounds = CudaMemory::AllocAsync<AABB>(primCount);

		const uint32_t blockSize = 1024;
		const uint32_t gridSize = DivideRoundUp(primCount, blockSize);

		void* args[2] = { &buildState, &primitives };

		// Step 0: compute triangles bounding boxes
		// ==============================================================================
		CUDA_CHECK(cudaLaunchKernel(ComputePrimBounds, gridSize, blockSize, args, 0, 0));
		// ==============================================================================

		BVH2* bvh = BuildBinary(buildState.primBounds, primCount);

		CudaMemory::FreeAsync(buildState.primBounds);

		return bvh;
	}

	BVH8* BVHBuilder::ConvertToWideBVH(BVH2* binaryBVH)
	{
		return nullptr;
	}

	void BVHBuilder::FreeBVH(BVH2* bindaryBVH)
	{

	}

	void BVHBuilder::FreeBVH(BVH8* wideBVH)
	{

	}
}