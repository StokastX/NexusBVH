#include "BVHBuilder.h"

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

		// Launch scene bounds kernel
		// TODO

		buildState.mortonCodes = CudaMemory::AllocAsync<uint64_t>(primCount);
		buildState.primIdx = CudaMemory::AllocAsync<uint32_t>(primCount);

		// Launch morton codes kernel
		// TODO

		// Launch radix sort kernel
		// TODO
		RadixSort(buildState);

		//cudaLaunchKernel(&BuildBinaryKernel, dim3(1, 1, 1), dim3(1, 1, 1), nullptr, 0, 0);
		//cudaDeviceSynchronize();

		CudaMemory::FreeAsync(buildState.sceneBounds);
		CudaMemory::FreeAsync(buildState.mortonCodes);
		CudaMemory::FreeAsync(buildState.primIdx);

		return nullptr;
	}

	BVH2* BVHBuilder::BuildBinary(Triangle* primitives, uint32_t primCount)
	{
		BuildState buildState;
		buildState.primCount = primCount;
		buildState.primBounds = CudaMemory::AllocAsync<AABB>(primCount);

		// Launch primitive bounds computation kernel
		// TODO

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