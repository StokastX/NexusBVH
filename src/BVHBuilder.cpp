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
		ComputeSceneBounds(buildState);

		buildState.mortonCodes = CudaMemory::AllocAsync<uint64_t>(primCount);
		buildState.primIdx = CudaMemory::AllocAsync<uint32_t>(primCount);
		ComputeMortonCodes(buildState);

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
		ComputePrimBounds(buildState, primitives);

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
	void Test()
	{
		NXB::Triangle t0(
			make_float3(40.0f, -40.0f, 40.0f),
			make_float3(40.0f, -40.0f, -40.0f),
			make_float3(-40.0f, -40.0f, -40.0f)
		);

		NXB::Triangle t1(
			make_float3(-40.0f, -40.0f, 40.0f),
			make_float3(40.0f, -40.0f, 40.0f),
			make_float3(-40.0f, -40.0f, -40.0f)
		);

		NXB::Triangle t2(
			make_float3(40.0f, 40.0f, 40.0f),
			make_float3(40.0f, 40.0f, -40.0f),
			make_float3(-40.0f, 40.0f, -40.0f)
		);

		NXB::Triangle t3(
			make_float3(-40.0f, 40.0f, 40.0f),
			make_float3(40.0f, 40.0f, 40.0f),
			make_float3(-40.0f, 40.0f, -40.0f)
		);

		NXB::Triangle triangles[4] = { t0, t1, t2, t3 };

		NXB::BVHBuilder bvhBuilder;

		NXB::Triangle* dTriangles = NXB::CudaMemory::Allocate<NXB::Triangle>(4);
		NXB::CudaMemory::Copy<NXB::Triangle>(dTriangles, triangles, 4, cudaMemcpyHostToDevice);

		std::cout << "Building BVH" << std::endl;
		bvhBuilder.BuildBinary(dTriangles, 4);

		std::cout << "Building done" << std::endl;

		//NXB::CudaMemory::Free(dTriangles);
	}
}