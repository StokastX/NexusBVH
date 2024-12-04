#include "BVHBuilder.h"
#include "Cuda/BinaryBuilder.h"
#include "Cuda/BuilderUtils.h"

namespace NXB
{
	BVH2* BVHBuilder::BuildBinary(float3* primitives, uint32_t primCount, PrimType primType)
	{
		cudaLaunchKernel(&BuildBinaryKernel, dim3(1, 1, 1), dim3(1, 1, 1), nullptr, 0, 0);
		cudaDeviceSynchronize();
		return nullptr;
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