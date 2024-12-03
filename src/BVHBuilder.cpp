#include "BVHBuilder.h"
#include "Cuda/BinaryBuilder.h"

namespace NXB
{
	BVH2* BVHBuilder::BuildBinary(float* primitives, uint32_t primCount, unsigned char primType)
	{
		cudaLaunchKernel(&BuildBinaryKernel, dim3(1, 1, 1), dim3(1, 1, 1), nullptr, 0, 0);
		cudaDeviceSynchronize();
		return nullptr;

	}

	BVH8* BVHBuilder::ConvertToWideBVH(BVH2* binaryBVH)
	{
		return nullptr;
	}
}