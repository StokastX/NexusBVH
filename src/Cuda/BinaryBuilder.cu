#include "BinaryBuilder.cuh"
#include <iostream>
#include <cub/device/device_radix_sort.cuh>
#include "Math/AABB.h"

namespace NXB
{
	__global__ void BuildBinaryKernel()
	{
		printf("Building binary BVH");
	}

	__global__ void ComputeMortonCodes()
	{

	}
}