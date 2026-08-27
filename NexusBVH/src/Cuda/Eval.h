#pragma once

#include <cuda_runtime.h>
#include "NXB/BVH.h"

namespace NXB
{
	__global__ void ComputeBVH2CostKernel(BVH2::DeviceView bvh, float* cost);

	__global__ void ComputeBVH8CostKernel(BVH8::DeviceView bvh, float* cost);
}