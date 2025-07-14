#pragma once

#include <cuda_runtime.h>
#include "NXB/BVH.h"

namespace NXB
{
	__global__ void ComputeBVHCostKernel(BVH2 bvh, float* cost);
}