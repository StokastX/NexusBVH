#pragma once

#include <cuda_runtime.h>
#include "NXB/BVH.h"

namespace NXB
{
	__global__ void ComputeBVHCost(BVH2 bvh, float* cost);
}