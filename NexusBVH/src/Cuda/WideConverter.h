#pragma once
#include <cuda_runtime.h>
#include "Cuda/BuildState.h"

namespace NXB
{
	__global__ void BuildBVH8(BVH8BuildState buildState);
}