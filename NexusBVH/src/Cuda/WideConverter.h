#pragma once
#include <cuda_runtime.h>
#include "Cuda/BuildState.h"

namespace NXB
{
	__global__ void BuildWideBVH(BVH8BuildState buildState);
}