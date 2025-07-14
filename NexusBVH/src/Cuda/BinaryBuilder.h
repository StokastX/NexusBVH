#pragma once
#include <cuda_runtime.h>
#include "Cuda/BuildState.h"

namespace NXB
{
	/*
	 * \brief HPLOC based binary BVH building
	 */
	template <typename McT>
	__global__ void BuildBVH2Kernel(BVH2BuildState buildState, McT* mortonCodes);
}