#pragma once
#include <cuda_runtime.h>
#include "Cuda/BuildState.h"

namespace NXB
{
	/*
	 * \brief HPLOC based binary BVH building
	 */
	template <typename McT>
	__global__ void BuildBinaryBVH(BuildState buildState, McT* mortonCodes);
}