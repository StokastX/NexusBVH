#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include "NXB/AABB.h"
#include "NXB/Triangle.h"
#include "NXB/BVHBuildMetrics.h"
#include "BuildState.h"

namespace NXB
{
	/* \brief Computes the bounds of both the primitives and the scene
	 * 
	 * \param primitives The list of triangles
	 */
	template <typename PrimT>
	__global__ void ComputeSceneBoundsKernel(BVH2BuildState buildState, PrimT* primitives);

	/*
	 * \brief Compute a list of Morton codes from the centroid of the nodes' AABBs contained in buildState
	 */
	template <typename McT>
	__global__ void ComputeMortonCodesKernel(BVH2BuildState buildState, McT* mortonCodes);

	/*
	 * \brief Performs one sweep radix sort for Morton codes (keys) and cluster indices (values)
	 */
	template <typename McT>
	void RadixSort(BVH2BuildState& buildState, McT*& mortonCodes, BVHBuildMetrics* buildMetrics);

}