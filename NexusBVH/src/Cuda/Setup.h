#pragma once

#include <cuda_runtime.h>
#include "NXB/AABB.h"
#include "NXB/DeviceBuffer.h"
#include "NXB/Triangle.h"
#include "NXB/BVHBuildMetrics.h"
#include "NXB/BuildConfig.h"
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
	 *
	 * cub sorts through a double buffer and may leave the result in either half. Both
	 * buffers are taken by reference and swapped with the scratch halves when it does, so
	 * on return each one owns the sorted data. buildState.clusterIdx is repointed to match.
	 */
	template <typename McT>
	void RadixSort(BVH2BuildState& buildState, DeviceBuffer<McT>& mortonCodes,
		DeviceBuffer<uint32_t>& clusterIdx, const BuildConfig& buildConfig, BVHBuildMetrics* buildMetrics);

}