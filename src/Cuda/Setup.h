#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include "Math/AABB.h"
#include "BuilderUtils.h"

namespace NXB
{
	/* \brief Computes the AABBs of the triangles
	 * 
	 * \param primitives The list of triangles the AABBs will be computed from
	 */
	__global__ void ComputePrimBounds(BuildState buildState, float3* primitives);

	/* \brief Computes the bounds of the scene
	 * 
	 * \param primitives The list scene primitives
	 */
	__global__ void ComputeSceneBounds(BuildState buildState, float3* primitives);

	/*
	 * \brief Compute a list of 64-bit Morton codes from the centroid of the AABBs contained in buildState
	 */
	__global__ void ComputeMortonCodes(BuildState buildState);

	/* \brief Performs one sweep radix sort for 64-bit Morton codes
	 *
	 * \param mortonCodes The list of Morton codes to be sorted
	 * \param primIds The list of primitive indices
	 * \param size The number of Morton codes
	 */
	void RadixSort(uint64_t* mortonCodes, uint32_t* primIds, uint32_t size);
}