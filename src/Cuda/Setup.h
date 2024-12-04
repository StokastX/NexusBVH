#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include "Math/AABB.h"
#include "Math/Triangle.h"
#include "BuildState.h"

namespace NXB
{
	/* \brief Computes the AABBs of the triangles
	 * 
	 * \param primitives The list of triangles the AABBs will be computed from
	 */
	__global__ void ComputePrimBounds(BuildState buildState, Triangle* primitives);

	/* \brief Computes the bounds of the scene
	 * 
	 * \param primitives The list scene primitives
	 */
	__global__ void ComputeSceneBounds(BuildState buildState);

	/*
	 * \brief Compute a list of 64-bit Morton codes from the centroid of the AABBs contained in buildState
	 */
	__global__ void ComputeMortonCodes(BuildState buildState);

	/*
	 * \brief Performs one sweep radix sort for 64-bit Morton codes
	 */
	void RadixSort(BuildState buildState);
}