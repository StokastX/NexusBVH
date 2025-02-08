#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include "NXB/AABB.h"
#include "NXB/Triangle.h"
#include "NXB/BVHBuildMetrics.h"
#include "BuildState.h"

namespace NXB
{
	/* \brief Computes the bounds of the scene
	 * 
	 * \param primitives The list of bounding boxes
	 */
	__global__ void ComputeSceneBounds(BuildState buildstate, AABB* primitives);

	/* \brief Computes the bounds of both the triangles and the scene
	 * 
	 * \param primitives The list of triangles
	 */
	__global__ void ComputeSceneBounds(BuildState buildState, Triangle* primitives);

	/*
	 * \brief Compute a list of 64-bit Morton codes from the centroid of the AABBs contained in buildState
	 */
	__global__ void ComputeMortonCodes(BuildState buildState);

	/*
	 * \brief Performs one sweep radix sort for 64-bit Morton codes
	 */
	void RadixSort(BuildState& buildState, BVHBuildMetrics* buildMetrics);

	/*
	 * \brief Initialize the data (leaf nodes, clusters) required for HPLOC kernel
	 */
	__global__ void InitClusters(BuildState buildState);
}