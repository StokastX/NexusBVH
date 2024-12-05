#pragma once

#include <iostream>
#include "Math/AABB.h"
#include "BVH/BVH.h"

namespace NXB
{
	struct BuildState
	{
		// List of Morton codes
		uint64_t* mortonCodes;

		// Bounds of the primitives (or primitives if primType == AABB)
		AABB* primBounds;

		// Scene bounds
		AABB* sceneBounds;

		// BVH2 nodes
		BVH2::Node* nodes;

		// Primitive indices
		uint32_t* primIdx;

		// Cluster indices
		uint32_t* clusterIdx;

		// BVH2 parent indices
		int32_t* parentIdx;

		// Number of primitives
		uint32_t primCount;
	};
}
