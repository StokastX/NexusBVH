#pragma once

#include <iostream>
#include "NXB/AABB.h"
#include "NXB/BVH.h"

namespace NXB
{
	struct BuildState
	{
		// List of Morton codes
		uint64_t* mortonCodes;

		// Scene bounds
		AABB* sceneBounds;

		// BVH2 nodes
		BVH2::Node* nodes;

		// Cluster indices
		uint32_t* clusterIdx;

		// BVH2 parent indices
		int32_t* parentIdx;

		// Number of primitives
		uint32_t primCount;

		// Number of merged clusters
		uint32_t* clusterCount;
	};
}
