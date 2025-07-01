#pragma once

#include <iostream>
#include "NXB/AABB.h"
#include "NXB/BVH.h"

namespace NXB
{
	struct BVH2BuildState
	{
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

	struct BVH8BuildState
	{
		// BVH2 nodes
		BVH2::Node* bvh2Nodes;

		// BVH8 nodes
		BVH8::Node* bvh8Nodes;
		uint32_t* primIdx;
		uint32_t primCount;

		// Number of BVH8 nodes
		uint32_t* nodeCount;

		// Index pairs
		uint64_t* indexPairs;

		// Atomic counter to dispatch work items
		uint32_t* workCounter;
	};
}
