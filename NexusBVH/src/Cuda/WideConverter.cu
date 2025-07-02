#include "WideConverter.h"
#include "BuilderUtils.h"
#include <device_launch_parameters.h>

__global__ void NXB::BuildBVH8(BVH8BuildState buildState)
{
	uint32_t threadWarpId = threadIdx.x & (WARP_SIZE - 1);
	BVH2::Node* bvh2Nodes = buildState.bvh2Nodes;
	BVH8::Node* bvh8Nodes = buildState.bvh8Nodes;

	// NOTE: CUDA does not guarantee that thread blocks are executed in order (i.e. block 0 is executed first)
	// so we have to use atomic counters to assign work ids between threads
	uint32_t workId;
	if (threadWarpId == 0)
		workId = atomicAdd(buildState.workCounter, WARP_SIZE);

	workId = __shfl_sync(FULL_MASK, workId, 0) + threadWarpId;
	
	while (true)
	{
		uint64_t indexPair = buildState.indexPairs[workId];
		uint32_t bvh2NodeIdx = indexPair >> 32;
		uint32_t bvh8NodeIdx = (uint32_t)indexPair;

		// If no work assigned, skip
		if (bvh2NodeIdx == INVALID_IDX)
			continue;

		// If leaf node, create a new BVH8 leaf
		if (bvh2Nodes[bvh2NodeIdx].leftChild == INVALID_IDX)
		{
			// For now, a leaf only contains one triangle
			buildState.primIdx[bvh8NodeIdx] = bvh2Nodes[bvh2NodeIdx].rightChild;
		}

		uint32_t leafCount = 0;
		uint32_t innerCount = 0;
		uint32_t innerNodes[8];
		uint32_t leafNodes[8];

		AABB childBounds[2];
		BVH2::Node bvh2Node = bvh2Nodes[bvh2NodeIdx];
		uint32_t childrenIdx[2] = { bvh2Node.leftChild, bvh2Node.rightChild };

		// Top-down traversal until we get 8 nodes or no inner nodes are remaining
		while (leafCount + innerCount < 8 && innerCount > 0)
		{
			childBounds[0] = bvh2Nodes[childrenIdx[0]].bounds;
			childBounds[1] = bvh2Nodes[childrenIdx[1]].bounds;

			bool first = childBounds[0].Area() < childBounds[1].Area() ? 0 : 1;

			// Push both children onto the stack
			#pragma unroll
			for (uint32_t i = 0; i < 2; i++)
			{
				if (bvh2Nodes[childrenIdx[first]].leftChild == INVALID_IDX)
					leafNodes[leafCount++] = childrenIdx[first];
				else
					innerNodes[innerCount++] = childrenIdx[first];

				first = !first;
			}

			// Pop the last inner node from the stack
			uint32_t nodeIdx = innerNodes[--innerCount];
			childrenIdx[0] = bvh2Nodes[nodeIdx].leftChild;
			childrenIdx[1] = bvh2Nodes[nodeIdx].rightChild;
		}

		uint32_t nodeCount = atomicAdd(&buildState.nodeCount, innerCount + leafCount);

		for (uint32_t i = 0; i < innerCount + leafCount; i++)
		{
			uint32_t pair;
			if (i < innerCount)
				pair = ((uint64_t)innerNodes[i] << 32) | (nodeCount + i);
			else
				pair = ((uint64_t)leafNodes[i] << 32) | (nodeCount + i);

			uint32_t idx = i == 0 ? workId : nodeCount + i - 1;
			buildState.indexPairs[idx] = pair;
		}

		BVH8::Node bvh8Node;

		for (uint32_t i = 0; i < innerCount + leafCount; i++)
		{

		}
	}
}
