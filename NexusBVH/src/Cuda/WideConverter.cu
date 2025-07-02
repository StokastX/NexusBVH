#include "WideConverter.h"
#include "BuilderUtils.h"
#include <device_launch_parameters.h>

__global__ void NXB::BuildWideBVH(BVH8BuildState buildState)
{
	uint32_t threadWarpId = threadIdx.x & (WARP_SIZE - 1);

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

		if (bvh2NodeIdx == INVALID_IDX)
			continue;

		// If leaf node, create a new BVH8 leaf
		if (buildState.bvh2Nodes[bvh2NodeIdx].leftChild == INVALID_IDX)
		{
			// For now, a leaf only contains one triangle
			buildState.primIdx[bvh8NodeIdx] = buildState.bvh2Nodes[bvh2NodeIdx].rightChild;
		}

		uint32_t childCount = 0;
		while (childCount < 8)
		{
		}

	}
}
