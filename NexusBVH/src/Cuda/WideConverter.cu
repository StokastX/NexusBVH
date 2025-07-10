#include "WideConverter.h"
#include "BuilderUtils.h"
#include <device_launch_parameters.h>

#define NQ 8

using byte = unsigned char;
constexpr float scale = 1.0f / ((float)(1 << NQ) - 1);

__device__ __forceinline__ uint32_t ceilLog2(float x)
{
    uint32_t ix = __float_as_uint(x);
    uint32_t exp = ((ix >> 23) & 0xFF);
    // check if x is exactly 2^exp => mantissa bits are zero
    bool isPow2 = (ix & ((1 << 23) - 1)) == 0;
    return exp + !isPow2;
}

__device__ __forceinline__ float invPow2(byte eBiased)
{
    return __uint_as_float((uint32_t)(254 - eBiased) << 23);
}


__global__ void NXB::BuildWideBVH(BVH8BuildState buildState)
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

	if (workId >= buildState.primCount)
		return;
	
	while (true)
	{
		uint64_t indexPair = buildState.indexPairs[workId];
		uint32_t bvh2NodeIdx = indexPair >> 32;
		uint32_t bvh8NodeIdx = (uint32_t)indexPair;

		// If no work assigned, skip
		if (bvh2NodeIdx == INVALID_IDX)
		{
			__nanosleep(1000);
			continue;
		}

		// If leaf node, create a new BVH8 leaf
		//if (bvh2Nodes[bvh2NodeIdx].leftChild == INVALID_IDX && *buildState.leafCounter > 0)
		if (bvh2Nodes[bvh2NodeIdx].leftChild == INVALID_IDX)
		{
			// For now, a leaf only contains one triangle
			buildState.primIdx[bvh8NodeIdx] = bvh2Nodes[bvh2NodeIdx].rightChild;
			break;
		}

		uint32_t leafCount = 0;
		uint32_t innerCount = 0;
		uint32_t innerNodes[8];
		uint32_t leafNodes[8];

		AABB childBounds[2];
		BVH2::Node bvh2Node = bvh2Nodes[bvh2NodeIdx];
		uint32_t childrenIdx[2] = { bvh2Node.leftChild, bvh2Node.rightChild };

		// Top-down traversal until we get 8 nodes or no inner nodes are remaining
		while (true)
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
			if (innerCount == 0 || leafCount + innerCount == 8)
				break;

			// Pop the last inner node from the stack
			uint32_t nodeIdx = innerNodes[--innerCount];
			childrenIdx[0] = bvh2Nodes[nodeIdx].leftChild;
			childrenIdx[1] = bvh2Nodes[nodeIdx].rightChild;
		}

		uint32_t childBaseIdx = atomicAdd(buildState.nodeCounter, innerCount);
		uint32_t workBaseIdx = atomicAdd(buildState.workAllocCounter, innerCount + leafCount - 1);
		uint32_t primBaseIdx = 0;
		if (leafCount > 0)
			primBaseIdx = atomicAdd(buildState.leafCounter, leafCount);

		for (uint32_t i = 0; i < innerCount + leafCount; i++)
		{
			uint64_t pair;
			if (i < leafCount)
				pair = ((uint64_t)leafNodes[i] << 32) | (primBaseIdx + i);
			else
				pair = ((uint64_t)innerNodes[i - leafCount] << 32) | (childBaseIdx + i - leafCount);

			uint32_t idx = i == 0 ? workId : workBaseIdx + i - 1;
			buildState.indexPairs[idx] = pair;
		}

		BVH8::NodeExplicit bvh8Node;
		float3 diagonal = bvh2Node.bounds.bMax - bvh2Node.bounds.bMin;

		bvh8Node.p = bvh2Node.bounds.bMin;
		bvh8Node.e[0] = ceilLog2(diagonal.x * scale);
		bvh8Node.e[1] = ceilLog2(diagonal.y * scale);
		bvh8Node.e[2] = ceilLog2(diagonal.z * scale);
		bvh8Node.imask = (((uint32_t)1 << innerCount) - 1);

		bvh8Node.childBaseIdx = childBaseIdx;
		bvh8Node.primBaseIdx = primBaseIdx;

		float3 invE = make_float3(invPow2(bvh8Node.e[0]), invPow2(bvh8Node.e[1]), invPow2(bvh8Node.e[2]));

		for (uint32_t i = 0; i < 8; i++)
		{
			bvh8Node.meta[i] = 0;
			AABB childBounds;

			if (i < leafCount)
			{
				bvh8Node.meta[i] |= 1 << 5 | i;
				childBounds = bvh2Nodes[leafNodes[i]].bounds;
			}
			else if (i < innerCount + leafCount)
			{
				bvh8Node.meta[i] |= 1 << 5;
				bvh8Node.meta[i] |= 24 + i - leafCount;
				childBounds = bvh2Nodes[innerNodes[i - leafCount]].bounds;
			}
			else
				continue;

			bvh8Node.qlox[i] = (byte)floorf((childBounds.bMin.x - bvh2Node.bounds.bMin.x) * invE.x);
			bvh8Node.qloy[i] = (byte)floorf((childBounds.bMin.y - bvh2Node.bounds.bMin.y) * invE.y);
			bvh8Node.qloz[i] = (byte)floorf((childBounds.bMin.z - bvh2Node.bounds.bMin.z) * invE.z);

			bvh8Node.qhix[i] = (byte)ceilf((childBounds.bMax.x - bvh2Node.bounds.bMin.x) * invE.x);
			bvh8Node.qhiy[i] = (byte)ceilf((childBounds.bMax.y - bvh2Node.bounds.bMin.y) * invE.y);
			bvh8Node.qhiz[i] = (byte)ceilf((childBounds.bMax.z - bvh2Node.bounds.bMin.z) * invE.z);
		}

		bvh8Nodes[bvh8NodeIdx] = *(BVH8::Node*)&bvh8Node;
	}
}
