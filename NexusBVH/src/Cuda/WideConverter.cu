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

		uint32_t childBaseIdx = atomicAdd(buildState.nodeCount, innerCount);
		uint32_t primBaseIdx = atomicAdd(buildState.leafCount, leafCount);

		for (uint32_t i = 0; i < innerCount + leafCount; i++)
		{
			uint32_t pair;
			if (i < leafCount)
				pair = ((uint64_t)leafNodes[i] << 32) | (primBaseIdx + i);
			else
				pair = ((uint64_t)innerNodes[i] << 32) | (childBaseIdx + i - leafCount);

			uint32_t idx = i == 0 ? workId : childBaseIdx + i - 1;
			buildState.indexPairs[idx] = pair;
		}

		BVH8::Node bvh8Node;
		AABB bounds = bvh2Node.bounds;
		float3 diagonal = bvh2Node.bounds.bMax - bvh2Node.bounds.bMin;

		byte ex = ceilLog2(diagonal.x * scale);
		byte ey = ceilLog2(diagonal.y * scale);
		byte ez = ceilLog2(diagonal.z * scale);

		uint32_t e_imask = (uint32_t)ex << 24 | (uint32_t)ey << 16 | (uint32_t)ez << 8 | (((uint32_t)1 << innerCount) - 1);
		
		bvh8Node.p_e_imask.x = bvh2Node.bounds.bMin.x;
		bvh8Node.p_e_imask.y = bvh2Node.bounds.bMin.y;
		bvh8Node.p_e_imask.z = bvh2Node.bounds.bMin.z;
		bvh8Node.p_e_imask.w = __uint_as_float(e_imask);

		bvh8Node.childidx_tridx_meta.x = __uint_as_float(childBaseIdx);
		bvh8Node.childidx_tridx_meta.y = __uint_as_float(primBaseIdx);

		float3 invE = make_float3(invPow2(ex), invPow2(ey), invPow2(ez));
		byte* meta = (byte*)&bvh8Node.childidx_tridx_meta.z;
		byte* qlox = (byte*)&bvh8Node.qlox_qloy;
		byte* qloy = (byte*)&bvh8Node.qlox_qloy.z;
		byte* qloz = (byte*)&bvh8Node.qloz_qhix;
		byte* qhix = (byte*)&bvh8Node.qloz_qhix.z;
		byte* qhiy = (byte*)&bvh8Node.qhiy_qhiz;
		byte* qhiz = (byte*)&bvh8Node.qhiy_qhiz.z;

		for (uint32_t i = 0; i < innerCount + leafCount; i++)
		{
			meta[i] = 0;
			AABB childBounds;

			if (i < leafCount)
			{
				meta[i] |= 1 << 5 | i;
				childBounds = bvh2Nodes[leafNodes[i]].bounds;
			}
			else if (i < leafCount + innerCount)
			{
				meta[i] |= 1 << 5;
				meta[i] |= 24 + i - leafCount;
				childBounds = bvh2Nodes[innerNodes[i - leafCount]].bounds;
			}

			qlox[i] = (byte)floorf((childBounds.bMin.x - bvh2Node.bounds.bMin.x) * invE.x);
			qloy[i] = (byte)floorf((childBounds.bMin.y - bvh2Node.bounds.bMin.y) * invE.y);
			qloz[i] = (byte)floorf((childBounds.bMin.z - bvh2Node.bounds.bMin.z) * invE.z);

			qhix[i] = (byte)ceilf((childBounds.bMax.x - bvh2Node.bounds.bMin.x) * invE.x);
			qhiy[i] = (byte)ceilf((childBounds.bMax.y - bvh2Node.bounds.bMin.y) * invE.y);
			qhiz[i] = (byte)ceilf((childBounds.bMax.z - bvh2Node.bounds.bMin.z) * invE.z);
		}

		bvh8Nodes[bvh8NodeIdx] = bvh8Node;
	}
}
