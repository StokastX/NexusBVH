#include "WideConverter.h"
#include "BuilderUtils.h"
#include <device_launch_parameters.h>

#define NQ 8

using byte = unsigned char;
constexpr float scale = 1.0f / ((float)(1 << NQ) - 1);

__device__ __forceinline__ uint32_t CeilLog2(float x)
{
    uint32_t ix = __float_as_uint(x);
    uint32_t exp = ((ix >> 23) & 0xFF);
    // check if x is exactly 2^exp => mantissa bits are zero
    bool isPow2 = (ix & ((1 << 23) - 1)) == 0;
    return exp + !isPow2;
}

__device__ __forceinline__ float InvPow2(byte eBiased)
{
    return __uint_as_float((uint32_t)(254 - eBiased) << 23);
}

__device__ __forceinline__ uint64_t GlobalLoad(uint64_t* addr)
{
	uint64_t value;
	asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(value) : "l"(addr));
	return value;
}

__device__ __forceinline__ void GlobalStore(uint64_t* addr, uint64_t value)
{
    asm volatile("st.global.cg.u64 [%0], %1;" :: "l"(addr), "l"(value));
}

__device__ __forceinline__ uint32_t CountBitsBelow(uint32_t x, uint32_t i)
{
	uint32_t mask = (1u << i) - 1;
	return __popc(x & mask);
}

namespace NXB
{
	__global__ void BuildBVH8Kernel(BVH8BuildState buildState)
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
			// We don't want to load index pairs from L1 cache, since we want the global updated version
			uint64_t indexPair = GlobalLoad(&buildState.indexPairs[workId]);
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
				break;
			}

			// Mask to know which of the children are inner nodes
			uint32_t innerMask = 0;
			uint32_t childCount = 0;
			uint32_t childNodes[8];

			AABB childBounds[2];
			BVH2::Node bvh2Node = bvh2Nodes[bvh2NodeIdx];
			uint32_t leftRightChild[2] = { bvh2Node.leftChild, bvh2Node.rightChild };
			int32_t msb = 0;

			// Top-down traversal until we get 8 nodes or no inner nodes are remaining
			while (true)
			{
				childBounds[0] = bvh2Nodes[leftRightChild[0]].bounds;
				childBounds[1] = bvh2Nodes[leftRightChild[1]].bounds;

				bool first = childBounds[0].Area() < childBounds[1].Area() ? 0 : 1;

				// Push both children onto the stack
#pragma unroll
				for (uint32_t i = 0; i < 2; i++)
				{
					uint32_t idx = i == 0 ? msb : childCount;

					if (bvh2Nodes[leftRightChild[first]].leftChild != INVALID_IDX)
						innerMask |= 1 << idx;

					childNodes[idx] = leftRightChild[first];

					childCount++;
					first = !first;
				}

				// Pop the last inner node from the stack
				msb = 31 - __clz(innerMask);

				if (msb < 0 || childCount == 8)
					break;

				// Set the msb to 0
				innerMask &= ~(1 << msb);
				childCount--;

				uint32_t newIdx = childNodes[msb];
				leftRightChild[0] = bvh2Nodes[newIdx].leftChild;
				leftRightChild[1] = bvh2Nodes[newIdx].rightChild;
			}

			uint32_t innerCount = __popc(innerMask);
			uint32_t leafCount = childCount - innerCount;

			uint32_t childBaseIdx = atomicAdd(buildState.nodeCounter, innerCount);
			uint32_t workBaseIdx = atomicAdd(buildState.workAllocCounter, childCount - 1);

			uint32_t primBaseIdx = 0;
			if (leafCount > 0)
				primBaseIdx = atomicAdd(buildState.leafCounter, leafCount);

			for (uint32_t i = 0; i < childCount; i++)
			{
				uint64_t pair = (uint64_t)childNodes[i] << 32;
				if (innerMask & (1 << i))
					pair |= childBaseIdx + CountBitsBelow(innerMask, i);
				else
					pair |= primBaseIdx + CountBitsBelow(~innerMask, i);

				uint32_t idx = i == 0 ? workId : workBaseIdx + i - 1;
				GlobalStore(&buildState.indexPairs[idx], pair);
			}

			// Reorder children using auction algorithm
			// See https://dspace.mit.edu/bitstream/handle/1721.1/3233/P-2064-24690022.pdf

			float3 parentCentroid = (bvh2Node.bounds.bMin + bvh2Node.bounds.bMax) * 0.5f;

			// Fill the table cost(c, s)
			float cost[8][8];

			for (uint32_t c = 0; c < childCount; c++)
			{
				// If no more children, break
				//if (childNodes[c] == -1)
				//	break;
				AABB childBounds = bvh2Nodes[childNodes[c]].bounds;

				for (int s = 0; s < 8; s++)
				{
					// Ray direction
					float dsx = (s & 0b100) ? -1.0f : 1.0f;
					float dsy = (s & 0b010) ? -1.0f : 1.0f;
					float dsz = (s & 0b001) ? -1.0f : 1.0f;
					float3 ds = make_float3(dsx, dsy, dsz);

					float3 centroid = (childBounds.bMin + childBounds.bMax) * 0.5f;
					cost[c][s] = dot(centroid - parentCentroid, ds);
				}
			}

			float prices[8];
			uint32_t assignments[8];
			uint32_t bidders[8];
			for (uint32_t i = 0; i < 8; i++)
			{
				if (i < childCount)
					assignments[i] = i;
				else
					assignments[i] = INVALID_IDX;
			}
			assignments[1] = 0;
			assignments[0] = 1;
			//for (uint32_t i = 0; i < 8; i++)
			//{
			//	prices[i] = 0.0f;
			//	assignments[i] = INVALID_IDX;
			//	bidders[i] = i;
			//}

			//uint32_t bidderCount = childCount;
			//float epsilon = 1.0f / childCount;

			//while (bidderCount > 0)
			//{
			//	uint32_t c = bidders[--bidderCount];
			//	float winningReward = -FLT_MAX;
			//	float secondWinningReward = -FLT_MAX;
			//	uint32_t winningSlot = INVALID_IDX;
			//	uint32_t secondWinningSlot = INVALID_IDX;

			//	for (uint32_t s = 0; s < 8; s++)
			//	{
			//		float reward = cost[c][s] - prices[s];
			//		if (reward > winningReward)
			//		{
			//			winningReward = reward;
			//			secondWinningReward = winningReward;
			//			winningSlot = s;
			//			secondWinningSlot = winningSlot;
			//		}
			//		else if (reward > secondWinningReward)
			//		{
			//			secondWinningReward = reward;
			//			secondWinningSlot = s;
			//		}
			//	}

			//	prices[winningSlot] += (winningReward - secondWinningReward) + epsilon;

			//	uint32_t previousAssignment = assignments[winningSlot];
			//	assignments[winningSlot] = c;

			//	if (previousAssignment != INVALID_IDX)
			//		bidders[bidderCount++] = previousAssignment;
			//}

			uint32_t newInnerMask = 0;
			for (uint32_t i = 0; i < 8; i++)
			{
				uint32_t bit = assignments[i] == INVALID_IDX ? 0 : (innerMask >> assignments[i]) & 1;
				newInnerMask |= (bit << i);
			}
			innerMask = newInnerMask;

			BVH8::NodeExplicit bvh8Node;
			float3 diagonal = bvh2Node.bounds.bMax - bvh2Node.bounds.bMin;

			bvh8Node.p = bvh2Node.bounds.bMin;
			bvh8Node.e[0] = CeilLog2(diagonal.x * scale);
			bvh8Node.e[1] = CeilLog2(diagonal.y * scale);
			bvh8Node.e[2] = CeilLog2(diagonal.z * scale);
			bvh8Node.imask = 0;

			bvh8Node.childBaseIdx = childBaseIdx;
			bvh8Node.primBaseIdx = primBaseIdx;

			float3 invE = make_float3(InvPow2(bvh8Node.e[0]), InvPow2(bvh8Node.e[1]), InvPow2(bvh8Node.e[2]));

			for (uint32_t i = 0; i < 8; i++)
			{
				bvh8Node.meta[i] = 0;
				uint32_t childIdx = assignments[i];

				if (innerMask & (1 << i))
				{
					bvh8Node.imask |= 1 << i;
					bvh8Node.meta[i] |= 1 << 5;
					bvh8Node.meta[i] |= 24 + i;
				}
				else if (childIdx != INVALID_IDX)
				{
					bvh8Node.meta[i] |= (1 << 5) | CountBitsBelow(~innerMask, i);
				}
				else
					continue;

				AABB childBounds = bvh2Nodes[childNodes[childIdx]].bounds;
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
}
