#include "BinaryBuilder.h"
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Math/AABB.h"

#define WARP_SIZE 32
#define BLOCK_SIZE 64

namespace NXB
{
	// Highest differing bit.
	// "In practice, logical xor can be used instead of finding the index as we can compare the numbers."
	static __forceinline__ __device__ uint32_t Delta(uint32_t a, uint32_t b, uint64_t* mortonCodes)
	{
		return mortonCodes[a] ^ mortonCodes[b];
	}

	// From Ciprian Apetrei: "Fast and Simple Agglomerative LBVH Construction"
	// See https://doi.org/10.2312/cgvc.20141206
	static __device__ uint32_t FindParentId(uint32_t left, uint32_t right, uint32_t primCount, uint64_t* mortonCodes)
	{
		if (left == 0 || (right != primCount - 1 && Delta(right, right + 1, mortonCodes) < Delta(left - 1, left, mortonCodes)))
			return right;
		else
			return left - 1;
	}

	// Load cluster indices of one of the LBVH node's children from global to shared memory
	// At the end of the function, the warp contains up to WARP_SIZE / 2 cluster indices in shared memory
	static __device__ uint32_t LoadIndices(uint32_t start, uint32_t end, uint32_t clusterIdx[][WARP_SIZE], BuildState buildState, uint32_t offset)
	{
		uint32_t laneWarpId = threadIdx.x & (WARP_SIZE - 1);

		// Load up to WARP_SIZE / 2 cluster indices
		bool validLaneId = laneWarpId < min(end - start + 1, WARP_SIZE / 2);

		if (validLaneId)
			clusterIdx[blockIdx.x][laneWarpId + offset] = buildState.clusterIdx[start + laneWarpId];

		__syncthreads();

		// Count of valid cluster indices among the threads that took part in the load
		uint32_t validClusterCount = __popc(__ballot_sync(0xffffffff, validLaneId && clusterIdx[blockIdx.x][laneWarpId + offset] != INVALID_IDX));

		return validClusterCount;
	}

	static __device__ void PlocMerge(uint32_t laneId, uint32_t left, uint32_t right, uint32_t split, bool final, uint32_t clusterIdx[][WARP_SIZE], uint32_t nearestNeighbor[][WARP_SIZE], BuildState buildState)
	{
		// Share current lane's LBVH node with other threads in the warp
		uint32_t lStart = __shfl_sync(0xffffffff, left, laneId);
		uint32_t rEnd = __shfl_sync(0xffffffff, right, laneId);
		uint32_t lEnd = __shfl_sync(0xffffffff, split, laneId);
		uint32_t rStart = __shfl_sync(0xffffffff, split, laneId);

		// Load left and right child's cluster indices into shared memory
		uint32_t numLeft = LoadIndices(lStart, lEnd, clusterIdx, buildState, 0);
		uint32_t numRight = LoadIndices(rStart, rEnd, clusterIdx, buildState, numLeft);
		uint32_t numPrim = numLeft + numRight;

		// If we reached the root node, we want to merge all the remaining clusters (ie threshold = 1)
		uint32_t threshold = __shfl_sync(0xffffffff, final, laneId) ? 1 : WARP_SIZE / 2;

		while (numPrim > threshold)
		{
			// TODO: find nearest neighbor and merge clusters
		}

		// Thread index in the warp
		uint32_t laneWarpId = threadIdx.x & (WARP_SIZE - 1);
	}

	__global__ void BuildBinaryBVH(BuildState buildState)
	{
		const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;

		__shared__ uint32_t clusterIdx[BLOCK_SIZE / WARP_SIZE][WARP_SIZE];
		__shared__ uint32_t nearestNeighbor[BLOCK_SIZE / WARP_SIZE][WARP_SIZE];

		// Left and right bounds of the current LBVH node
		uint32_t left = idx;
		uint32_t right = idx;

		// Index of the current LBVH node
		uint32_t split = 0;

		bool laneActive = false;
		if (idx < buildState.primCount)
		{
			laneActive = true;

			// Initialize first N leaf nodes
			BVH2::Node node;
			node.bounds = buildState.primBounds[buildState.primIdx[idx]];
			node.leftChild = INVALID_IDX;
			node.rightChild = idx;
			buildState.nodes[idx] = node;

			// Initialize cluster indices to leaf node indices
			buildState.clusterIdx[idx] = idx;
		}

		// Do bottom-up traversal as long as active lanes in warp
		while (__ballot_sync(0xffffffff, laneActive))
		{
			if (laneActive)
			{
				int32_t previousId;
				if (FindParentId(left, right, buildState.primCount, buildState.mortonCodes) == right)
				{
					// Parent is at the right boundary, current LBVH node is its left child
					previousId = atomicExch(&buildState.parentIdx[right], left);

					// If right child has already reached parent
					if (previousId != -1)
					{
						split = right + 1;

						// PreviousId contains right boundary since atomicExch
						// has previously been called by right child
						right = previousId;
					}
				}
				else
				{
					// Parent is at the left boundary, current LBVH node is its right child
					previousId = atomicExch(&buildState.parentIdx[left - 1], right);

					// If left child has already reached parent
					if (previousId != -1)
					{
						split = left;

						// PreviousId contains left boundary since atomicExch
						// has previously been called by left child
						left = previousId;
					}
				}

				// Stop traversal and let the other child reach the parent
				if (previousId == -1)
					laneActive = false;
			}

			uint32_t size = right - left + 1;
			bool final = laneActive && size == buildState.primCount;
			uint32_t warpMask = __ballot_sync(0xffffffff, laneActive && (size > WARP_SIZE / 2) || final);

			while (warpMask)
			{
				// Trailing zero count
				uint32_t laneId = __ffs(warpMask) - 1;

				PlocMerge(laneId, left, right, split, final, clusterIdx, nearestNeighbor, buildState);

				// Remove last set bit
				warpMask = warpMask & (warpMask - 1);
			}
		}
	}
}