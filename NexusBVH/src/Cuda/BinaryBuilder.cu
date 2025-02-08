#include "BinaryBuilder.h"
#include "BuilderUtils.h"
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "NXB/AABB.h"

#define WARP_SIZE 32
#define SEARCH_RADIUS 8
#define FULL_MASK 0xffffffff

namespace NXB
{
	// Highest differing bit.
	// "In practice, logical xor can be used instead of finding the index as we can compare the numbers." (Apetrei)
	static __forceinline__ __device__ uint32_t Delta(uint32_t a, uint32_t b, uint64_t* mortonCodes)
	{
		return mortonCodes[a] ^ mortonCodes[b];
	}

	// From Ciprian Apetrei: "Fast and Simple Agglomerative LBVH Construction"
	// See https://doi.org/10.2312/cgvc.20141206
	static __device__ uint32_t FindParentId(uint32_t left, uint32_t right, uint32_t primCount, uint64_t* mortonCodes)
	{
		// TODO: test "primCount - 1" differing from paper here
		if (left == 0 || (right != primCount - 1 && Delta(right, right + 1, mortonCodes) < Delta(left - 1, left, mortonCodes)))
			return right;
		else
			return left - 1;
	}

	static __device__ uint32_t LoadIndices(uint32_t start, uint32_t end, uint32_t& clusterIdx, BuildState buildState, uint32_t offset)
	{
		uint32_t laneWarpId = threadIdx.x & (WARP_SIZE - 1);

		// Load up to WARP_SIZE / 2 cluster indices
		uint32_t index = laneWarpId - offset;
		bool validLaneId = index >= 0 && index < min(end - start, WARP_SIZE / 2);

		if (validLaneId)
			clusterIdx = buildState.clusterIdx[start + index];

		// Count of valid cluster indices among the threads that took part in the load
		uint32_t validClusterCount = __popc(__ballot_sync(FULL_MASK, validLaneId && clusterIdx != INVALID_IDX));

		return validClusterCount;
	}

	static __device__ void StoreIndices(uint32_t previousNumPrim, uint32_t clusterIdx, BuildState buildState, uint32_t lStart)
	{
		uint32_t laneWarpId = threadIdx.x & (WARP_SIZE - 1);

		if (laneWarpId < previousNumPrim)
			buildState.clusterIdx[lStart + laneWarpId] = clusterIdx;

		// Thread fence is necessary: any lane in another block later attempting to read clusterIdx
		// will have performed an atomicExch in main loop which means clusterIdx will be available
		__threadfence();
	}

	// PLOC++ based merging
	static inline __device__ uint32_t MergeClustersCreateBVH2Node(uint32_t numPrim, uint32_t nearestNeighbor, uint32_t& clusterIdx, AABB& clusterBounds, BuildState buildState)
	{
		uint32_t laneWarpId = threadIdx.x & (WARP_SIZE - 1);

		bool laneActive = laneWarpId < numPrim;

		uint32_t nearestNeighborNN = __shfl_sync(FULL_MASK, nearestNeighbor, nearestNeighbor) & 0xffffffff;
		bool mutualNeighbor = laneActive && laneWarpId == nearestNeighborNN;
		bool merge = mutualNeighbor && laneWarpId < nearestNeighbor;

		uint32_t mergeMask = __ballot_sync(FULL_MASK, merge);
		uint32_t mergeCount = __popc(mergeMask);

		uint32_t baseIdx;
		// Per-warp atomic to reduce global memory access
		if (laneWarpId == 0)
			baseIdx = atomicAdd(buildState.clusterCount, mergeCount);

		// Share baseIdx with warp
		baseIdx = __shfl_sync(FULL_MASK, baseIdx, 0);

		// Number of merging lanes with indices less than laneWarpId
		uint32_t relativeIdx = __popc(mergeMask << (WARP_SIZE - laneWarpId));

		uint32_t neighborClusterIdx = __shfl_sync(FULL_MASK, clusterIdx, nearestNeighbor);
		AABB neighborBounds = shfl_sync(FULL_MASK, clusterBounds, nearestNeighbor);

		if (merge)
		{
			clusterBounds.Grow(neighborBounds);

			BVH2::Node node;
			node.bounds = clusterBounds;
			node.leftChild = clusterIdx;
			node.rightChild = neighborClusterIdx;
			clusterIdx = baseIdx + relativeIdx;
			buildState.nodes[clusterIdx] = node;
		}

		// Cluster idx compaction
		uint32_t validMask = __ballot_sync(FULL_MASK, merge || !mutualNeighbor);

		// Shift = cluster idx before compaction
		int32_t shift = __fns(validMask, 0, laneWarpId + 1);

		clusterIdx = __shfl_sync(FULL_MASK, clusterIdx, shift);
		if (shift == -1)
			clusterIdx = INVALID_IDX;

		clusterBounds = shfl_sync(FULL_MASK, clusterBounds, shift);

		return numPrim - mergeCount;
	}

	// PLOC++ based nearest neighbor search
	static inline __device__ uint32_t FindNearestNeighbor(uint32_t numPrim, uint32_t clusterIdx, AABB clusterBounds, BuildState buildState)
	{
		int32_t laneWarpId = threadIdx.x & (WARP_SIZE - 1);

		uint2 minAreaIdx = make_uint2(INVALID_IDX);

		for (int32_t r = 1; r <= SEARCH_RADIUS; r++)
		{
			uint32_t neighborIdx = laneWarpId + r;
			uint32_t area = (uint32_t)(-1);
			AABB neighborBounds = shfl_sync(FULL_MASK, clusterBounds, neighborIdx);

			if (neighborIdx < numPrim)
			{
				neighborBounds.Grow(clusterBounds);

				// Encode area + neighbor index in a 64-bit variable
				area = __float_as_uint(neighborBounds.Area());

				// Update min_distance[i, i + r]
				if (area < minAreaIdx.x)
					minAreaIdx = make_uint2(area, neighborIdx);
			}

			// Get nearest neighbor of cluster i + r
			uint2 neighborNN = shfl_sync(FULL_MASK, minAreaIdx, neighborIdx);

			// Update min_distance[i + r, i]
			if (area < neighborNN.x)
				neighborNN = make_uint2(area, laneWarpId);

			// Get result back from cluster i - r
			minAreaIdx = shfl_sync(FULL_MASK, neighborNN, laneWarpId - r);
		}

		return minAreaIdx.y;
	}

	static __device__ void PlocMerge(uint32_t laneId, uint32_t left, uint32_t right, uint32_t split, bool final, BuildState buildState)
	{
		// Share current lane's LBVH node with other threads in the warp
		uint32_t lStart = __shfl_sync(FULL_MASK, left, laneId);
		uint32_t rEnd = __shfl_sync(FULL_MASK, right, laneId) + 1;
		uint32_t lEnd = __shfl_sync(FULL_MASK, split, laneId);
		uint32_t rStart = lEnd;

		// Thread index in the warp
		uint32_t laneWarpId = threadIdx.x & (WARP_SIZE - 1);

		uint32_t clusterIdx = INVALID_IDX;

		// Load left and right child's cluster indices into shared memory
		uint32_t numLeft = LoadIndices(lStart, lEnd, clusterIdx, buildState, 0);
		uint32_t numRight = LoadIndices(rStart, rEnd, clusterIdx, buildState, numLeft);
		uint32_t numPrim = numLeft + numRight;

		AABB clusterBounds;
		if (laneWarpId < numPrim)
			clusterBounds = buildState.nodes[clusterIdx].bounds;

		// If we reached the root node, we want to merge all the remaining clusters (ie threshold = 1)
		uint32_t threshold = __shfl_sync(FULL_MASK, final, laneId) ? 1 : WARP_SIZE / 2;

		while (numPrim > threshold)
		{
			uint32_t nearestNeighbor = FindNearestNeighbor(numPrim, clusterIdx, clusterBounds, buildState);
			numPrim = MergeClustersCreateBVH2Node(numPrim, nearestNeighbor, clusterIdx, clusterBounds, buildState);
		}

		StoreIndices(numLeft + numRight, clusterIdx, buildState, lStart);
	}

	__global__ void BuildBinaryBVH(BuildState buildState)
	{
		const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;

		// Left and right bounds of the current LBVH node
		uint32_t left = idx;
		uint32_t right = idx;

		// Index of the current LBVH node
		uint32_t split = 0;

		bool laneActive = idx < buildState.primCount;

		// Do bottom-up traversal as long as active lanes in warp
		while (__ballot_sync(FULL_MASK, laneActive))
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

			uint32_t warpMask = __ballot_sync(FULL_MASK, laneActive && (size > WARP_SIZE / 2) || final);

			while (warpMask)
			{
				// Trailing zero count
				uint32_t laneId = __ffs(warpMask) - 1;

				PlocMerge(laneId, left, right, split, final, buildState);

				// Remove last set bit
				warpMask = warpMask & (warpMask - 1);
			}
		}
	}
}