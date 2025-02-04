#include "BinaryBuilder.h"
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Math/AABB.h"

#define WARP_SIZE 32
#define BLOCK_SIZE 64
#define FULL_MASK 0xffffffff

namespace NXB
{
	// float3 version of __shfl_sync
	static __forceinline__ __device__ float3 shfl_sync(uint32_t mask, float3 value, uint32_t shift)
	{
		float x = __shfl_sync(mask, value.x, shift);
		float y = __shfl_sync(mask, value.y, shift);
		float z = __shfl_sync(mask, value.z, shift);
		return make_float3(x, y, z);
	}

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

	// Load cluster indices of one of the LBVH node's children from global to shared memory
	// At the end of the function, the warp contains up to WARP_SIZE / 2 cluster indices in shared memory
	static __device__ uint32_t LoadIndices(uint32_t start, uint32_t end, uint32_t* clusterIdx, BuildState buildState, uint32_t offset)
	{
		uint32_t laneWarpId = threadIdx.x & (WARP_SIZE - 1);

		// Load up to WARP_SIZE / 2 cluster indices
		bool validLaneId = laneWarpId < min(end - start, WARP_SIZE / 2);

		if (validLaneId)
			clusterIdx[laneWarpId + offset] = buildState.clusterIdx[start + laneWarpId];

		__syncthreads();

		// Count of valid cluster indices among the threads that took part in the load
		uint32_t validClusterCount = __popc(__ballot_sync(FULL_MASK, validLaneId && clusterIdx[laneWarpId + offset] != INVALID_IDX));

		return validClusterCount;
	}

	static __device__ void StoreIndices(uint32_t previousNumPrim, uint32_t* clusterIdx, BuildState buildState, uint32_t lStart)
	{
		uint32_t laneWarpId = threadIdx.x & (WARP_SIZE - 1);

		if (laneWarpId < previousNumPrim)
			buildState.clusterIdx[lStart + laneWarpId] = clusterIdx[laneWarpId];

		__threadfence();
	}

	// PLOC++ based merging
	static __device__ uint32_t MergeClustersCreateBVH2Node(uint32_t numPrim, uint64_t* nearestNeighbors, uint32_t* clusterIdx, AABB* clusterBounds, BuildState buildState)
	{
		uint32_t laneWarpId = threadIdx.x & (WARP_SIZE - 1);
		uint32_t newClusterIdx = INVALID_IDX;
		AABB newClusterBounds;
		uint32_t nearestNeighbor;

		bool laneActive = laneWarpId < numPrim;

		if (laneActive)
		{
			newClusterIdx = clusterIdx[laneWarpId];
			newClusterBounds = clusterBounds[laneWarpId];
			nearestNeighbor = nearestNeighbors[laneWarpId] & 0xffffffff;
		}

		bool mutualNeighbor = laneActive && laneWarpId == (nearestNeighbors[nearestNeighbor] & 0xffffffff);
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

		if (merge)
		{
			newClusterBounds.Grow(clusterBounds[nearestNeighbor]);

			BVH2::Node node;
			node.bounds = newClusterBounds;
			node.leftChild = newClusterIdx;
			node.rightChild = clusterIdx[nearestNeighbor];
			newClusterIdx = baseIdx + relativeIdx;
			buildState.nodes[newClusterIdx] = node;
		}

		// Cluster idx compaction
		uint32_t validMask = __ballot_sync(FULL_MASK, merge || !mutualNeighbor);

		// Shift = cluster idx before compaction
		int32_t shift = __fns(validMask, 0, laneWarpId + 1);

		clusterIdx[laneWarpId] = __shfl_sync(FULL_MASK, newClusterIdx, shift);
		if (shift == -1)
			clusterIdx[laneWarpId] = INVALID_IDX;

		AABB aabb;
		aabb.bMin = shfl_sync(FULL_MASK, newClusterBounds.bMin, shift);
		aabb.bMax = shfl_sync(FULL_MASK, newClusterBounds.bMax, shift);
		clusterBounds[laneWarpId] = aabb;

		__syncthreads();

		return numPrim - mergeCount;
	}

	// PLOC++ based nearest neighbor search
	static __device__ void FindNearestNeighbor(uint64_t* nearestNeighbors, uint32_t numPrim, uint32_t* clusterIdx, AABB* clusterBounds, BuildState buildState)
	{
		uint32_t laneWarpId = threadIdx.x & (WARP_SIZE - 1);
		nearestNeighbors[laneWarpId] = (uint64_t)(-1);

		__syncthreads();

		if (laneWarpId < numPrim)
		{
			AABB aabb = clusterBounds[laneWarpId];
			uint64_t minAreaIdx = (uint64_t)(-1);

			for (uint32_t r = 1; r <= WARP_SIZE / 2; r++)
			{
				uint32_t neighborIdx = laneWarpId + r;

				if (neighborIdx < numPrim)
				{
					AABB neighborBounds = clusterBounds[neighborIdx];
					neighborBounds.Grow(aabb);

					// Encode area + neighbor index in a 64-bit variable
					uint32_t area = __float_as_uint(neighborBounds.Area());
					uint64_t encode0 = (uint64_t)area << 32 | neighborIdx;
					uint64_t encode1 = (uint64_t)area << 32 | laneWarpId;

					// Update min_distance[i, i + r]
					minAreaIdx = min(minAreaIdx, encode0);

					// Update min_distance[i + r, i]
					atomicMin(&nearestNeighbors[neighborIdx], encode1);
				}
			}
			// Store closest neighbor back in shared memory
			atomicMin(&nearestNeighbors[laneWarpId], minAreaIdx);
		}
	}

	static __device__ void PlocMerge(uint32_t laneId, uint32_t left, uint32_t right, uint32_t split, bool final, uint32_t* clusterIdx, uint64_t* nearestNeighbor, AABB* clusterBounds, BuildState buildState)
	{
		// Share current lane's LBVH node with other threads in the warp
		uint32_t lStart = __shfl_sync(FULL_MASK, left, laneId);
		uint32_t rEnd = __shfl_sync(FULL_MASK, right, laneId) + 1;
		uint32_t lEnd = __shfl_sync(FULL_MASK, split, laneId);
		uint32_t rStart = lEnd;

		// Thread index in the warp
		uint32_t laneWarpId = threadIdx.x & (WARP_SIZE - 1);

		clusterIdx[laneWarpId] = INVALID_IDX;
		__syncthreads();

		// Load left and right child's cluster indices into shared memory
		uint32_t numLeft = LoadIndices(lStart, lEnd, clusterIdx, buildState, 0);
		uint32_t numRight = LoadIndices(rStart, rEnd, clusterIdx, buildState, numLeft);
		uint32_t numPrim = numLeft + numRight;

		// Load cluster bounds in shared memory
		if (laneWarpId < numPrim)
			clusterBounds[laneWarpId] = buildState.nodes[clusterIdx[laneWarpId]].bounds;

		__syncthreads();

		// If we reached the root node, we want to merge all the remaining clusters (ie threshold = 1)
		uint32_t threshold = __shfl_sync(FULL_MASK, final, laneId) ? 1 : WARP_SIZE / 2;


		while (numPrim > threshold)
		{
			FindNearestNeighbor(nearestNeighbor, numPrim, clusterIdx, clusterBounds, buildState);
			numPrim = MergeClustersCreateBVH2Node(numPrim, nearestNeighbor, clusterIdx, clusterBounds, buildState);
		}

		StoreIndices(numLeft + numRight, clusterIdx, buildState, lStart);

	}

	__global__ void NXB::BuildBinaryBVH(BuildState buildState)
	{
		const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;

		// Cached memory for cluster search and merging
		__shared__ uint32_t clusterIdx[BLOCK_SIZE / WARP_SIZE][WARP_SIZE];
		__shared__ AABB clusterBounds[BLOCK_SIZE / WARP_SIZE][WARP_SIZE];

		// Nearest distance (32 bits), nearest neighbor idx (32 bits)
		__shared__ uint64_t nearestNeighbor[BLOCK_SIZE / WARP_SIZE][WARP_SIZE];

		// Left and right bounds of the current LBVH node
		uint32_t left = idx;
		uint32_t right = idx;
		uint32_t warpId = threadIdx.x / WARP_SIZE;

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

				PlocMerge(laneId, left, right, split, final, clusterIdx[warpId], nearestNeighbor[warpId], clusterBounds[warpId], buildState);

				// Remove last set bit
				warpMask = warpMask & (warpMask - 1);
			}
		}
	}
}