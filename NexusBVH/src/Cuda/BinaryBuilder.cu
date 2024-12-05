#include "BinaryBuilder.h"
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Math/AABB.h"

#define WARP_SIZE 32

namespace NXB
{
	// Highest differing bit.
	// "In practice, logical xor can be used instead of finding the index as we can compare the numbers."
	static __forceinline__ __device__ uint32_t Delta(uint32_t a, uint32_t b, uint32_t primCount, uint64_t* mortonCodes)
	{
		if (a < 0 || b >= primCount) return (uint32_t)(-1);
		return mortonCodes[a] ^ mortonCodes[b];
	}

	// From Ciprian Apetrei: "Fast and Simple Agglomerative LBVH Construction"
	// See https://doi.org/10.2312/cgvc.20141206
	static __device__ uint32_t FindParentId(uint32_t left, uint32_t right, uint32_t primCount, uint64_t* mortonCodes)
	{
		if (left == 0 || (right != primCount &&
			Delta(right, right + 1, primCount, mortonCodes) < Delta(left - 1, left, primCount, mortonCodes)))
			return right;
		else
			return left;
	}

	__global__ void BuildBinaryBVH(BuildState buildState)
	{
		const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;

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
			node.firstPrimId = idx;
			node.primCount = 1;
			buildState.nodes[idx] = node;
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
				// TODO: Ploc merge

				// Remove last set bit
				warpMask = warpMask & (warpMask - 1);
			}

		}
	}
}