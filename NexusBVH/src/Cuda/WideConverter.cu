#include "WideConverter.h"
#include "BuilderUtils.h"
#include <float.h>
#include <device_launch_parameters.h>

//#define USE_AUCTION
#define INVALID_ASSIGNMENT 0xf

// Quantization scale
#define NQ 8

// Factor used in epsilon scaling for the auction algorithm
// A larger theta reduces the number of iterations which result in a less precise assignment
#define THETA 8

// Scaling factor that determines an order of magnitude of the maximum possible cost of the cost table
#define MAX_COST 10.0f

constexpr float quantStep = 1.0f / ((float)(1 << NQ) - 1);
constexpr float invTheta = 1.0f / (float)THETA;


namespace NXB
{
	// Compute the cost of placing child c in slot s
	__device__ __forceinline__ float GetCost(uint32_t c, uint32_t s, float3 offsets[8])
	{
		float3 offset = offsets[c];
		return ((s >> 2) & 1 ? -1.0f : 1.0f) * offset.x +
			   ((s >> 1) & 1 ? -1.0f : 1.0f) * offset.y +
			   (s & 1 ? -1.0f : 1.0f) * offset.z;
	}

	__device__ __forceinline__ float GetCost(uint32_t s, float3 offset)
	{
		return ((s >> 2) & 1 ? -1.0f : 1.0f) * offset.x +
			   ((s >> 1) & 1 ? -1.0f : 1.0f) * offset.y +
			   (s & 1 ? -1.0f : 1.0f) * offset.z;
	}

	// For debugging purposes
	__device__ float ComputeAssignmentCost(float3 offsets[8], uint32_t assignments)
	{
		float cost = 0.0f;
		for (uint32_t s = 0; s < 8; s++)
		{
			if (GetNibble(assignments, s) != INVALID_ASSIGNMENT)
				cost += GetCost(GetNibble(assignments, s), s, offsets);
		}
		return cost;
	}

	// Auction algorithm
	// See https://dspace.mit.edu/bitstream/handle/1721.1/3233/P-2064-24690022.pdf
	__device__ __forceinline__ void AuctionAssignment(float3 offsets[8], float maxCost, uint32_t n, uint32_t& assignments)
	{
		float prices[8];
		for (uint32_t i = 0; i < 8; i++)
			prices[i] = 0.0f;

		// Initialize epsilon, see p.260 of https://web.mit.edu/dimitrib/www/netbook_Full_Book_NEW.pdf
		// and p.13 of https://www.math.purdue.edu/~jacob225/papers/auction_revised.pdf
		float threshold = 1.0f / n;
		float epsilon = fmaxf(maxCost, threshold);

		while (epsilon >= threshold)
		{
			// Reset assignments
			assignments = INVALID_IDX;
			// Reset bidders with slots ranging from 0 to 7
			uint32_t bidders = 0x76543210;
			uint32_t bidderCount = n;

			while (bidderCount > 0)
			{
				uint32_t c = GetNibble(bidders, --bidderCount);
				float winningReward = -FLT_MAX;
				float secondWinningReward = -FLT_MAX;
				uint32_t winningSlot = INVALID_ASSIGNMENT;
				uint32_t secondWinningSlot = INVALID_ASSIGNMENT;

				for (uint32_t s = 0; s < 8; s++)
				{
					float reward = GetCost(c, s, offsets) - prices[s];
					if (reward > winningReward)
					{
						secondWinningReward = winningReward;
						winningReward = reward;
						secondWinningSlot = winningSlot;
						winningSlot = s;
					}
					else if (reward > secondWinningReward)
					{
						secondWinningReward = reward;
						secondWinningSlot = s;
					}
				}

				prices[winningSlot] += (winningReward - secondWinningReward) + epsilon;

				uint32_t previousAssignment = GetNibble(assignments, winningSlot);
				SetNibble(assignments, winningSlot, c);

				if (previousAssignment != INVALID_ASSIGNMENT)
					SetNibble(bidders, bidderCount++, previousAssignment);
			}
			// Epsilon scaling
			epsilon *= invTheta;
		}
	}

	// Greedy assignment algorithm detailed in the BVH8 paper.
	// It's actually way faster than auction and gives similar tracing times, so I might as well stick with it.
	__device__ __forceinline__ void GreedyAssignment(float3 parentCentroid, uint32_t childNodes[8], uint32_t n, uint32_t& assignments, BVH8BuildState buildState)
	{
		assignments = INVALID_IDX;
		//uint32_t slotsAvailable = 0xff;

		for (uint32_t c = 0; c < n; c++)
		{
			float maxCost = -FLT_MAX;
			uint32_t bestSlot = INVALID_ASSIGNMENT;

			AABB childBounds = buildState.bvh2Nodes[childNodes[c]].bounds;
			float3 childCentroid = childBounds.bMax + childBounds.bMin;
			float3 offset = parentCentroid - childCentroid;

			for (uint32_t s = 0; s < 8; s++)
			{
				// If slot already assigned, skip
				if (GetNibble(assignments, s) != INVALID_ASSIGNMENT)
					continue;

				float cost = GetCost(s, offset);
				if (cost > maxCost)
				{
					maxCost = cost;
					bestSlot = s;
				}
			}
			SetNibble(assignments, bestSlot, c);

			//uint32_t slotsMask = slotsAvailable;
			//while(true)
			//{
			//	int32_t s = 31 - __clz(slotsMask);
			//	if (s < 0)
			//		break;

			//	slotsMask &= ~(1 << s);

			//	float cost = GetCost(c, s, offsets);
			//	if (cost > maxCost)
			//	{
			//		maxCost = cost;
			//		bestSlot = s;
			//	}
			//}
			//slotsAvailable &= ~(1 << bestSlot);
		}
	}


	__device__ __forceinline__ BVH8::Node CreateBVH8Node(
		AABB bounds, uint32_t childNodes[8], uint32_t childBaseIdx, uint32_t primBaseIdx,
		uint32_t assignments, uint32_t innerMask, uint32_t leafMask, BVH8BuildState buildState)
	{
		BVH8::NodeExplicit bvh8Node;

		float3 diagonal = bounds.bMax - bounds.bMin;
		bvh8Node.p = bounds.bMin;
		bvh8Node.e[0] = CeilLog2(diagonal.x * quantStep);
		bvh8Node.e[1] = CeilLog2(diagonal.y * quantStep);
		bvh8Node.e[2] = CeilLog2(diagonal.z * quantStep);
		bvh8Node.imask = 0;

		bvh8Node.childBaseIdx = childBaseIdx;
		bvh8Node.primBaseIdx = primBaseIdx;

		float3 invE = make_float3(InvPow2(bvh8Node.e[0]), InvPow2(bvh8Node.e[1]), InvPow2(bvh8Node.e[2]));

		for (uint32_t i = 0; i < 8; i++)
		{
			bvh8Node.meta[i] = 0;
			uint32_t assignment = GetNibble(assignments, i);

			if (innerMask & (1 << i))
			{
				// Inner node
				bvh8Node.imask |= 1 << i;
				bvh8Node.meta[i] |= 1 << 5;
				bvh8Node.meta[i] |= 24 + i;
			}
			else if (assignment != INVALID_ASSIGNMENT)
			{
				//  Leaf node
				bvh8Node.meta[i] |= (1 << 5) | CountBitsBelow(leafMask, i);
			}
			else
				continue;

			AABB childBounds = buildState.bvh2Nodes[childNodes[assignment]].bounds;
			bvh8Node.qlox[i] = (uint8_t)floorf((childBounds.bMin.x - bounds.bMin.x) * invE.x);
			bvh8Node.qloy[i] = (uint8_t)floorf((childBounds.bMin.y - bounds.bMin.y) * invE.y);
			bvh8Node.qloz[i] = (uint8_t)floorf((childBounds.bMin.z - bounds.bMin.z) * invE.z);

			bvh8Node.qhix[i] = (uint8_t)ceilf((childBounds.bMax.x - bounds.bMin.x) * invE.x);
			bvh8Node.qhiy[i] = (uint8_t)ceilf((childBounds.bMax.y - bounds.bMin.y) * invE.y);
			bvh8Node.qhiz[i] = (uint8_t)ceilf((childBounds.bMax.z - bounds.bMin.z) * invE.z);
		}

		BVH8::Node node = *(BVH8::Node*)&bvh8Node;
		return node;
	}


	__device__ inline void CreateBVH8SingleLeaf(uint32_t workId, BVH8BuildState buildState)
	{
		if (workId == 0)
		{
			uint32_t childNodes[8];
			childNodes[0] = 0;
			uint32_t assignments = 0xfffffff0;
			BVH2::Node bvh2Node = buildState.bvh2Nodes[0];
			atomicAdd(buildState.leafCounter, 1);
			buildState.primIdx[0] = 0;

			buildState.bvh8Nodes[0] = CreateBVH8Node(bvh2Node.bounds, childNodes, 0, 0, assignments, 0x0, 0x1, buildState);
		}
	}


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

		// If the BVH2 only consists of a single leaf node, the algorithm doesn't work.
		// I did not find a better way to handle this frustrating case
		if (buildState.primCount == 1)
		{
			CreateBVH8SingleLeaf(workId, buildState);
			return;
		}

		// The kernel is launched with enough threads to process primCount work items (as in the paper)
		// I've tried another approach with just enough threads to fill the GPU and fetching new work dynamically,
		// but this results in longer build times likely because threads in a warp finish at about the same time and
		// cache coherency is better without it
		bool laneActive = workId < buildState.primCount;

		while (true)
		{
			// Synchronization to prevent threads that have not yet been assigned work from looping indefinitely and stealing the work
			// of active threads in a block. I'm not sure that this is flawless since inactive blocks could still starve the GPU and prevent
			// active ones from doing useful work. Maybe using dynamic parallelism would increase performance.
			// If no active threads remaining in the block, exit
			if (__syncthreads_count(laneActive) == 0)
				break;

			// If work is done, skip
			if (!laneActive)
				continue;

			// We don't want to load index pairs from L1 cache, since we want the updated version in global memory
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
				laneActive = false;
				continue;
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

			float3 parentCentroid = (bvh2Node.bounds.bMin + bvh2Node.bounds.bMax);

			// Reorder the child nodes
			uint32_t assignments;// = 0x76543210 | (0xffffffff << (childCount * 4));

#ifdef USE_AUCTION
			// We  want to keep the cost at the same order of magnitude, regardless of the dimensions
			// This ensures that the auction algorithm performs consistently across iterations
			float maxDim = fmaxf(bvh2Node.bounds.bMax - bvh2Node.bounds.bMin);
			float costScale = maxDim == 0.0f ? 0.0f : MAX_COST / maxDim;
			costScale *= 0.5;

			// Fill the table cost(c, s)
			float maxCost = -FLT_MAX;
			float3 offsets[8];

			for (uint32_t c = 0; c < childCount; c++)
			{
				AABB childBounds = bvh2Nodes[childNodes[c]].bounds;
				float3 centroid = (childBounds.bMin + childBounds.bMax);

				// Since auction is a maximization algorithm, the cost function has an opposite sign to that of the BVH8 paper
				offsets[c] = (parentCentroid - centroid) * costScale;

				float cost = fabs(offsets[c].x) + fabs(offsets[c].y) + fabs(offsets[c].z);
				if (cost > maxCost)
					maxCost = cost;
			}

			AuctionAssignment(offsets, maxCost, childCount, assignments);

#else
			GreedyAssignment(parentCentroid, childNodes, childCount, assignments, buildState);
#endif

			// Compute the new masks after reordering
			uint32_t newInnerMask = 0;
			uint32_t leafMask = 0;
			for (uint32_t i = 0; i < 8; i++)
			{
				if (GetNibble(assignments, i) == INVALID_ASSIGNMENT)
					continue;

				bool bit = (innerMask >> GetNibble(assignments, i)) & 1;
				newInnerMask |= (uint32_t)bit << i;
				leafMask |= (uint32_t)(!bit) << i;
			}
			innerMask = newInnerMask;


			uint32_t innerCount = __popc(innerMask);
			uint32_t leafCount = childCount - innerCount;

			// Allocate new inner nodes, leaf nodes and work items
			uint32_t childBaseIdx = atomicAdd(buildState.nodeCounter, innerCount);
			uint32_t workBaseIdx = atomicAdd(buildState.workAllocCounter, childCount - 1);

			uint32_t primBaseIdx = 0;
			if (leafCount > 0)
				primBaseIdx = atomicAdd(buildState.leafCounter, leafCount);

			// Add new work in the index pair list
			for (uint32_t i = 0; i < 8; i++)
			{
				if (GetNibble(assignments, i) == INVALID_ASSIGNMENT)
					continue;

				uint64_t pair = (uint64_t)childNodes[GetNibble(assignments, i)] << 32;
				if (innerMask & (1 << i))
					pair |= childBaseIdx + CountBitsBelow(innerMask, i);
				else
					pair |= primBaseIdx + CountBitsBelow(leafMask, i);


				uint32_t c = CountBitsBelow(innerMask | leafMask, i);
				uint32_t idx = c == 0 ? workId : workBaseIdx + c - 1;
				GlobalStore(&buildState.indexPairs[idx], pair);
			}

			// Create and store the new BVH8 node
			bvh8Nodes[bvh8NodeIdx] = CreateBVH8Node(bvh2Node.bounds, childNodes, childBaseIdx, primBaseIdx, assignments, innerMask, leafMask, buildState);
		}
	}
}
