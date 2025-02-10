#include "Eval.h"
#include <device_launch_parameters.h>

#include "NXB/BVH.h"
#include "BuilderUtils.h"

#define C_I 2 // Cost of a ray-primitive intersection
#define C_T 3 // Cost of a traversal step

namespace NXB
{
	__global__ void ComputeBVHCost(BVH2 bvh, float* cost)
	{
		uint32_t nodeIdx = blockDim.x * blockIdx.x + threadIdx.x;
		uint32_t laneId = threadIdx.x & (WARP_SIZE - 1);

		float sceneBoundsArea = bvh.bounds.Area();

		float area = 0.0f;

		if (nodeIdx < bvh.nodeCount)
		{
			BVH2::Node node = bvh.nodes[nodeIdx];
			float m = node.bounds.Area() / sceneBoundsArea;
			if (node.leftChild != INVALID_IDX)
				area = C_T * m;
			else
				area = C_I * m;
		}
		area = BlockReduceSum(area);

		if (threadIdx.x == 0)
			atomicAdd(cost, area);
	}
}
