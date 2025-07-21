#include "Eval.h"
#include <device_launch_parameters.h>

#include "NXB/BVH.h"
#include "BuilderUtils.h"

#define C_I 2 // Cost of a ray-primitive intersection
#define C_T 3 // Cost of a traversal step

namespace NXB
{
	__global__ void ComputeBVH2CostKernel(BVH2 bvh, float* cost)
	{
		uint32_t nodeIdx = blockDim.x * blockIdx.x + threadIdx.x;

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


	__global__ void ComputeBVH8CostKernel(BVH8 bvh, float* cost)
	{
		uint32_t nodeIdx = blockDim.x * blockIdx.x + threadIdx.x;

		float sceneBoundsArea = bvh.bounds.Area();

		float area = 0.0f;

		if (nodeIdx < bvh.nodeCount)
		{
			BVH8::NodeExplicit node = *(BVH8::NodeExplicit*)&bvh.nodes[nodeIdx];

			for (uint32_t i = 0; i < 8; i++)
			{
				if (!node.meta[i])
					continue;

				bool internal = (node.meta[i] & 0b11111) >= 24;

				float blox = node.p.x + __uint_as_float(node.e[0] << 23) * node.qlox[i];
				float bloy = node.p.y + __uint_as_float(node.e[1] << 23) * node.qloy[i];
				float bloz = node.p.z + __uint_as_float(node.e[2] << 23) * node.qloz[i];
				float3 blo = make_float3(blox, bloy, bloz);

				float bhix = node.p.x + __uint_as_float(node.e[0] << 23) * node.qhix[i];
				float bhiy = node.p.y + __uint_as_float(node.e[1] << 23) * node.qhiy[i];
				float bhiz = node.p.z + __uint_as_float(node.e[2] << 23) * node.qhiz[i];
				float3 bhi = make_float3(bhix, bhiy, bhiz);

				AABB childBounds;
				childBounds.bMin = blo;
				childBounds.bMax = bhi;

				float m = childBounds.Area() / sceneBoundsArea;
				if (internal)
					area += C_I * m;
				else
					area += C_T * m;
			}
		}
		area = BlockReduceSum(area);

		if (threadIdx.x == 0)
			atomicAdd(cost, area);
	}
}
