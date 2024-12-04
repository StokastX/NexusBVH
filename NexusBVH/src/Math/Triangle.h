#pragma once

#include <cuda_runtime.h>

#include "CudaMath.h"
#include "AABB.h"

namespace NXB
{
	struct Triangle
	{
		__host__ __device__ Triangle() = default;

		__host__ __device__ Triangle(float3 pos0, float3 pos1, float3 pos2)
			: v0(pos0), v1(pos1), v2(pos2) { }

		__host__ __device__ float3 Centroid()
		{
			return (v0 + v1 + v2) * 0.5;
		}

		__host__ __device__ AABB Bounds()
		{
			return AABB(v0, v1, v2);
		}

		float3 v0, v1, v2;
	};
}