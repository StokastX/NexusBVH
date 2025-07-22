#pragma once

#include <cuda_runtime.h>

#include "Math/CudaMath.h"
#include "AABB.h"

namespace NXB
{
	struct Triangle
	{
		Triangle() = default;

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

		// Normal (not normalized)
		__host__ __device__ float3 Normal() const
		{
			float3 edge0 = v1 - v0;
			float3 edge1 = v2 - v0;

			return cross(edge0, edge1);
		}

		// See https://community.khronos.org/t/how-can-i-find-the-area-of-a-3d-triangle/49777/2
		__host__ __device__ float Area() const
		{
			float3 edge0 = v1 - v0;
			float3 edge1 = v2 - v0;

			float3 normal = cross(edge0, edge1);

			return 0.5f * length(normal);
		}

		float3 v0, v1, v2;
	};
}