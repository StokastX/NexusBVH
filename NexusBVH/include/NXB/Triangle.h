#pragma once

#include <math.h>

#include <cuda_runtime.h>

#include "AABB.h"

namespace NXB
{
	struct Triangle
	{
		Triangle() = default;

		__host__ __device__ Triangle(float3 pos0, float3 pos1, float3 pos2)
			: v0(pos0), v1(pos1), v2(pos2) { }

		__host__ __device__ float3 Centroid() const
		{
			return make_float3((v0.x + v1.x + v2.x) / 3.0f, (v0.y + v1.y + v2.y) / 3.0f, (v0.z + v1.z + v2.z) / 3.0f);
		}

		__host__ __device__ AABB Bounds() const
		{
			return AABB(v0, v1, v2);
		}

		// Normal (not normalized)
		__host__ __device__ float3 Normal() const
		{
			float3 edge0 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
			float3 edge1 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);

			return make_float3(
				edge0.y * edge1.z - edge0.z * edge1.y,
				edge0.z * edge1.x - edge0.x * edge1.z,
				edge0.x * edge1.y - edge0.y * edge1.x);
		}

		// See https://community.khronos.org/t/how-can-i-find-the-area-of-a-3d-triangle/49777/2
		__host__ __device__ float Area() const
		{
			float3 normal = Normal();

			return 0.5f * sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
		}

		float3 v0, v1, v2;
	};
}