#pragma once

#include "Math/CudaMath.h"

namespace NXB
{
	struct AABB
	{
		AABB() = default;
		__host__ __device__ AABB(float3 v0, float3 v1)
		{
			bMin = fminf(v0, v1);
			bMax = fmaxf(v0, v1);
		}
		__host__ __device__ AABB(float3 v0, float3 v1, float3 v2)
		{
			bMin = fminf(v0, fminf(v1, v2));
			bMax = fmaxf(v0, fmaxf(v1, v2));
		}

		__host__ __device__ void Grow(float3 v)
		{
			bMin = fminf(bMin, v);
			bMax = fmaxf(bMax, v);
		}

		__host__ __device__ void Grow(const AABB& other)
		{
			bMin = fminf(bMin, other.bMin);
			bMax = fmaxf(bMax, other.bMax);
		}

		__host__ __device__ void Clear()
		{
			bMin = make_float3(FLT_MAX);
			bMax = make_float3(-FLT_MAX);
		}

		__host__ __device__ float3 Centroid()
		{
			return (bMin + bMax) * 0.5f;
		}

		// Returns area / 2
		__host__ __device__ float Area()
		{
			float3 diff = bMax - bMin;
			return diff.x * diff.y + diff.y * diff.z + diff.z * diff.x;
		}

		float3 bMin;
		float3 bMax;
	};
}