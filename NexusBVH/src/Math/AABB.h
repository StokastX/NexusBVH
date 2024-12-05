#pragma once

#include "Math/CudaMath.h"

namespace NXB
{
	struct AABB
	{
		__host__ __device__ AABB() = default;
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

		__host__ __device__ void Clear()
		{
			bMin = make_float3(INFINITY);
			bMax = make_float3(-INFINITY);
		}

		__host__ __device__ float3 Centroid()
		{
			return (bMin + bMax) * 0.5f;
		}

		float3 bMin;
		float3 bMax;
	};
}