#pragma once
#include "Math/CudaMath.h"

namespace NXB
{
	struct AABB
	{
		AABB() = default;
		AABB(float3 v0, float3 v1)
		{
			bMin = fminf(v0, v1);
			bMax = fmaxf(v0, v1);
		}
		AABB(float3 v0, float3 v1, float3 v2)
		{
			bMin = fminf(v0, fminf(v1, v2));
			bMax = fmaxf(v0, fmaxf(v1, v2));
		}

		__device__ void Clear()
		{
			bMin = make_float3(-INFINITY);
			bMax = make_float3(INFINITY);
		}

		__device__ float3 Centroid()
		{
			return (bMin + bMax) * 0.5f;
		}

		float3 bMin;
		float3 bMax;
	};

	__device__ void AtomicGrow(AABB* aabb, const AABB& other)
	{
		atomicMin(&aabb->bMin.x, other.bMin.x);
		atomicMin(&aabb->bMin.y, other.bMin.y);
		atomicMin(&aabb->bMin.z, other.bMin.z);

		atomicMax(&aabb->bMax.x, other.bMax.x);
		atomicMax(&aabb->bMax.y, other.bMax.y);
		atomicMax(&aabb->bMax.z, other.bMax.z);
	}
}