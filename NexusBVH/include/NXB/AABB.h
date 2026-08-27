#pragma once

#include <float.h>
#include <math.h>

#include <cuda_runtime.h>

namespace NXB
{
	struct AABB
	{
		AABB() = default;
		__host__ __device__ AABB(float3 v0, float3 v1)
		{
			bMin = make_float3(fminf(v0.x, v1.x), fminf(v0.y, v1.y), fminf(v0.z, v1.z));
			bMax = make_float3(fmaxf(v0.x, v1.x), fmaxf(v0.y, v1.y), fmaxf(v0.z, v1.z));
		}
		__host__ __device__ AABB(float3 v0, float3 v1, float3 v2)
		{
			bMin = make_float3(fminf(v0.x, fminf(v1.x, v2.x)), fminf(v0.y, fminf(v1.y, v2.y)), fminf(v0.z, fminf(v1.z, v2.z)));
			bMax = make_float3(fmaxf(v0.x, fmaxf(v1.x, v2.x)), fmaxf(v0.y, fmaxf(v1.y, v2.y)), fmaxf(v0.z, fmaxf(v1.z, v2.z)));
		}

		__host__ __device__ void Grow(float3 v)
		{
			bMin = make_float3(fminf(bMin.x, v.x), fminf(bMin.y, v.y), fminf(bMin.z, v.z));
			bMax = make_float3(fmaxf(bMax.x, v.x), fmaxf(bMax.y, v.y), fmaxf(bMax.z, v.z));
		}

		__host__ __device__ void Grow(const AABB& other)
		{
			bMin = make_float3(fminf(bMin.x, other.bMin.x), fminf(bMin.y, other.bMin.y), fminf(bMin.z, other.bMin.z));
			bMax = make_float3(fmaxf(bMax.x, other.bMax.x), fmaxf(bMax.y, other.bMax.y), fmaxf(bMax.z, other.bMax.z));
		}

		__host__ __device__ void Clear()
		{
			bMin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
			bMax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		}

		__host__ __device__ float3 Centroid() const
		{
			return make_float3((bMin.x + bMax.x) * 0.5f, (bMin.y + bMax.y) * 0.5f, (bMin.z + bMax.z) * 0.5f);
		}

		// Returns area / 2
		__host__ __device__ float Area() const
		{
			float dx = bMax.x - bMin.x;
			float dy = bMax.y - bMin.y;
			float dz = bMax.z - bMin.z;
			return dx * dy + dy * dz + dz * dx;
		}

		float3 bMin;
		float3 bMax;
	};
}