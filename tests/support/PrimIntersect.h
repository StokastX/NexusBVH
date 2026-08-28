#pragma once

#include "NXB/AABB.h"
#include "NXB/BVHTraversal.h"
#include "NXB/Triangle.h"

#include "Rays.h"

namespace NXB::Test
{
	/*
	 * The primitive tests, __host__ __device__ so that the host walk, the device kernel
	 * and brute force all run the same instruction sequence.
	 *
	 * That matters for the device case specifically: if the kernel had its own copy, a
	 * disagreement between host and device could be either traversal or two intersectors
	 * that drifted apart, and telling those apart afterwards is miserable.
	 */

	/*
	 * Moller-Trumbore, two sided. The epsilon rejects rays parallel to the triangle
	 * plane: a generated scene contains near degenerate triangles, and those would
	 * otherwise produce a NaN distance that compares unequal to itself.
	 */
	__host__ __device__ inline bool IntersectPrim(const Triangle& tri, const Ray& ray,
		float tMin, float tMax, float& t)
	{
		const float3 e1 = make_float3(tri.v1.x - tri.v0.x, tri.v1.y - tri.v0.y, tri.v1.z - tri.v0.z);
		const float3 e2 = make_float3(tri.v2.x - tri.v0.x, tri.v2.y - tri.v0.y, tri.v2.z - tri.v0.z);

		const float3 p = make_float3(
			ray.direction.y * e2.z - ray.direction.z * e2.y,
			ray.direction.z * e2.x - ray.direction.x * e2.z,
			ray.direction.x * e2.y - ray.direction.y * e2.x);

		const float det = e1.x * p.x + e1.y * p.y + e1.z * p.z;
		if (fabsf(det) < 1e-12f)
			return false;

		const float invDet = 1.0f / det;
		const float3 s = make_float3(
			ray.origin.x - tri.v0.x, ray.origin.y - tri.v0.y, ray.origin.z - tri.v0.z);

		const float u = (s.x * p.x + s.y * p.y + s.z * p.z) * invDet;
		if (u < 0.0f || u > 1.0f)
			return false;

		const float3 q = make_float3(
			s.y * e1.z - s.z * e1.y, s.z * e1.x - s.x * e1.z, s.x * e1.y - s.y * e1.x);

		const float v = (ray.direction.x * q.x + ray.direction.y * q.y + ray.direction.z * q.z) * invDet;
		if (v < 0.0f || u + v > 1.0f)
			return false;

		const float hit = (e2.x * q.x + e2.y * q.y + e2.z * q.z) * invDet;
		if (hit < tMin || hit > tMax)
			return false;

		t = hit;
		return true;
	}

	// An AABB primitive is hit exactly where the slab test says a node box is, so the
	// descent and the hit are decided by the same code
	__host__ __device__ inline bool IntersectPrim(const AABB& box, const Ray& ray,
		float tMin, float tMax, float& t)
	{
		t = IntersectAABB(box, ray.origin, ray.invDirection, tMin, tMax);
		return t < RayMiss;
	}

	// Closest hit by testing every primitive. Shared so the device kernel can run the
	// same reference the host does.
	template <typename PrimT>
	__host__ __device__ inline Hit BruteForce(const PrimT* prims, uint32_t primCount,
		const Ray& ray, float tMin, float tMax)
	{
		Hit hit{ RayMiss, InvalidIdx };
		for (uint32_t i = 0; i < primCount; ++i)
		{
			float t;
			if (IntersectPrim(prims[i], ray, tMin, tMax, t) && t < hit.t)
			{
				hit.t = t;
				hit.primIdx = i;
			}
		}
		return hit;
	}
}
