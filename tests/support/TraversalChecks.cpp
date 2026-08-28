#include "TraversalChecks.h"

#include <string>
#include <utility>

#include "NXB/BVHTraversal.h"

namespace NXB::Test
{
	namespace
	{
		std::string Str(float v) { return std::to_string(v); }
		std::string Str(uint32_t v) { return std::to_string(v); }

		/*
		 * Moller-Trumbore, two sided. The epsilon rejects rays parallel to the triangle
		 * plane: a generated scene contains near degenerate triangles, and those would
		 * otherwise produce a NaN distance that compares unequal to itself.
		 */
		bool IntersectTriangle(const Triangle& tri, const Ray& ray, float tMin, float tMax, float& t)
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

		template <typename PrimT>
		Hit BruteForce(const std::vector<PrimT>& prims, const Ray& ray, float tMin, float tMax)
		{
			Hit hit{ RayMiss, InvalidIdx };
			for (uint32_t i = 0; i < (uint32_t)prims.size(); ++i)
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

		template <typename PrimT>
		ValidationResult Validate(const BVH2::Host& bvh, const std::vector<PrimT>& prims,
			const std::vector<Ray>& rays, float tMin)
		{
			ValidationResult result;
			const uint32_t primCount = (uint32_t)prims.size();

			size_t traversalTests = 0;
			size_t bruteForceTests = 0;

			// Stop collecting once the output would be noise; the first failures are the
			// ones worth reading
			for (size_t r = 0; r < rays.size() && result.errors.size() < 16; ++r)
			{
				const Ray& ray = rays[r];
				const std::string at = " (ray " + std::to_string(r) + ")";

				Hit hit{ RayMiss, InvalidIdx };
				bool indexOutOfRange = false;
				float tMax = RayMiss;

				bool complete = TraverseBVH2(bvh, ray.origin, ray.invDirection, tMin, tMax,
					[&](uint32_t primIdx, float& tMax)
					{
						++traversalTests;
						if (primIdx >= primCount)
						{
							indexOutOfRange = true;
							result.Add("traversal reported primitive " + Str(primIdx) +
								" outside [0, " + Str(primCount) + ")" + at);
							return false;
						}

						float t;
						if (IntersectPrim(prims[primIdx], ray, tMin, tMax, t))
						{
							tMax = t;
							hit.t = t;
							hit.primIdx = primIdx;
						}
						return true;
					});

				if (!complete && !indexOutOfRange)
					result.Add("closest hit traversal reported an early out it never asked for" + at);

				bruteForceTests += primCount;
				const Hit reference = BruteForce(prims, ray, tMin, RayMiss);

				// A tie on distance is legitimate: two primitives can meet at a point and
				// either is a correct answer. A different distance never is.
				if (hit.t != reference.t)
					result.Add("traversal found t=" + Str(hit.t) + " prim=" + Str(hit.primIdx) +
						", brute force found t=" + Str(reference.t) + " prim=" +
						Str(reference.primIdx) + at);

				if (hit.primIdx != InvalidIdx && hit.t < tMin)
					result.Add("traversal reported t=" + Str(hit.t) + " below tMin" + at);

				// Any hit: stopping at the first intersecting leaf has to agree with brute
				// force on whether anything is there, and has to report that it stopped
				bool found = false;
				float anyTMax = RayMiss;
				bool ranToEnd = TraverseBVH2(bvh, ray.origin, ray.invDirection, tMin, anyTMax,
					[&](uint32_t primIdx, float& tMax)
					{
						float t;
						if (IntersectPrim(prims[primIdx], ray, tMin, tMax, t))
							found = true;
						return !found;
					});

				const bool referenceHit = reference.primIdx != InvalidIdx;
				if (found != referenceHit)
					result.Add(std::string("any hit says ") + (found ? "hit" : "miss") +
						" where brute force says " + (referenceHit ? "hit" : "miss") + at);

				if (found == ranToEnd)
					result.Add("any hit early out flag disagrees with whether it stopped" + at);
			}

			// A traversal that has stopped culling passes every check above
			if (result.Ok() && primCount > 64 && traversalTests * 4 > bruteForceTests)
				result.Add("traversal tested " + std::to_string(traversalTests) +
					" primitives against brute force's " + std::to_string(bruteForceTests) +
					", so it is barely culling");

			return result;
		}
	}

	bool IntersectPrim(const Triangle& tri, const Ray& ray, float tMin, float tMax, float& t)
	{
		return IntersectTriangle(tri, ray, tMin, tMax, t);
	}

	bool IntersectPrim(const AABB& box, const Ray& ray, float tMin, float tMax, float& t)
	{
		t = IntersectAABB(box, ray.origin, ray.invDirection, tMin, tMax);
		return t < RayMiss;
	}

	Hit BruteForceClosestHit(const std::vector<Triangle>& prims, const Ray& ray, float tMin, float tMax)
	{
		return BruteForce(prims, ray, tMin, tMax);
	}

	Hit BruteForceClosestHit(const std::vector<AABB>& prims, const Ray& ray, float tMin, float tMax)
	{
		return BruteForce(prims, ray, tMin, tMax);
	}

	ValidationResult ValidateTraversal(const BVH2::Host& bvh, const std::vector<Triangle>& prims,
		const std::vector<Ray>& rays, float tMin)
	{
		return Validate(bvh, prims, rays, tMin);
	}

	ValidationResult ValidateTraversal(const BVH2::Host& bvh, const std::vector<AABB>& prims,
		const std::vector<Ray>& rays, float tMin)
	{
		return Validate(bvh, prims, rays, tMin);
	}

	uint32_t MaxDepth(const BVH2::Host& bvh)
	{
		if (bvh.nodes.empty())
			return 0;

		uint32_t deepest = 0;
		std::vector<std::pair<uint32_t, uint32_t>> stack{ { bvh.RootIdx(), 1u } };

		while (!stack.empty())
		{
			const std::pair<uint32_t, uint32_t> item = stack.back();
			stack.pop_back();

			if (item.second > deepest)
				deepest = item.second;

			const BVH2::Node& node = bvh.nodes[item.first];
			if (node.leftChild == InvalidIdx)
				continue;

			stack.push_back({ node.leftChild, item.second + 1 });
			stack.push_back({ node.rightChild, item.second + 1 });
		}
		return deepest;
	}
}
