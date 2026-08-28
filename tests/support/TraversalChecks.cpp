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
				const Hit reference = BruteForce(prims.data(), primCount, ray, tMin, RayMiss);

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

	Hit BruteForceClosestHit(const std::vector<Triangle>& prims, const Ray& ray, float tMin, float tMax)
	{
		return BruteForce(prims.data(), (uint32_t)prims.size(), ray, tMin, tMax);
	}

	Hit BruteForceClosestHit(const std::vector<AABB>& prims, const Ray& ray, float tMin, float tMax)
	{
		return BruteForce(prims.data(), (uint32_t)prims.size(), ray, tMin, tMax);
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
