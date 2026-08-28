#pragma once

#include <vector>

#include "NXB/AABB.h"
#include "NXB/BVH.h"
#include "NXB/Triangle.h"

#include "BVHChecks.h"
#include "PrimIntersect.h"
#include "Rays.h"

namespace NXB::Test
{
	/*
	 * Traversal is checked against brute force, the one oracle that shares nothing with
	 * the thing under test. The structural checks in BVHChecks pin what the tree looks
	 * like; none of them can see whether a ray finds the right primitive in it, because
	 * child order lives in the traversal and nowhere in the stored data.
	 *
	 * The primitive tests below are shared by both sides on purpose. Traversal and brute
	 * force reach the same primitive by different routes, so using one intersector for
	 * both makes a mismatch mean traversal, and never the triangle test.
	 */

	// IntersectPrim and BruteForce live in PrimIntersect.h, __host__ __device__ so the
	// device kernel in DeviceTraversal.cu runs the identical code

	// Closest hit found by testing every primitive
	Hit BruteForceClosestHit(const std::vector<Triangle>& prims, const Ray& ray, float tMin, float tMax);
	Hit BruteForceClosestHit(const std::vector<AABB>& prims, const Ray& ray, float tMin, float tMax);

	/*
	 * Runs every ray through TraverseBVH2 and through brute force and reports where the
	 * two disagree. Per ray:
	 *
	 *  - the closest hit matches, primitive and distance
	 *  - the any hit query agrees on whether anything was hit, and reports an early out
	 *    exactly when it stopped
	 *  - every primitive index the callback hands back is in range
	 *  - the reported distance lies within the ray interval
	 *
	 * And once over the whole set: that traversal tested materially fewer primitives than
	 * brute force did, which is what separates a working BVH from one walked exhaustively.
	 */
	ValidationResult ValidateTraversal(const BVH2::Host& bvh, const std::vector<Triangle>& prims,
		const std::vector<Ray>& rays, float tMin = 0.0f);
	ValidationResult ValidateTraversal(const BVH2::Host& bvh, const std::vector<AABB>& prims,
		const std::vector<Ray>& rays, float tMin = 0.0f);

	/*
	 * Depth of the deepest leaf, counting the root as depth 1.
	 *
	 * TraverseBVH2 sizes its stack from a template parameter defaulting to 32, and the
	 * builder bounds tree depth nowhere, so that default is an assumption about real
	 * scenes rather than a guarantee. This is what turns it into a measured claim.
	 */
	uint32_t MaxDepth(const BVH2::Host& bvh);
}
