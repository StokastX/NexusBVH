#pragma once

#include <cstdint>
#include <vector>

#include "NXB/AABB.h"

namespace NXB::Test
{
	/*
	 * A ray and the hit it produced.
	 *
	 * The library owns neither type on purpose -- traversal takes an origin, a reciprocal
	 * direction and a callback -- so the suite carries its own, exactly the way a consumer
	 * has to. Nothing here is passed into NXB.
	 */
	struct Ray
	{
		float3 origin;
		float3 direction;
		float3 invDirection;
	};

	struct Hit
	{
		// NXB::RayMiss and NXB::InvalidIdx when nothing was hit
		float t;
		uint32_t primIdx;
	};

	// No epsilon on the reciprocal: a zero direction component is meant to produce an
	// infinity, because that is what a caller's own code will hand the slab test.
	Ray MakeRay(float3 origin, float3 direction);

	bool operator==(const Hit& a, const Hit& b);

	/*
	 * Deterministic ray sets, fixed seed like the scene generators, so a failure at a
	 * given ray count reproduces exactly and can be shrunk by hand.
	 */

	// Origins on a sphere around the scene, aimed at random points inside it. The bulk of
	// the coverage: most of these hit something, so they exercise the pruning path.
	std::vector<Ray> GenerateRays(const AABB& sceneBounds, uint32_t count);

	// Origins inside the scene, random directions. Reaches the case where the ray starts
	// inside a node box, which reports tMin rather than a positive entry distance.
	std::vector<Ray> GenerateInteriorRays(const AABB& sceneBounds, uint32_t count);

	// Axis aligned rays: two direction components are exactly zero, so invDirection holds
	// infinities and the slab test runs its 0 * inf path. Returns 6 * perAxis rays.
	std::vector<Ray> GenerateAxisAlignedRays(const AABB& sceneBounds, uint32_t perAxis);

	// Rays pointing away from the scene, which must hit nothing at all
	std::vector<Ray> GenerateMissRays(const AABB& sceneBounds, uint32_t count);
}
