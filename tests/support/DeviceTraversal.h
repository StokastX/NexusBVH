#pragma once

#include <vector>

#include "NXB/AABB.h"
#include "NXB/BVH.h"
#include "NXB/Triangle.h"

#include "Rays.h"

namespace NXB::Test
{
	/*
	 * The same TraverseBVH2 template, compiled by nvcc and run in a kernel.
	 *
	 * The host walk proves the algorithm; this proves the header survives the device
	 * compiler -- a stack array that spills, a lambda nvcc declines to inline, an
	 * intrinsic that behaves differently. None of that is visible from a host build,
	 * and all of it is what a consumer will actually hit.
	 *
	 * The declarations are plain C++ so a .cpp test case can call them. Only the .cu is
	 * compiled by nvcc.
	 */

	// One ray per thread, closest hit
	std::vector<Hit> DeviceClosestHits(const BVH2::DeviceView& bvh,
		const Triangle* devicePrims, const std::vector<Ray>& rays);
	std::vector<Hit> DeviceClosestHits(const BVH2::DeviceView& bvh,
		const AABB* devicePrims, const std::vector<Ray>& rays);

	// Terminating at the first intersecting leaf, returning the early out flag rather
	// than the hit, so the callback's false return is what is under test
	std::vector<uint8_t> DeviceAnyHits(const BVH2::DeviceView& bvh,
		const Triangle* devicePrims, const std::vector<Ray>& rays);

	// Brute force over the same primitives, on the device, from the same shared
	// intersector. Gives the kernel an oracle that never touches the BVH.
	std::vector<Hit> DeviceBruteForceHits(const Triangle* devicePrims, uint32_t primCount,
		const std::vector<Ray>& rays);
}
