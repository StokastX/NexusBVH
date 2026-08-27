#pragma once

#include <cstdint>
#include <vector>

#include "NXB/AABB.h"
#include "NXB/Triangle.h"

namespace NXB::Test
{
	/*
	 * Deterministic scene generators.
	 *
	 * Every generator is seeded with a fixed value, so a case that fails at 200k
	 * primitives fails the same way on the next run and can be shrunk by hand.
	 */

	// Scatters small triangles through a gridSize^3 grid of cells, one triangle per
	// draw. Clustered rather than uniform noise, which is closer to what a real mesh
	// looks like to the Morton sort than uniformly random vertices would be.
	std::vector<Triangle> GenerateTriangles(uint32_t primCount, uint32_t gridSize);

	// The same scene handed to the builder as bounding boxes instead of triangles,
	// to exercise the AABB instantiation of the build templates
	std::vector<AABB> GenerateAABBs(uint32_t primCount, uint32_t gridSize);
}
