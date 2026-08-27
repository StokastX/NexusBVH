#include "vendor/doctest.h"

#include <cmath>
#include <vector>

#include "NXB/BVHBuilder.h"

#include "TestChecks.h"
#include "support/BVHChecks.h"
#include "support/Scenes.h"
#include "support/TestConfig.h"

using namespace NXB::Test;

namespace
{
	/*
	 * The BVH8 is checked more loosely than the BVH2: there is no ToHost(BVH8) and the
	 * node encoding is not readable from host code, so what can be reached from here is
	 * the node count, the leaf index list and the scene bounds.
	 *
	 * Nothing below verifies that a node's quantized child bounds actually contain their
	 * children, or that meta/e/p decode correctly. Closing that gap needs rays cast
	 * against the tree and compared with brute force -- which is what the traversal tests
	 * will do once traversal is part of the public API.
	 */
	template <typename PrimT>
	void BuildAndValidateBVH8(const std::vector<PrimT>& prims, NXB::BuildConfig buildConfig,
		NXB::BVHBuildMetrics* buildMetrics)
	{
		const uint32_t primCount = (uint32_t)prims.size();
		NXB::DeviceBuffer<PrimT> devicePrims(prims);

		NXB::BVH8 bvh = NXB::BuildBVH8<PrimT>(devicePrims.Get(), primCount, buildConfig, buildMetrics);

		// Node counts are exact, not conservative: the allocation is sized at (4n - 1) / 7
		// rounded up, and overrunning it would corrupt memory rather than fail loudly
		const uint32_t maxNodeCount = (4 * primCount - 1 + 6) / 7;
		CHECK(bvh.nodeCount > 0);
		CHECK(bvh.nodeCount <= maxNodeCount);
		CHECK(bvh.primCount == primCount);
		CHECK(bvh.nodes != nullptr);
		CHECK(bvh.primIdx != nullptr);

		CheckValid(ValidateSceneBounds(bvh.bounds, ReferenceSceneBounds(prims)));

		// Every primitive has to appear exactly once in the leaf index list
		CheckValid(ValidatePrimIdxPermutation(NXB::CopyToHost(bvh.primIdx, primCount), primCount));

		NXB::FreeDeviceBVH(bvh);
	}

	template <typename PrimT>
	void BuildAndValidateBVH8(const std::vector<PrimT>& prims, NXB::BuildConfig buildConfig)
	{
		BuildAndValidateBVH8(prims, buildConfig, nullptr);
	}
}


TEST_SUITE("fast")
{

TEST_CASE("BVH8 over tiny primitive counts")
{
	NXB::BuildConfig buildConfig;
	buildConfig.prioritizeSpeed = true;

	for (uint32_t primCount : { 1u, 2u, 3u, 7u, 64u, 1000u })
	{
		CAPTURE(primCount);
		BuildAndValidateBVH8(GenerateTriangles(primCount, smallSceneGridSize), buildConfig);
	}
}

} // TEST_SUITE fast


TEST_SUITE("slow")
{

TEST_CASE("BVH8 over a large scene")
{
	NXB::BuildConfig buildConfig;
	buildConfig.prioritizeSpeed = true;

	CAPTURE(largeScenePrimCount);

	SUBCASE("triangles")
	{
		BuildAndValidateBVH8(GenerateTriangles(largeScenePrimCount, largeSceneGridSize), buildConfig);
	}
	SUBCASE("AABBs")
	{
		BuildAndValidateBVH8(GenerateAABBs(largeScenePrimCount, largeSceneGridSize), buildConfig);
	}
}

// Requesting metrics also switches on the SAH cost kernels, so this is the only place
// the BVH8 gets any quality check at all
TEST_CASE("BVH8 collapse reduces the SAH cost")
{
	NXB::BuildConfig buildConfig;
	buildConfig.prioritizeSpeed = true;

	NXB::BVHBuildMetrics metrics = {};
	SUBCASE("triangles")
	{
		BuildAndValidateBVH8(GenerateTriangles(largeScenePrimCount, largeSceneGridSize), buildConfig, &metrics);
	}
	SUBCASE("AABBs")
	{
		BuildAndValidateBVH8(GenerateAABBs(largeScenePrimCount, largeSceneGridSize), buildConfig, &metrics);
	}

	REQUIRE(std::isfinite(metrics.bvh2Cost));
	REQUIRE(std::isfinite(metrics.bvh8Cost));
	CHECK(metrics.bvh2Cost > 0.0f);
	CHECK(metrics.bvh8Cost > 0.0f);

	// A quality expectation rather than a hard invariant. If a legitimate change trips
	// it, judge the change -- do not reflexively loosen the bound.
	CHECK(metrics.bvh8Cost <= metrics.bvh2Cost);

	CHECK(metrics.averageChildPerNode > 1.0f);
	CHECK(metrics.averageChildPerNode <= 8.0f);
}

} // TEST_SUITE slow
