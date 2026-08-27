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
	 * The counts and the scene bounds are checked here; the hierarchy itself is checked by
	 * ValidateBVH8, which walks it through the independent decoder in BVH8Decode.h rather
	 * than through the builder's own helpers.
	 *
	 * What none of it covers is traversal ORDER. This builder's slot convention shows up
	 * in the order a ray visits children and nowhere in the stored data, so closing that
	 * gap needs rays cast against the tree and compared with brute force -- which is what
	 * the traversal tests will do once traversal is part of the public API.
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
		CHECK(bvh.NodeCount() > 0);
		CHECK(bvh.NodeCount() <= maxNodeCount);
		CHECK(bvh.PrimCount() == primCount);
		CHECK(!bvh.Empty());

		CheckValid(ValidateSceneBounds(bvh.Bounds(), ReferenceSceneBounds(prims)));

		NXB::BVH8::Host hostBvh = bvh.ToHost();

		// Every primitive has to appear exactly once in the leaf index list
		CheckValid(ValidatePrimIdxPermutation(
			hostBvh.primIdx, primCount));

		CheckValid(ValidateBVH8(hostBvh, PrimBounds(prims)));

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
