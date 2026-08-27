#include "vendor/doctest.h"

#include <cmath>
#include <vector>

#include "NXB/BVHBuilder.h"
#include "NXB/BVHCost.h"

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
	void BuildAndValidateBVH8(const std::vector<PrimT>& prims, NXB::BuildConfig buildConfig)
	{
		const uint32_t primCount = (uint32_t)prims.size();
		NXB::DeviceBuffer<PrimT> devicePrims(prims);

		NXB::BVH8 bvh = NXB::BuildBVH8<PrimT>(devicePrims.Get(), primCount, buildConfig);

		// Node counts are exact, not conservative: the allocation is sized at (4n - 1) / 7
		// rounded up, and overrunning it would corrupt memory rather than fail loudly
		const uint32_t maxNodeCount = (4 * primCount - 1 + 6) / 7;
		CHECK(bvh.NodeCount() > 0);
		CHECK(bvh.NodeCount() <= maxNodeCount);
		CHECK(bvh.PrimCount() == primCount);
		CHECK(!bvh.Empty());

		// Pure arithmetic on the two counts above, so every BVH8 case can afford it.
		// Exactly 1.0 is reachable: a one-primitive scene collapses to a single node
		// holding a single leaf.
		CHECK(bvh.AverageChildPerNode() >= 1.0f);
		CHECK(bvh.AverageChildPerNode() <= 8.0f);

		CheckValid(ValidateSceneBounds(bvh.Bounds(), ReferenceSceneBounds(prims)));

		NXB::BVH8::Host hostBvh = bvh.ToHost();

		// Every primitive has to appear exactly once in the leaf index list
		CheckValid(ValidatePrimIdxPermutation(
			hostBvh.primIdx, primCount));

		CheckValid(ValidateBVH8(hostBvh, PrimBounds(prims)));

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

/*
 * ComputeSAHCost is a call on a finished BVH rather than something a build reports, so
 * this builds both trees from the same primitives and compares them.
 *
 * The BVH2 here is not the one the collapse consumed -- BuildBVH8 builds its own scratch
 * BVH2 and releases it -- and the builder is not deterministic, so the two differ
 * slightly. The reduction the collapse buys is far larger than that difference, which is
 * what makes the comparison meaningful anyway.
 */
template <typename PrimT>
void CheckCollapseReducesCost(const std::vector<PrimT>& prims, NXB::BuildConfig buildConfig)
{
	const uint32_t primCount = (uint32_t)prims.size();
	NXB::DeviceBuffer<PrimT> devicePrims(prims);

	NXB::BVH2 bvh2 = NXB::BuildBVH2<PrimT>(devicePrims.Get(), primCount, buildConfig);
	NXB::BVH8 bvh8 = NXB::BuildBVH8<PrimT>(devicePrims.Get(), primCount, buildConfig);

	const float bvh2Cost = NXB::ComputeSAHCost(bvh2);
	const float bvh8Cost = NXB::ComputeSAHCost(bvh8);

	REQUIRE(std::isfinite(bvh2Cost));
	REQUIRE(std::isfinite(bvh8Cost));
	CHECK(bvh2Cost > 0.0f);
	CHECK(bvh8Cost > 0.0f);

	// A quality expectation rather than a hard invariant. If a legitimate change trips
	// it, judge the change -- do not reflexively loosen the bound.
	CHECK(bvh8Cost <= bvh2Cost);
}

TEST_CASE("BVH8 collapse reduces the SAH cost")
{
	NXB::BuildConfig buildConfig;
	buildConfig.prioritizeSpeed = true;

	SUBCASE("triangles")
	{
		CheckCollapseReducesCost(GenerateTriangles(largeScenePrimCount, largeSceneGridSize), buildConfig);
	}
	SUBCASE("AABBs")
	{
		CheckCollapseReducesCost(GenerateAABBs(largeScenePrimCount, largeSceneGridSize), buildConfig);
	}
}

} // TEST_SUITE slow
