#include "vendor/doctest.h"

#include <string>
#include <vector>

#include "NXB/BVHBuilder.h"

#include "TestChecks.h"
#include "support/BVHChecks.h"
#include "support/DeviceBuffer.h"
#include "support/Scenes.h"
#include "support/TestConfig.h"

using namespace NXB::Test;

namespace
{
	template <typename PrimT>
	void BuildAndValidateBVH2(const std::vector<PrimT>& prims, NXB::BuildConfig buildConfig)
	{
		DeviceBuffer<PrimT> devicePrims(prims);

		NXB::BVH2 deviceBvh = NXB::BuildBVH2<PrimT>(devicePrims.Get(), (uint32_t)prims.size(), buildConfig);
		NXB::BVH2 hostBvh = NXB::ToHost(deviceBvh);

		CheckValid(ValidateBVH2(hostBvh));
		CheckValid(ValidateSceneBounds(hostBvh.bounds, ReferenceSceneBounds(prims)));

		NXB::FreeHostBVH(hostBvh);
		NXB::FreeDeviceBVH(deviceBvh);
	}
}


/*
 * Each case runs twice, once per Morton code width. The subcases only set the flag:
 * doctest re-runs the whole body once per subcase, so everything after them executes
 * in both runs.
 */

TEST_SUITE("fast")
{

TEST_CASE("BVH2 over tiny primitive counts")
{
	NXB::BuildConfig buildConfig;
	SUBCASE("32-bit morton codes") { buildConfig.prioritizeSpeed = true; }
	SUBCASE("64-bit morton codes") { buildConfig.prioritizeSpeed = false; }

	// These reach the single leaf and near trivial trees that a large scene never does
	for (uint32_t primCount : { 1u, 2u, 3u, 7u, 64u, 1000u })
	{
		CAPTURE(primCount);
		BuildAndValidateBVH2(GenerateTriangles(primCount, smallSceneGridSize), buildConfig);
	}
}

} // TEST_SUITE fast


TEST_SUITE("slow")
{

TEST_CASE("BVH2 over a large triangle scene")
{
	NXB::BuildConfig buildConfig;
	SUBCASE("32-bit morton codes") { buildConfig.prioritizeSpeed = true; }
	SUBCASE("64-bit morton codes") { buildConfig.prioritizeSpeed = false; }

	CAPTURE(largeScenePrimCount);
	BuildAndValidateBVH2(GenerateTriangles(largeScenePrimCount, largeSceneGridSize), buildConfig);
}

// The AABB instantiation of the build templates had never been exercised anywhere
// before this suite existed
TEST_CASE("BVH2 over a large AABB scene")
{
	NXB::BuildConfig buildConfig;
	SUBCASE("32-bit morton codes") { buildConfig.prioritizeSpeed = true; }
	SUBCASE("64-bit morton codes") { buildConfig.prioritizeSpeed = false; }

	CAPTURE(largeScenePrimCount);
	BuildAndValidateBVH2(GenerateAABBs(largeScenePrimCount, largeSceneGridSize), buildConfig);
}

} // TEST_SUITE slow
