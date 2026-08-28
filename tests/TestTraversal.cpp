#include "vendor/doctest.h"

#include <vector>

#include "NXB/BVHBuilder.h"
#include "NXB/BVHTraversal.h"

#include "TestChecks.h"
#include "support/BVHChecks.h"
#include "support/Rays.h"
#include "support/Scenes.h"
#include "support/TestConfig.h"
#include "support/TraversalChecks.h"

using namespace NXB::Test;

namespace
{
	// Brute force is O(rays * primitives), so the ray counts are what keeps these cases
	// inside their suite's time budget rather than the scenes
	constexpr uint32_t fastRayCount = 256;
	constexpr uint32_t slowRayCount = 64;

	template <typename PrimT>
	NXB::BVH2 Build(const std::vector<PrimT>& prims, NXB::BuildConfig buildConfig = {})
	{
		NXB::DeviceBuffer<PrimT> devicePrims(prims);
		return NXB::BuildBVH2<PrimT>(devicePrims.Get(), (uint32_t)prims.size(), buildConfig);
	}

	// Every generator over one scene, so a case covers the ordinary path and the
	// degenerate ones without repeating itself
	std::vector<Ray> AllRayKinds(const NXB::AABB& bounds, uint32_t count)
	{
		std::vector<Ray> rays = GenerateRays(bounds, count);

		for (const std::vector<Ray>& more : {
			GenerateInteriorRays(bounds, count / 4),
			GenerateAxisAlignedRays(bounds, count / 24),
			GenerateMissRays(bounds, count / 4) })
			rays.insert(rays.end(), more.begin(), more.end());

		return rays;
	}

	template <typename PrimT>
	void BuildAndValidateTraversal(const std::vector<PrimT>& prims, uint32_t rayCount,
		NXB::BuildConfig buildConfig = {})
	{
		NXB::BVH2 deviceBvh = Build(prims, buildConfig);
		NXB::BVH2::Host bvh = deviceBvh.ToHost();

		CheckValid(ValidateTraversal(bvh, prims, AllRayKinds(bvh.bounds, rayCount)));
	}
}


TEST_SUITE("fast")
{

TEST_CASE("BVH2 traversal matches brute force over tiny scenes")
{
	NXB::BuildConfig buildConfig;
	SUBCASE("32-bit morton codes") { buildConfig.prioritizeSpeed = true; }
	SUBCASE("64-bit morton codes") { buildConfig.prioritizeSpeed = false; }

	// The single leaf and near trivial trees a large scene never reaches. A one
	// primitive BVH puts a leaf at the root, which is its own path through the loop.
	for (uint32_t primCount : { 1u, 2u, 3u, 7u, 64u, 1000u })
	{
		CAPTURE(primCount);
		BuildAndValidateTraversal(GenerateTriangles(primCount, smallSceneGridSize), fastRayCount, buildConfig);
	}
}

TEST_CASE("BVH2 traversal over AABB primitives")
{
	// Leaf boxes and primitive boxes are the same box here, so the slab test decides
	// both the descent and the hit -- any disagreement is ordering, not geometry
	BuildAndValidateTraversal(GenerateAABBs(1000, smallSceneGridSize), fastRayCount);
}

TEST_CASE("Traversal of an empty BVH visits nothing")
{
	NXB::BVH2::Host empty;
	REQUIRE(empty.RootIdx() == NXB::InvalidIdx);

	uint32_t calls = 0;
	float tMax = NXB::RayMiss;
	bool complete = NXB::TraverseBVH2(empty, make_float3(0.0f, 0.0f, 0.0f),
		make_float3(1.0f, 1.0f, 1.0f), 0.0f, tMax,
		[&](uint32_t, float&) { calls++; return true; });

	CHECK(complete);
	CHECK(calls == 0);
	CHECK(tMax == NXB::RayMiss);
}

TEST_CASE("tMin and tMax clip the interval traversal reports")
{
	std::vector<NXB::Triangle> prims = GenerateTriangles(1000, smallSceneGridSize);
	NXB::BVH2 deviceBvh = Build(prims);
	NXB::BVH2::Host bvh = deviceBvh.ToHost();

	// Pushing tMin past a hit has to lose it, and the same rays with the full interval
	// have to find it again -- otherwise a clip that culls everything would pass
	std::vector<Ray> rays = GenerateRays(bvh.bounds, fastRayCount);
	CheckValid(ValidateTraversal(bvh, prims, rays, 0.0f));

	uint32_t hitsFromZero = 0;
	uint32_t hitsFromFar = 0;
	for (const Ray& ray : rays)
	{
		if (BruteForceClosestHit(prims, ray, 0.0f, NXB::RayMiss).primIdx != NXB::InvalidIdx)
			hitsFromZero++;

		float tMax = NXB::RayMiss;
		bool hit = !NXB::TraverseBVH2(bvh, ray.origin, ray.invDirection, 1e6f, tMax,
			[&](uint32_t primIdx, float& tMax)
			{
				float t;
				return !IntersectPrim(prims[primIdx], ray, 1e6f, tMax, t);
			});
		if (hit)
			hitsFromFar++;
	}

	CHECK(hitsFromZero > 0);
	CHECK(hitsFromFar == 0);
}

TEST_CASE("BVH2 depth fits the default traversal stack")
{
	// TraverseBVH2 defaults to a 32 entry stack and the builder bounds depth nowhere,
	// so the default is only as good as this measurement
	for (uint32_t primCount : { 1u, 1000u })
	{
		NXB::BVH2 deviceBvh = Build(GenerateTriangles(primCount, smallSceneGridSize));
		uint32_t depth = MaxDepth(deviceBvh.ToHost());

		CAPTURE(primCount);
		CAPTURE(depth);
		CHECK(depth >= 1);
		CHECK(depth <= 32);
	}
}

TEST_CASE("Near to far ordering culls what far to near would not")
{
	/*
	 * Correctness does not depend on child order -- the closest hit is the closest hit
	 * whichever way the tree is walked -- so every other case here passes with the order
	 * reversed. This is what actually pins the ordering.
	 *
	 * Measured on this scene and this ray set: near to far visits 7.4 leaves per ray,
	 * far to near visits 38.2. The budget sits between the two, so it fails long before
	 * a regression becomes subtle, and CAPTURE prints the real number when it does.
	 */
	constexpr uint32_t orderingPrimCount = 20000;
	constexpr uint32_t orderingRayCount = 512;
	constexpr double leafBudgetPerRay = 15.0;

	std::vector<NXB::Triangle> prims = GenerateTriangles(orderingPrimCount, smallSceneGridSize);
	NXB::BVH2 deviceBvh = Build(prims);
	NXB::BVH2::Host bvh = deviceBvh.ToHost();

	std::vector<Ray> rays = GenerateRays(bvh.bounds, orderingRayCount);
	size_t leaves = 0;
	size_t hits = 0;

	for (const Ray& ray : rays)
	{
		uint32_t hit = NXB::InvalidIdx;
		float tMax = NXB::RayMiss;
		NXB::TraverseBVH2(bvh, ray.origin, ray.invDirection, 0.0f, tMax,
			[&](uint32_t primIdx, float& tMax)
			{
				leaves++;
				float t;
				if (IntersectPrim(prims[primIdx], ray, 0.0f, tMax, t))
				{
					tMax = t;
					hit = primIdx;
				}
				return true;
			});
		if (hit != NXB::InvalidIdx)
			hits++;
	}

	const double leavesPerRay = (double)leaves / rays.size();
	CAPTURE(leavesPerRay);

	// A ray set that misses everything would meet any budget
	CHECK(hits > rays.size() / 8);
	CHECK(leavesPerRay < leafBudgetPerRay);
}

/*
 * The composition case.
 *
 * A TLAS is BuildBVH2<AABB> over instance world bounds, and two level traversal is the
 * leaf callback calling TraverseBVH2 again on the instance's own BVH. Nothing in NXB
 * knows what an instance is, which is the property this pins: if the callback seam ever
 * stops expressing this, it stops here rather than in somebody's renderer.
 *
 * Transforms are uniform scale plus translation, and the ray direction is deliberately
 * left unnormalized when it crosses into object space. That is what keeps t in the same
 * units on both sides, so tMax carries across the boundary unchanged -- the header says
 * so, and this is what checks it.
 */
TEST_CASE("Two level traversal composes through the leaf callback")
{
	struct Instance
	{
		uint32_t meshIdx;
		float3 translation;
		float scale;
	};

	const std::vector<std::vector<NXB::Triangle>> meshes = {
		GenerateTriangles(64, smallSceneGridSize),
		GenerateTriangles(200, smallSceneGridSize),
		GenerateTriangles(7, smallSceneGridSize)
	};

	const std::vector<Instance> instances = {
		{ 0, make_float3(0.0f, 0.0f, 0.0f), 1.0f },
		{ 1, make_float3(20.0f, 0.0f, 0.0f), 0.5f },
		{ 0, make_float3(0.0f, 20.0f, 0.0f), 2.0f },
		{ 2, make_float3(-15.0f, -15.0f, 5.0f), 1.5f },
		{ 1, make_float3(0.0f, 0.0f, 25.0f), 1.0f }
	};

	auto ToObjectSpace = [](const Instance& inst, const Ray& ray)
	{
		const float inv = 1.0f / inst.scale;
		return MakeRay(
			make_float3((ray.origin.x - inst.translation.x) * inv,
				(ray.origin.y - inst.translation.y) * inv,
				(ray.origin.z - inst.translation.z) * inv),
			make_float3(ray.direction.x * inv, ray.direction.y * inv, ray.direction.z * inv));
	};

	// One BLAS per mesh, held for the whole case: a BVH frees on the stream it was built
	// on, so it has to outlive every traversal of it
	std::vector<NXB::BVH2> blasDevice;
	std::vector<NXB::BVH2::Host> blas;
	for (const std::vector<NXB::Triangle>& mesh : meshes)
	{
		blasDevice.push_back(Build(mesh));
		blas.push_back(blasDevice.back().ToHost());
	}

	// The TLAS is a BVH over instance world bounds, built by the ordinary AABB path
	std::vector<NXB::AABB> instanceBounds;
	for (const Instance& inst : instances)
	{
		const NXB::AABB& local = blas[inst.meshIdx].bounds;
		instanceBounds.push_back(NXB::AABB(
			make_float3(local.bMin.x * inst.scale + inst.translation.x,
				local.bMin.y * inst.scale + inst.translation.y,
				local.bMin.z * inst.scale + inst.translation.z),
			make_float3(local.bMax.x * inst.scale + inst.translation.x,
				local.bMax.y * inst.scale + inst.translation.y,
				local.bMax.z * inst.scale + inst.translation.z)));
	}

	NXB::BVH2 tlasDevice = Build(instanceBounds);
	NXB::BVH2::Host tlas = tlasDevice.ToHost();

	uint32_t hits = 0;
	for (const Ray& ray : AllRayKinds(tlas.bounds, fastRayCount))
	{
		// Nested traversal: the TLAS callback transforms the ray and traverses the BLAS
		Hit nested{ NXB::RayMiss, NXB::InvalidIdx };
		uint32_t nestedInstance = NXB::InvalidIdx;
		float tMax = NXB::RayMiss;

		NXB::TraverseBVH2(tlas, ray.origin, ray.invDirection, 0.0f, tMax,
			[&](uint32_t instIdx, float& tMax)
			{
				const Instance& inst = instances[instIdx];
				const Ray local = ToObjectSpace(inst, ray);

				NXB::TraverseBVH2(blas[inst.meshIdx], local.origin, local.invDirection, 0.0f, tMax,
					[&](uint32_t primIdx, float& tMax)
					{
						float t;
						if (IntersectPrim(meshes[inst.meshIdx][primIdx], local, 0.0f, tMax, t))
						{
							tMax = t;
							nested.t = t;
							nested.primIdx = primIdx;
							nestedInstance = instIdx;
						}
						return true;
					});
				return true;
			});

		// Brute force over every instance, transforming the ray the same way so that a
		// mismatch can only be traversal
		Hit reference{ NXB::RayMiss, NXB::InvalidIdx };
		uint32_t referenceInstance = NXB::InvalidIdx;
		for (uint32_t i = 0; i < (uint32_t)instances.size(); ++i)
		{
			const Ray local = ToObjectSpace(instances[i], ray);
			const Hit hit = BruteForceClosestHit(meshes[instances[i].meshIdx], local, 0.0f, NXB::RayMiss);

			if (hit.t < reference.t)
			{
				reference = hit;
				referenceInstance = i;
			}
		}

		CHECK(nested.t == reference.t);
		CHECK(nestedInstance == referenceInstance);
		if (nested.t == reference.t && reference.primIdx != NXB::InvalidIdx)
			CHECK(nested.primIdx == reference.primIdx);

		if (reference.primIdx != NXB::InvalidIdx)
			hits++;
	}

	// A case where every ray missed would assert nothing at all
	CHECK(hits > 0);
}

} // TEST_SUITE fast


TEST_SUITE("slow")
{

TEST_CASE("BVH2 traversal over a large triangle scene")
{
	NXB::BuildConfig buildConfig;
	SUBCASE("32-bit morton codes") { buildConfig.prioritizeSpeed = true; }
	SUBCASE("64-bit morton codes") { buildConfig.prioritizeSpeed = false; }

	CAPTURE(largeScenePrimCount);
	BuildAndValidateTraversal(GenerateTriangles(largeScenePrimCount, largeSceneGridSize),
		slowRayCount, buildConfig);
}

TEST_CASE("BVH2 traversal over a large AABB scene")
{
	CAPTURE(largeScenePrimCount);
	BuildAndValidateTraversal(GenerateAABBs(largeScenePrimCount, largeSceneGridSize), slowRayCount);
}

TEST_CASE("Large scene depth fits the default traversal stack")
{
	NXB::BVH2 deviceBvh = Build(GenerateTriangles(largeScenePrimCount, largeSceneGridSize));
	uint32_t depth = MaxDepth(deviceBvh.ToHost());

	CAPTURE(largeScenePrimCount);
	CAPTURE(depth);
	CHECK(depth <= 32);
}

} // TEST_SUITE slow
