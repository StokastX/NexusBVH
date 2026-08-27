#include "vendor/doctest.h"

#include <cmath>
#include <vector>

#include <cuda_runtime.h>

#include <sstream>
#include <string>

#include "NXB/BVHBuilder.h"
#include "NXB/BenchmarkReport.h"

#include "TestChecks.h"
#include "support/BVHChecks.h"
#include "support/CudaTestCheck.h"
#include "support/DeviceBuffer.h"
#include "support/Scenes.h"
#include "support/TestConfig.h"

using namespace NXB::Test;

/*
 * Contract of the public API itself, as opposed to the shape of the hierarchies it
 * produces: degenerate inputs, the caller's stream, and the metrics plumbing.
 */

TEST_SUITE("fast")
{

TEST_CASE("Empty input returns an empty BVH")
{
	NXB::BuildConfig buildConfig;

	NXB::BVH2 bvh2 = NXB::BuildBVH2<NXB::Triangle>(nullptr, 0, buildConfig);
	CHECK(bvh2.nodes == nullptr);
	CHECK(bvh2.nodeCount == 0);
	CHECK(bvh2.primCount == 0);

	NXB::BVH8 bvh8 = NXB::BuildBVH8<NXB::Triangle>(nullptr, 0, buildConfig);
	CHECK(bvh8.nodes == nullptr);
	CHECK(bvh8.nodeCount == 0);
	CHECK(bvh8.primCount == 0);
}

} // TEST_SUITE fast


TEST_SUITE("slow")
{

/*
 * The build has to run on the stream the caller asked for, and drain it before
 * returning. Built at the large scene size on purpose: a tiny build can complete before
 * a missing synchronize would ever be observable, so this case would pass either way.
 */
TEST_CASE("Build on a caller owned stream")
{
	cudaStream_t stream = nullptr;
	NXB_TEST_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

	NXB::BuildConfig buildConfig;
	buildConfig.prioritizeSpeed = true;
	buildConfig.stream = stream;

	std::vector<NXB::Triangle> triangles = GenerateTriangles(largeScenePrimCount, largeSceneGridSize);
	DeviceBuffer<NXB::Triangle> devicePrims(triangles);

	NXB::BVH2 deviceBvh = NXB::BuildBVH2<NXB::Triangle>(devicePrims.Get(), (uint32_t)triangles.size(), buildConfig);
	NXB::BVH2 hostBvh = NXB::ToHost(deviceBvh);

	CheckValid(ValidateBVH2(hostBvh));
	CheckValid(ValidateSceneBounds(hostBvh.bounds, ReferenceSceneBounds(triangles)));

	// The build synchronizes its own stream, so it must have drained by now
	CHECK(cudaStreamQuery(stream) == cudaSuccess);

	NXB::FreeHostBVH(hostBvh);
	NXB::FreeDeviceBVH(deviceBvh);
	NXB_TEST_CUDA_CHECK(cudaStreamDestroy(stream));
}

/*
 * Every step runs unconditionally when metrics are requested, so a zero here means the
 * timer is not wired up, not that the step was skipped.
 */
} // TEST_SUITE slow


TEST_SUITE("fast")
{

TEST_CASE("Requesting metrics times every step of a BVH2 build")
{
	NXB::BuildConfig buildConfig;
	buildConfig.prioritizeSpeed = true;

	std::vector<NXB::Triangle> triangles = GenerateTriangles(1000, smallSceneGridSize);
	DeviceBuffer<NXB::Triangle> devicePrims(triangles);

	NXB::BVHBuildMetrics metrics = {};
	NXB::BVH2 bvh = NXB::BuildBVH2<NXB::Triangle>(devicePrims.Get(), (uint32_t)triangles.size(),
		buildConfig, &metrics);

	CHECK(metrics.computeSceneBoundsTime > 0.0f);
	CHECK(metrics.computeMortonCodesTime > 0.0f);
	CHECK(metrics.radixSortTime > 0.0f);
	CHECK(metrics.bvhBuildTime > 0.0f);

	// A BVH2 build never runs the collapse, so totalTime is the sum of the four steps
	// above and the conversion timer stays untouched
	CHECK(metrics.bvh8ConversionTime == 0.0f);

	float stepSum = metrics.computeSceneBoundsTime + metrics.computeMortonCodesTime
		+ metrics.radixSortTime + metrics.bvhBuildTime;
	CHECK(std::fabs(stepSum - metrics.totalTime) < 1e-3f);

	CHECK(metrics.bvh2Cost > 0.0f);

	NXB::FreeDeviceBVH(bvh);
}

TEST_CASE("Requesting metrics times the collapse of a BVH8 build")
{
	NXB::BuildConfig buildConfig;
	buildConfig.prioritizeSpeed = true;

	std::vector<NXB::Triangle> triangles = GenerateTriangles(1000, smallSceneGridSize);
	DeviceBuffer<NXB::Triangle> devicePrims(triangles);

	NXB::BVHBuildMetrics metrics = {};
	NXB::BVH8 bvh = NXB::BuildBVH8<NXB::Triangle>(devicePrims.Get(), (uint32_t)triangles.size(),
		buildConfig, &metrics);

	CHECK(metrics.bvh8ConversionTime > 0.0f);

	float stepSum = metrics.computeSceneBoundsTime + metrics.computeMortonCodesTime
		+ metrics.radixSortTime + metrics.bvhBuildTime + metrics.bvh8ConversionTime;
	CHECK(std::fabs(stepSum - metrics.totalTime) < 1e-3f);

	NXB::FreeDeviceBVH(bvh);
}

/*
 * BenchmarkBuild is public API, and the exe that used to be the only thing exercising it
 * has been deleted. Tiny counts here: this checks the wrapper's bookkeeping, not how fast
 * the build is.
 */
TEST_CASE("BenchmarkBuild returns one sample per measured iteration")
{
	NXB::BuildConfig buildConfig;
	buildConfig.prioritizeSpeed = true;

	std::vector<NXB::Triangle> triangles = GenerateTriangles(1000, smallSceneGridSize);
	DeviceBuffer<NXB::Triangle> devicePrims(triangles);

	std::vector<NXB::BVHBuildMetrics> samples = NXB::BenchmarkBuild(NXB::BuildBVH8<NXB::Triangle>,
		2, 3, devicePrims.Get(), (uint32_t)triangles.size(), buildConfig);

	REQUIRE(samples.size() == 3);
	for (const NXB::BVHBuildMetrics& sample : samples)
	{
		CHECK(sample.totalTime > 0.0f);
		CHECK(sample.bvh8ConversionTime > 0.0f);
	}

	const NXB::BVHBuildMetrics mean = NXB::Mean(samples);
	const NXB::BVHBuildMetrics median = NXB::Median(samples);
	const NXB::BVHBuildMetrics best = NXB::Min(samples);

	// The minimum bounds both of the others from below; median vs mean can go either way,
	// so there is nothing to assert between those two
	CHECK(best.totalTime <= mean.totalTime);
	CHECK(best.totalTime <= median.totalTime);

	// Only the mean is linear, so only the mean preserves the step sum. Taking the median
	// or the minimum of each field independently does not: the median of a sum is not the
	// sum of the medians.
	float stepSum = mean.computeSceneBoundsTime + mean.computeMortonCodesTime
		+ mean.radixSortTime + mean.bvhBuildTime + mean.bvh8ConversionTime;
	CHECK(std::fabs(stepSum - mean.totalTime) < 1e-3f);
}

// The old implementation divided by measuredIterations unguarded and returned NaNs here
TEST_CASE("BenchmarkBuild with no measured iterations yields no samples")
{
	NXB::BuildConfig buildConfig;
	buildConfig.prioritizeSpeed = true;

	std::vector<NXB::Triangle> triangles = GenerateTriangles(1000, smallSceneGridSize);
	DeviceBuffer<NXB::Triangle> devicePrims(triangles);

	std::vector<NXB::BVHBuildMetrics> samples = NXB::BenchmarkBuild(NXB::BuildBVH8<NXB::Triangle>,
		0, 0, devicePrims.Get(), (uint32_t)triangles.size(), buildConfig);

	CHECK(samples.empty());

	// Zeroed, not NaN
	CHECK(NXB::Mean(samples).totalTime == 0.0f);
	CHECK(NXB::Median(samples).totalTime == 0.0f);
	CHECK(NXB::Min(samples).totalTime == 0.0f);
}

// The report is opt-in and lives in its own header so the rest of the API stays free of
// stream I/O. Writing to a stringstream keeps the suite's own output clean.
TEST_CASE("PrintReport renders a report without touching stdout")
{
	NXB::BuildConfig buildConfig;
	buildConfig.prioritizeSpeed = true;

	std::vector<NXB::Triangle> triangles = GenerateTriangles(1000, smallSceneGridSize);
	DeviceBuffer<NXB::Triangle> devicePrims(triangles);

	std::vector<NXB::BVHBuildMetrics> samples = NXB::BenchmarkBuild(NXB::BuildBVH8<NXB::Triangle>,
		0, 2, devicePrims.Get(), (uint32_t)triangles.size(), buildConfig);

	std::ostringstream out;
	NXB::PrintReport(out, samples);

	const std::string report = out.str();
	CHECK(report.find("BVH8 conversion") != std::string::npos);
	CHECK(report.find("Total") != std::string::npos);

	// An empty sample set must not divide by zero or print a bogus table
	std::ostringstream emptyOut;
	NXB::PrintReport(emptyOut, {});
	CHECK(emptyOut.str().find("no measured iterations") != std::string::npos);
}

/*
 * Triangle's accessors are public API that nothing inside the library calls -- the
 * builder goes through AABB::Centroid(). That is exactly how Centroid() shipped
 * returning (v0 + v1 + v2) * 0.5 instead of / 3.0f without anything noticing.
 */
TEST_CASE("Triangle accessors")
{
	// Right triangle in the z = 0 plane, so every expected value is exact
	const NXB::Triangle tri({ 0.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f });

	float3 centroid = tri.Centroid();
	CHECK(centroid.x == doctest::Approx(1.0f / 3.0f));
	CHECK(centroid.y == doctest::Approx(1.0f / 3.0f));
	CHECK(centroid.z == doctest::Approx(0.0f));

	NXB::AABB bounds = tri.Bounds();
	CHECK(bounds.bMin.x == doctest::Approx(0.0f));
	CHECK(bounds.bMin.y == doctest::Approx(0.0f));
	CHECK(bounds.bMin.z == doctest::Approx(0.0f));
	CHECK(bounds.bMax.x == doctest::Approx(1.0f));
	CHECK(bounds.bMax.y == doctest::Approx(1.0f));
	CHECK(bounds.bMax.z == doctest::Approx(0.0f));

	// Not normalized, and right handed: cross(v1 - v0, v2 - v0)
	float3 normal = tri.Normal();
	CHECK(normal.x == doctest::Approx(0.0f));
	CHECK(normal.y == doctest::Approx(0.0f));
	CHECK(normal.z == doctest::Approx(1.0f));

	CHECK(tri.Area() == doctest::Approx(0.5f));

	// AABB::Area() returns *half* the surface area (the xy + yz + zx sum), on purpose
	CHECK(bounds.Area() == doctest::Approx(1.0f));
}

} // TEST_SUITE fast
