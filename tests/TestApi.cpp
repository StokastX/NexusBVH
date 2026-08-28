#include "vendor/doctest.h"

#include <cmath>
#include <vector>

#include <cuda_runtime.h>

#include <sstream>
#include <string>

#include "NXB/BVHBuilder.h"
#include "NXB/BVHCost.h"
#include "NXB/BenchmarkReport.h"
#include "NXB/DeviceBuffer.h"
#include "NXB/Error.h"

#include "TestChecks.h"
#include "support/BVHChecks.h"
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
	CHECK(bvh2.Empty());
	CHECK(bvh2.NodeCount() == 0);
	CHECK(bvh2.PrimCount() == 0);

	NXB::BVH8 bvh8 = NXB::BuildBVH8<NXB::Triangle>(nullptr, 0, buildConfig);
	CHECK(bvh8.Empty());
	CHECK(bvh8.NodeCount() == 0);
	CHECK(bvh8.PrimCount() == 0);
}

/*
 * ComputeSAHCost is not part of a build, so it has to work on any BVH the caller holds --
 * including an empty one, where there is no root to divide by and no kernel to launch.
 */
TEST_CASE("ComputeSAHCost evaluates a finished BVH")
{
	NXB::BuildConfig buildConfig;

	CHECK(NXB::ComputeSAHCost(NXB::BVH2()) == 0.0f);
	CHECK(NXB::ComputeSAHCost(NXB::BVH8()) == 0.0f);

	std::vector<NXB::Triangle> triangles = GenerateTriangles(1000, smallSceneGridSize);
	NXB::DeviceBuffer<NXB::Triangle> devicePrims(triangles);
	const uint32_t primCount = (uint32_t)triangles.size();

	NXB::BVH2 bvh2 = NXB::BuildBVH2<NXB::Triangle>(devicePrims.Get(), primCount, buildConfig);
	const float bvh2Cost = NXB::ComputeSAHCost(bvh2);
	REQUIRE(std::isfinite(bvh2Cost));
	CHECK(bvh2Cost > 0.0f);

	// Asking twice is a query, not a mutation, so the answer has to agree -- but not
	// bit for bit. The kernel sums its per-node terms with a float atomic, so the
	// summation order varies between launches and the last digit moves with it.
	CHECK(NXB::ComputeSAHCost(bvh2) == doctest::Approx(bvh2Cost));

	// The pool overload reaches the same value by a different allocation path
	NXB::MemoryPool pool;
	CHECK(NXB::ComputeSAHCost(bvh2, 0, pool.Handle()) == doctest::Approx(bvh2Cost));

	NXB::BVH8 bvh8 = NXB::BuildBVH8<NXB::Triangle>(devicePrims.Get(), primCount, buildConfig);
	const float bvh8Cost = NXB::ComputeSAHCost(bvh8);
	REQUIRE(std::isfinite(bvh8Cost));
	CHECK(bvh8Cost > 0.0f);
}

/*
 * BuildConfig::pool is a performance knob, so what the suite can pin is that it changes
 * nothing else: a pooled build produces a hierarchy satisfying the same invariants.
 */
TEST_CASE("A build from a MemoryPool is still a valid build")
{
	NXB::MemoryPool pool;

	NXB::BuildConfig buildConfig;
	buildConfig.pool = pool.Handle();

	std::vector<NXB::Triangle> triangles = GenerateTriangles(1000, smallSceneGridSize);
	NXB::DeviceBuffer<NXB::Triangle> devicePrims(triangles);

	NXB::BVH2 deviceBvh2 = NXB::BuildBVH2<NXB::Triangle>(devicePrims.Get(), 1000, buildConfig);
	NXB::BVH2::Host hostBvh2 = deviceBvh2.ToHost();
	CheckValid(ValidateBVH2(hostBvh2));
	CheckValid(ValidateSceneBounds(hostBvh2.bounds, ReferenceSceneBounds(triangles)));

	NXB::BVH8 deviceBvh8 = NXB::BuildBVH8<NXB::Triangle>(devicePrims.Get(), 1000, buildConfig);
	NXB::BVH8::Host hostBvh8 = deviceBvh8.ToHost();
	CheckValid(ValidateBVH8(hostBvh8, PrimBounds(triangles)));
}

/*
 * The point of the pool: repeated builds have to find their memory already reserved
 * instead of re-acquiring it from the driver. That is invisible to a functional test, so
 * without this case the feature could stop working with every other case still green.
 *
 * The used byte assertion is the sharper of the two. It is what caught the BVH arrays
 * being released with cudaFree, which does free them, but returns them to the driver
 * behind the pool's back and leaves its accounting permanently overstated.
 */
TEST_CASE("A MemoryPool is reused across builds")
{
	NXB::MemoryPool pool;

	NXB::BuildConfig buildConfig;
	buildConfig.pool = pool.Handle();

	std::vector<NXB::Triangle> triangles = GenerateTriangles(1000, smallSceneGridSize);
	NXB::DeviceBuffer<NXB::Triangle> devicePrims(triangles);

	const uint32_t primCount = 1000;

	auto BuildAndFree = [&] {
		NXB::BVH2 bvh = NXB::BuildBVH2<NXB::Triangle>(devicePrims.Get(), primCount, buildConfig);
	};

	// A few builds first, to let the pool settle on the size this scene needs
	for (uint32_t i = 0; i < 3; i++)
		BuildAndFree();

	const uint64_t settledReserved = pool.ReservedBytes();
	CHECK(settledReserved > 0);
	CHECK(pool.UsedBytes() == 0);

	/*
	 * Every buffer a BVH2 build holds live at once has to have come from this pool. The
	 * peak is exact to the byte, which is what makes this the assertion that catches one
	 * allocation site being left on the default pool -- ReservedBytes cannot, because the
	 * pool rounds it up to a 32 MB chunk that does not move with the scene size.
	 *
	 * Default BuildConfig means 64-bit Morton codes. Anything not listed here (cub's
	 * temporary storage, the two single element counters) only makes the real peak larger.
	 */
	const uint64_t nodeBytes = (2ull * primCount - 1) * sizeof(NXB::BVH2::Node);
	const uint64_t scratchBytes = primCount * (4ull + 4ull + 8ull + 8ull + 4ull);

	pool.ResetPeakUsedBytes();
	BuildAndFree();
	CHECK(pool.PeakUsedBytes() >= nodeBytes + scratchBytes);

	for (uint32_t i = 0; i < 20; i++)
		BuildAndFree();

	// Every build asks for the same sizes, so they have to be served from what the pool
	// already holds rather than by growing it
	CHECK(pool.ReservedBytes() == settledReserved);

	// And nothing a build allocated is still outstanding once its BVH has been freed
	CHECK(pool.UsedBytes() == 0);

	// The caller can have the VRAM back on demand, and the pool still works afterwards.
	// The synchronize is load bearing: the BVHs above released their buffers stream
	// ordered, and TrimTo can only give back memory whose free has already landed.
	NXB_CUDA_CHECK(cudaStreamSynchronize(0));
	pool.TrimTo(0);
	CHECK(pool.ReservedBytes() == 0);

	BuildAndFree();
	CHECK(pool.ReservedBytes() > 0);
}

} // TEST_SUITE fast


TEST_SUITE("slow")
{

// The error handling contract itself
TEST_CASE("A failed allocation throws instead of exiting")
{
	NXB::BuildConfig buildConfig;
	buildConfig.prioritizeSpeed = true;

	// 500M primitives needs 2n - 1 nodes at 32 B each, i.e. roughly 32 GB of BVH2 nodes.
	// The allocation fails before any kernel is launched, so the null primitive pointer
	// below is never dereferenced, and an out of memory is not sticky.
	const uint32_t primCount = 500000000;

	bool threw = false;
	try
	{
		NXB::BVH2 bvh = NXB::BuildBVH2<NXB::Triangle>(nullptr, primCount, buildConfig);
	}
	catch (const NXB::CudaError& error)
	{
		threw = true;
		CHECK(error.code == cudaErrorMemoryAllocation);

		// The message names the call that failed
		CHECK(std::string(error.what()).find("cudaMallocAsync") != std::string::npos);
	}
	CHECK(threw);

	/*
	 * Whatever the failed build had already allocated was released on the way out, so a
	 * smaller build still succeeds afterwards -- and what it returns has to be a real
	 * hierarchy, not merely a non null pointer.
	 *
	 * Walking it is what covers the state the failed build leaves behind. Throwing used
	 * to leave the error pending in the runtime, cub read that as the device being
	 * invalid, its temporary storage query failed unchecked, and this build then sorted
	 * nothing -- producing a tree that is structurally valid and quality wise garbage.
	 * The sort is checked now, so a return of that leak throws here instead of lying.
	 */
	std::vector<NXB::Triangle> triangles = GenerateTriangles(1000, smallSceneGridSize);
	NXB::DeviceBuffer<NXB::Triangle> devicePrims(triangles);

	NXB::BVH2 deviceBvh = NXB::BuildBVH2<NXB::Triangle>(devicePrims.Get(), 1000, buildConfig);
	REQUIRE(!deviceBvh.Empty());

	NXB::BVH2::Host hostBvh = deviceBvh.ToHost();

	CheckValid(ValidateBVH2(hostBvh));
	CheckValid(ValidateSceneBounds(hostBvh.bounds, ReferenceSceneBounds(triangles)));

}

/*
 * The build has to run on the stream the caller asked for, and drain it before
 * returning. Built at the large scene size on purpose: a tiny build can complete before
 * a missing synchronize would ever be observable, so this case would pass either way.
 *
 * Note the inner scope. A BVH2 releases its nodes on the stream it was built on, so it
 * has to be destroyed before that stream is: freeing on a destroyed stream fails with
 * cudaErrorInvalidResourceHandle, and a destructor can only discard that. Letting the BVH
 * outlive the stream here is what surfaced the whole class of bug -- the discarded error
 * stayed pending in the runtime and the NEXT test case in the same process died in cub
 * with a bogus cudaErrorInvalidDevice.
 */
TEST_CASE("Build on a caller owned stream")
{
	cudaStream_t stream = nullptr;
	NXB_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

	NXB::BuildConfig buildConfig;
	buildConfig.prioritizeSpeed = true;
	buildConfig.stream = stream;

	std::vector<NXB::Triangle> triangles = GenerateTriangles(largeScenePrimCount, largeSceneGridSize);
	NXB::DeviceBuffer<NXB::Triangle> devicePrims(triangles);

	{
		NXB::BVH2 deviceBvh = NXB::BuildBVH2<NXB::Triangle>(devicePrims.Get(), (uint32_t)triangles.size(), buildConfig);
		NXB::BVH2::Host hostBvh = deviceBvh.ToHost();

		CheckValid(ValidateBVH2(hostBvh));
		CheckValid(ValidateSceneBounds(hostBvh.bounds, ReferenceSceneBounds(triangles)));

		// The build synchronizes its own stream, so it must have drained by now
		CHECK(cudaStreamQuery(stream) == cudaSuccess);
	}

	NXB_CUDA_CHECK(cudaStreamDestroy(stream));

	// Nothing above may have left a failure pending for the next case in this process
	CHECK(cudaGetLastError() == cudaSuccess);
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
	NXB::DeviceBuffer<NXB::Triangle> devicePrims(triangles);

	NXB::BVHBuildMetrics metrics = {};
	NXB::BVH2 bvh = NXB::BuildBVH2<NXB::Triangle>(devicePrims.Get(), (uint32_t)triangles.size(),
		buildConfig, &metrics);

	CHECK(metrics.computeSceneBoundsTime > 0.0f);
	CHECK(metrics.computeMortonCodesTime > 0.0f);
	CHECK(metrics.radixSortTime > 0.0f);
	CHECK(metrics.bvhBuildTime > 0.0f);

	// A BVH2 build never runs the collapse, so the conversion timer stays untouched
	CHECK(metrics.bvh8ConversionTime == 0.0f);

	/*
	 * totalTime is measured with its own event pair around the whole build rather than
	 * summed from the steps, so it covers what the steps do not: the gaps between
	 * kernels, the allocations, and the scene bounds upload. It is therefore an upper
	 * bound on the sum, never equal to it.
	 */
	float stepSum = metrics.computeSceneBoundsTime + metrics.computeMortonCodesTime
		+ metrics.radixSortTime + metrics.bvhBuildTime;
	CHECK(stepSum <= metrics.totalTime);
	CHECK(metrics.totalTime > 0.0f);

}

TEST_CASE("Requesting metrics times the collapse of a BVH8 build")
{
	NXB::BuildConfig buildConfig;
	buildConfig.prioritizeSpeed = true;

	std::vector<NXB::Triangle> triangles = GenerateTriangles(1000, smallSceneGridSize);
	NXB::DeviceBuffer<NXB::Triangle> devicePrims(triangles);

	NXB::BVHBuildMetrics metrics = {};
	NXB::BVH8 bvh = NXB::BuildBVH8<NXB::Triangle>(devicePrims.Get(), (uint32_t)triangles.size(),
		buildConfig, &metrics);

	CHECK(metrics.bvh8ConversionTime > 0.0f);

	// See the BVH2 case above: totalTime is measured, not summed. Here it also spans the
	// scratch BVH2 build, whose own StepTimers wrote and then lost the field.
	float stepSum = metrics.computeSceneBoundsTime + metrics.computeMortonCodesTime
		+ metrics.radixSortTime + metrics.bvhBuildTime + metrics.bvh8ConversionTime;
	CHECK(stepSum <= metrics.totalTime);
	CHECK(metrics.totalTime > 0.0f);

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
	NXB::DeviceBuffer<NXB::Triangle> devicePrims(triangles);

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

	// Only the mean is linear, so only the mean keeps the step sum below the measured
	// total field by field. Taking the median or the minimum of each independently does
	// not: the median of a sum is not the sum of the medians, and nothing stops the
	// median total landing below the median of the steps that made it up.
	float stepSum = mean.computeSceneBoundsTime + mean.computeMortonCodesTime
		+ mean.radixSortTime + mean.bvhBuildTime + mean.bvh8ConversionTime;
	CHECK(stepSum <= mean.totalTime);
}

// The old implementation divided by measuredIterations unguarded and returned NaNs here
TEST_CASE("BenchmarkBuild with no measured iterations yields no samples")
{
	NXB::BuildConfig buildConfig;
	buildConfig.prioritizeSpeed = true;

	std::vector<NXB::Triangle> triangles = GenerateTriangles(1000, smallSceneGridSize);
	NXB::DeviceBuffer<NXB::Triangle> devicePrims(triangles);

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
	NXB::DeviceBuffer<NXB::Triangle> devicePrims(triangles);

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
