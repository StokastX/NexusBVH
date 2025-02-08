#pragma once

namespace NXB
{
	struct BVHBuildMetrics
	{
		BVHBuildMetrics& operator+=(const BVHBuildMetrics& other) {
			computeSceneBoundsTime += other.computeSceneBoundsTime;
			computeMortonCodesTime += other.computeMortonCodesTime;
			radixSortTime += other.radixSortTime;
			bvhBuildTime += other.bvhBuildTime;
			totalTime += other.totalTime;
			cost += other.cost;
			return *this;
		}

		BVHBuildMetrics operator/(float divisor) const {
			BVHBuildMetrics result;
			result.computeSceneBoundsTime = computeSceneBoundsTime / divisor;
			result.computeMortonCodesTime = computeMortonCodesTime / divisor;
			result.radixSortTime = radixSortTime / divisor;
			result.bvhBuildTime = bvhBuildTime / divisor;
			result.totalTime = totalTime / divisor;
			result.cost = cost / divisor;
			return result;
		}

		float computeSceneBoundsTime = 0.0f;
		float computeMortonCodesTime = 0.0f;
		float radixSortTime = 0.0f;
		float bvhBuildTime = 0.0f;
		float totalTime = 0.0f;

		float cost = 0.0f;
	};

	/*
	 * \brief Benchmark the BVH build function
	 * 
	 * \param func The build function to benchmark
	 * \param warmupIterations The number of dummy calls to func to warm up the device
	 * \param measuredIterations The number of iterations used to measure the metrics
	 * \param args the arguments needed by func
	 * 
	 * \returns The average timing (in milliseconds) of every building step
	 */
	template<typename Func, typename ...Args>
	BVHBuildMetrics BenchmarkBuild(Func&& func, uint32_t warmupIterations, uint32_t measuredIterations, Args && ...args)
	{
		BVHBuildMetrics aggregatedMetrics = {};

		std::cout << std::endl << "========== BENCHMARKING BVH BUILD ==========" << std::endl << std::endl;
		// Warm-up: build several times
		for (uint32_t i = 0; i < warmupIterations; ++i) {
			BVHBuildMetrics dummy;
			auto bvh = std::forward<Func>(func)(std::forward<Args>(args)..., &dummy);
			FreeBVH(bvh);
		}

		for (uint32_t i = 0; i < measuredIterations; ++i) {
			BVHBuildMetrics iterationMetrics = {};
			auto bvh = std::forward<Func>(func)(std::forward<Args>(args)..., &iterationMetrics);
			FreeBVH(bvh);
			aggregatedMetrics += iterationMetrics;
			std::cout << "Iteration " << i << ", total time: " << iterationMetrics.totalTime << " ms" << std::endl;
		}
		aggregatedMetrics = aggregatedMetrics / static_cast<float>(measuredIterations);

		std::cout << std::endl << "========== BENCHMARK RESULTS ==========" << std::endl << std::endl;

		std::cout << "Scene bounds: " << aggregatedMetrics.computeSceneBoundsTime << " ms" << std::endl;
		std::cout << "Morton codes: " << aggregatedMetrics.computeMortonCodesTime << " ms" << std::endl;
		std::cout << "Radix sort: " << aggregatedMetrics.radixSortTime << " ms" << std::endl;
		std::cout << "Bvh build time: " << aggregatedMetrics.bvhBuildTime << " ms" << std::endl;
		std::cout << "Total BVH build time: " << aggregatedMetrics.totalTime << " ms" << std::endl;

		std::cout << std::endl << "========== BENCHMARKING DONE ==========" << std::endl << std::endl;

		return aggregatedMetrics;
	}
}