#pragma once
#include <cstdint>
#include "BVH.h"

namespace NXB
{
	struct BVHBuildMetrics
	{
		BVHBuildMetrics& operator+=(const BVHBuildMetrics& other) {
			computeSceneBoundsTime += other.computeSceneBoundsTime;
			computeMortonCodesTime += other.computeMortonCodesTime;
			radixSortTime += other.radixSortTime;
			bvhBuildTime += other.bvhBuildTime;
			bvh8ConversionTime += other.bvh8ConversionTime;
			totalTime += other.totalTime;
			bvh2Cost += other.bvh2Cost;
			bvh8Cost += other.bvh8Cost;
			averageChildPerNode += other.averageChildPerNode;
			return *this;
		}

		BVHBuildMetrics operator/(float divisor) const {
			BVHBuildMetrics result;
			result.computeSceneBoundsTime = computeSceneBoundsTime / divisor;
			result.computeMortonCodesTime = computeMortonCodesTime / divisor;
			result.radixSortTime = radixSortTime / divisor;
			result.bvhBuildTime = bvhBuildTime / divisor;
			result.bvh8ConversionTime = bvh8ConversionTime / divisor;
			result.totalTime = totalTime / divisor;
			result.bvh2Cost = bvh2Cost / divisor;
			result.bvh8Cost = bvh8Cost / divisor;
			result.averageChildPerNode = averageChildPerNode / divisor;
			return result;
		}

		// Timings
		float computeSceneBoundsTime = 0.0f;
		float computeMortonCodesTime = 0.0f;
		float radixSortTime = 0.0f;
		float bvhBuildTime = 0.0f;
		float bvh8ConversionTime = 0.0f;
		float totalTime = 0.0f;

		// SAH cost
		float bvh2Cost = 0.0f;
		float bvh8Cost = 0.0f;

		// For wide BVHs: average number of children per internal node
		float averageChildPerNode = 0.0f;
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
		using ReturnT = decltype(func(std::forward<Args>(args)..., nullptr));

		std::cout << std::endl << "========== BENCHMARKING BVH BUILD ==========" << std::endl << std::endl;
		// Warm-up: build several times
		for (uint32_t i = 0; i < warmupIterations; ++i) {
			BVHBuildMetrics dummy;
			auto bvh = std::forward<Func>(func)(std::forward<Args>(args)..., &dummy);
			FreeDeviceBVH(bvh);
		}

		for (uint32_t i = 0; i < measuredIterations; ++i) {
			BVHBuildMetrics iterationMetrics = {};
			auto bvh = std::forward<Func>(func)(std::forward<Args>(args)..., &iterationMetrics);
			FreeDeviceBVH(bvh);
			aggregatedMetrics += iterationMetrics;
			std::cout << "Iteration " << i << ", total time: " << iterationMetrics.totalTime << " ms" << std::endl;
		}
		aggregatedMetrics = aggregatedMetrics / static_cast<float>(measuredIterations);

		std::cout << std::endl << "========== BENCHMARK RESULTS ==========" << std::endl << std::endl;

		std::cout << "Scene bounds: " << aggregatedMetrics.computeSceneBoundsTime << " ms" << std::endl;
		std::cout << "Morton codes: " << aggregatedMetrics.computeMortonCodesTime << " ms" << std::endl;
		std::cout << "Radix sort: " << aggregatedMetrics.radixSortTime << " ms" << std::endl;
		std::cout << "BVH2 build time: " << aggregatedMetrics.bvhBuildTime << " ms" << std::endl;

		if constexpr (std::is_same_v<ReturnT, BVH8>)
			std::cout << "BVH8 conversion time: " << aggregatedMetrics.bvh8ConversionTime << " ms" << std::endl;

		std::cout << "Total BVH build time: " << aggregatedMetrics.totalTime << " ms" << std::endl;

		std::cout << std::endl << "BVH2 cost: " << aggregatedMetrics.bvh2Cost << std::endl;
		if constexpr (std::is_same_v<ReturnT, BVH8>)
		{
			std::cout << "BVH8 cost: " << aggregatedMetrics.bvh8Cost << std::endl;
			std::cout << "Average children per node: " << aggregatedMetrics.averageChildPerNode << std::endl;
		}

		std::cout << std::endl << "========== BENCHMARKING DONE ==========" << std::endl << std::endl;

		return aggregatedMetrics;
	}
}