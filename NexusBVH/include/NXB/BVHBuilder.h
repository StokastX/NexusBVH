#pragma once
#include "BVH.h"
#include "AABB.h"
#include "Triangle.h"
#include "BVHBuildMetrics.h"

namespace NXB
{
	class BVHBuilder
	{
	public:
		/* \brief Builds a binary BVH from a list of primitives
		 *
		 * \param primitives The primitives the BVH will be built from
		 * \param primCount The number of primitives
		 * 
		 * \returns A pointer to the device instance of the newly created binary BVH
		 */
		BVH2* BuildBinary(AABB* primitives, uint32_t primCount, BVHBuildMetrics* buildMetrics = nullptr);

		/* \brief Builds a binary BVH from a list of primitives
		 *
		 * \param primitives The primitives the BVH will be built from
		 * \param primCount The number of primitives
		 * 
		 * \returns A pointer to the device instance of the newly created binary BVH
		 */
		BVH2* BuildBinary(Triangle* primitives, uint32_t primCount, BVHBuildMetrics* buildMetrics = nullptr);

		/* \brief Converts a binary BVH into a compressed wide BVH
		 *
		 * \param binaryBVH The binary BVH to be converted
		 * 
		 * \returns A pointer to the device instance of the newly created compressed wide BVH
		 */

		BVH8* ConvertToWideBVH(BVH2* binaryBVH, BVHBuildMetrics* buildMetrics = nullptr);


	private:
	};

	/*
	 * \brief Free the device instance of the binary BVH
	 */
	void FreeBVH(BVH2* bvh2);

	/*
	 * \brief Free the device instance of the wide BVH
	 */
	void FreeBVH(BVH8* wideBVH);


	/*
	 * \brief Benchmark the BVH build function
	 * 
	 * \param func The build function to benchmark
	 * \param warmupIterations The number of dummy calls to func to warm up the device
	 * \param measuredIterations The number of iterations used to measure the metrics
	 * \param args the argument needed by func
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

		std::cout << "Triangle bounds: " << aggregatedMetrics.computeTriangleBoundsTime << " ms" << std::endl;
		std::cout << "Scene bounds: " << aggregatedMetrics.computeSceneBoundsTime << " ms" << std::endl;
		std::cout << "Morton codes: " << aggregatedMetrics.computeMortonCodesTime << " ms" << std::endl;
		std::cout << "Radix sort: " << aggregatedMetrics.radixSortTime << " ms" << std::endl;
		std::cout << "Clusters init: " << aggregatedMetrics.initClustersTime << " ms" << std::endl;
		std::cout << "Bvh build time: " << aggregatedMetrics.bvhBuildTime << " ms" << std::endl;
		std::cout << "Total BVH build time: " << aggregatedMetrics.totalTime << " ms" << std::endl;

		std::cout << std::endl << "========== BENCHMARKING DONE ==========" << std::endl << std::endl;

		return aggregatedMetrics;
	}
}