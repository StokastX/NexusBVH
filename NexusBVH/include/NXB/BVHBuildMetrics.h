#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "BVH.h"

namespace NXB
{
	struct BVHBuildMetrics
	{
		// Timings, in milliseconds
		float computeSceneBoundsTime = 0.0f;
		float computeMortonCodesTime = 0.0f;
		float radixSortTime = 0.0f;
		float bvhBuildTime = 0.0f;
		float bvh8ConversionTime = 0.0f;
		float totalTime = 0.0f;
	};


	namespace Detail
	{
		// Every field of BVHBuildMetrics, so the reductions below state the list once
		// instead of once per statistic
		inline constexpr float BVHBuildMetrics::* metricFields[] = {
			&BVHBuildMetrics::computeSceneBoundsTime,
			&BVHBuildMetrics::computeMortonCodesTime,
			&BVHBuildMetrics::radixSortTime,
			&BVHBuildMetrics::bvhBuildTime,
			&BVHBuildMetrics::bvh8ConversionTime,
			&BVHBuildMetrics::totalTime
		};
	}

	/*
	 * Reductions over a set of samples.
	 *
	 * Each field is reduced independently, so the result does not correspond to any one
	 * iteration. That is the right thing for per-step timings, but it means only Mean
	 * preserves the identity `sum of steps == totalTime`: the median of a sum is not the
	 * sum of the medians. All three return a zeroed struct for an empty sample set rather
	 * than dividing by zero.
	 */

	inline BVHBuildMetrics Mean(const std::vector<BVHBuildMetrics>& samples)
	{
		BVHBuildMetrics result = {};
		if (samples.empty())
			return result;

		for (float BVHBuildMetrics::* field : Detail::metricFields)
		{
			float sum = 0.0f;
			for (const BVHBuildMetrics& sample : samples)
				sum += sample.*field;

			result.*field = sum / (float)samples.size();
		}
		return result;
	}

	// More robust than the mean for kernel timings: one descheduled iteration or a driver
	// interrupt skews an average, but moves a median by nothing
	inline BVHBuildMetrics Median(std::vector<BVHBuildMetrics> samples)
	{
		BVHBuildMetrics result = {};
		if (samples.empty())
			return result;

		std::vector<float> values(samples.size());
		for (float BVHBuildMetrics::* field : Detail::metricFields)
		{
			for (size_t i = 0; i < samples.size(); ++i)
				values[i] = samples[i].*field;

			std::sort(values.begin(), values.end());

			const size_t mid = values.size() / 2;
			result.*field = (values.size() % 2 == 0)
				? 0.5f * (values[mid - 1] + values[mid])
				: values[mid];
		}
		return result;
	}

	// The fastest iteration observed, i.e. the run least disturbed by everything else on
	// the machine. Usually the number worth quoting for a kernel.
	inline BVHBuildMetrics Min(const std::vector<BVHBuildMetrics>& samples)
	{
		BVHBuildMetrics result = {};
		if (samples.empty())
			return result;

		for (float BVHBuildMetrics::* field : Detail::metricFields)
		{
			float best = samples.front().*field;
			for (const BVHBuildMetrics& sample : samples)
				best = std::min(best, sample.*field);

			result.*field = best;
		}
		return result;
	}


	/*
	 * \brief Benchmark a BVH build function
	 *
	 * \param func The build function to benchmark
	 * \param warmupIterations Dummy calls to func, discarded, to warm the device up
	 * \param measuredIterations The number of iterations that are kept
	 * \param args The arguments needed by func, minus the trailing metrics pointer
	 *
	 * \returns One BVHBuildMetrics per measured iteration, in the order they ran
	 *
	 * Returns the raw samples rather than a summary: a mean alone hides variance and
	 * cannot be turned back into a median or a minimum. Reduce them with Mean, Median or
	 * Min above.
	 *
	 * This prints nothing. Reporting is the application's job -- include
	 * NXB/BenchmarkReport.h if you want a ready made one.
	 *
	 * Note that passing a metrics pointer inserts sync points around every kernel, so a
	 * measured build is measurably slower than the build a user actually gets.
	 */
	template<typename Func, typename ...Args>
	std::vector<BVHBuildMetrics> BenchmarkBuild(Func&& func, uint32_t warmupIterations,
		uint32_t measuredIterations, Args&& ...args)
	{
		std::vector<BVHBuildMetrics> samples;
		samples.reserve(measuredIterations);

		// args are named lvalues inside this function and are deliberately NOT forwarded:
		// the loops below call func repeatedly, and forwarding would move the same
		// arguments once per iteration.
		for (uint32_t i = 0; i < warmupIterations; ++i)
		{
			BVHBuildMetrics warmupMetrics = {};

			// The BVH returned owns its memory and releases it at the end of the iteration
			func(args..., &warmupMetrics);
		}

		for (uint32_t i = 0; i < measuredIterations; ++i)
		{
			BVHBuildMetrics iterationMetrics = {};
			func(args..., &iterationMetrics);
			samples.push_back(iterationMetrics);
		}

		return samples;
	}
}
