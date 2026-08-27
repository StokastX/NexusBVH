#pragma once

#include <iomanip>
#include <ostream>
#include <vector>

#include "BVHBuildMetrics.h"

/*
 * Optional reporting helper for BenchmarkBuild.
 *
 * This is the only public NexusBVH header that pulls in stream I/O. The library itself
 * never writes to a stream: a library that prints owns the application's stdout, and
 * there is no way for the caller to turn it off. Include this header only if you want
 * the canned report; otherwise reduce the samples yourself with Mean/Median/Min.
 */
namespace NXB
{
	namespace Detail
	{
		// PrintReport sets fixed/precision on the stream it is given. Leaving those on
		// would silently reformat everything the caller prints afterwards, so they are
		// restored on the way out.
		class StreamFormatGuard
		{
		public:
			explicit StreamFormatGuard(std::ostream& stream)
				: out(stream), flags(stream.flags()), precision(stream.precision()) { }

			~StreamFormatGuard()
			{
				out.flags(flags);
				out.precision(precision);
			}

			StreamFormatGuard(const StreamFormatGuard&) = delete;
			StreamFormatGuard& operator=(const StreamFormatGuard&) = delete;

		private:
			std::ostream& out;
			std::ios_base::fmtflags flags;
			std::streamsize precision;
		};

		inline void PrintMetricRow(std::ostream& out, const char* label,
			float mean, float median, float best)
		{
			out << "  " << std::left << std::setw(22) << label << std::right
				<< std::fixed << std::setprecision(3)
				<< std::setw(10) << mean
				<< std::setw(10) << median
				<< std::setw(10) << best << "\n";
		}
	}

	inline void PrintReport(std::ostream& out, const std::vector<BVHBuildMetrics>& samples)
	{
		if (samples.empty())
		{
			out << "\nBVH build benchmark: no measured iterations\n\n";
			return;
		}

		const Detail::StreamFormatGuard formatGuard(out);

		const BVHBuildMetrics mean = Mean(samples);
		const BVHBuildMetrics median = Median(samples);
		const BVHBuildMetrics best = Min(samples);

		// A BVH2 build never runs the collapse and leaves this timer at exactly zero,
		// which is what tells the two shapes of report apart
		const bool isWide = mean.bvh8ConversionTime > 0.0f;

		out << "\n===== BVH BUILD BENCHMARK (" << samples.size() << " iterations) =====\n\n";
		out << "  " << std::left << std::setw(22) << "Step (ms)" << std::right
			<< std::setw(10) << "mean" << std::setw(10) << "median" << std::setw(10) << "best" << "\n";

		Detail::PrintMetricRow(out, "Scene bounds",
			mean.computeSceneBoundsTime, median.computeSceneBoundsTime, best.computeSceneBoundsTime);
		Detail::PrintMetricRow(out, "Morton codes",
			mean.computeMortonCodesTime, median.computeMortonCodesTime, best.computeMortonCodesTime);
		Detail::PrintMetricRow(out, "Radix sort",
			mean.radixSortTime, median.radixSortTime, best.radixSortTime);
		Detail::PrintMetricRow(out, "BVH2 build",
			mean.bvhBuildTime, median.bvhBuildTime, best.bvhBuildTime);

		if (isWide)
			Detail::PrintMetricRow(out, "BVH8 conversion",
				mean.bvh8ConversionTime, median.bvh8ConversionTime, best.bvh8ConversionTime);

		Detail::PrintMetricRow(out, "Total", mean.totalTime, median.totalTime, best.totalTime);

		// Quality metrics are deterministic across identical builds, so only one column
		// of them means anything
		out << std::setprecision(2);
		out << "\n  BVH2 SAH cost: " << mean.bvh2Cost << "\n";
		if (isWide)
		{
			out << "  BVH8 SAH cost: " << mean.bvh8Cost << "\n";
			out << "  Average children per node: " << mean.averageChildPerNode << "\n";
		}
		out << "\n";
	}
}
