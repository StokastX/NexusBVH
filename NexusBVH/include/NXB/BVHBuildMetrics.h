#pragma once

namespace NXB
{
	struct BVHBuildMetrics
	{
		BVHBuildMetrics& operator+=(const BVHBuildMetrics& other) {
			computeTriangleBoundsTime += other.computeTriangleBoundsTime;
			computeSceneBoundsTime += other.computeSceneBoundsTime;
			computeMortonCodesTime += other.computeMortonCodesTime;
			radixSortTime += other.radixSortTime;
			initClustersTime += other.initClustersTime;
			bvhBuildTime += other.bvhBuildTime;
			totalTime += other.totalTime;
			return *this;
		}

		BVHBuildMetrics operator/(float divisor) const {
			BVHBuildMetrics result;
			result.computeTriangleBoundsTime = computeTriangleBoundsTime / divisor;
			result.computeSceneBoundsTime = computeSceneBoundsTime / divisor;
			result.computeMortonCodesTime = computeMortonCodesTime / divisor;
			result.radixSortTime = radixSortTime / divisor;
			result.initClustersTime = initClustersTime / divisor;
			result.bvhBuildTime = bvhBuildTime / divisor;
			result.totalTime = totalTime / divisor;
			return result;
		}

		float computeTriangleBoundsTime = 0.0f;
		float computeSceneBoundsTime = 0.0f;
		float computeMortonCodesTime = 0.0f;
		float radixSortTime = 0.0f;
		float initClustersTime = 0.0f;
		float bvhBuildTime = 0.0f;
		float totalTime = 0.0f;
	};
}