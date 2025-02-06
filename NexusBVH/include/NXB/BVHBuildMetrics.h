#pragma once

namespace NXB
{
	struct BVHBuildMetrics
	{
		float computeTriangleBoundsTime = 0.0f;
		float computeSceneBoundsTime = 0.0f;
		float computeMortonCodesTime = 0.0f;
		float radixSortTime = 0.0f;
		float initClustersTime = 0.0f;
		float bvhBuildTime = 0.0f;
		float totalTime = 0.0f;
	};
}