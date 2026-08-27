#pragma once
#include <cuda_runtime.h>
#include "BVHBuildMetrics.h"

namespace NXB
{
	struct BuildConfig
	{
		// Wether to use 64-bit or 32-bit Morton keys for positional encoding.
		// When prioritizeSpeed is set to true, sorting is faster but positional
		// encoding has a limited accuracy which results in a lower BVH quality.
		bool prioritizeSpeed = false;

		// The stream every allocation, copy and kernel of the build is issued on.
		// The build is synchronized against this stream only, so a caller can
		// overlap it with unrelated work instead of having the library stall the
		// whole device. Defaults to the legacy default stream.
		cudaStream_t stream = nullptr;
	};
}
