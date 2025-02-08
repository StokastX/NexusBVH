#pragma once
#include "BVHBuildMetrics.h"

namespace NXB
{
	struct BuildConfig
	{
		// Wether to use 64-bit or 32-bit Morton keys for positional encoding.
		// When prioritizeSpeed is set to true, sorting is faster but positional
		// encoding has a limited accuracy which results in a lower BVH quality.
		bool prioritizeSpeed = false;
	};
}