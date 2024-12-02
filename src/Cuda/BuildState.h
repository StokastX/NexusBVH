#pragma once
#include <iostream>
#include "Math/AABB.h"

namespace NXB
{
	struct BuildState
	{
		// List of Morton codes
		uint64_t* mortonCodes;

		// Bounds of the primitives
		AABB* primBounds;

		// Number of primitives
		uint32_t primCount;
	};
}