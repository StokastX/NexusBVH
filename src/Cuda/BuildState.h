#pragma once

#include <iostream>
#include "Math/AABB.h"

namespace NXB
{
	struct BuildState
	{
		// List of Morton codes
		uint64_t* mortonCodes;

		// Bounds of the primitives (or primitives if primType == AABB)
		AABB* primBounds;

		// Scene bounds
		AABB* sceneBounds;

		// Indices of the primitives
		uint32_t* primIdx;

		// Number of primitives
		uint32_t primCount;
	};
}
