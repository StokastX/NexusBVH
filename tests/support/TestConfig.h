#pragma once

#include <cstdint>

/*
 * Primitive count used by the large scene cases. Set at configure time with
 * -DNEXUSBVH_TEST_PRIM_COUNT=<n>, which replaces the argv[1] the old suite took --
 * ctest runs each case with no arguments, so the knob has to live in the build.
 */
#ifndef NXB_TEST_PRIM_COUNT
	#define NXB_TEST_PRIM_COUNT 200000
#endif

namespace NXB::Test
{
	constexpr uint32_t largeScenePrimCount = NXB_TEST_PRIM_COUNT;

	// Cell count per axis for the generated scenes. The large scenes spread over a fine
	// grid; the tiny ones would leave most of a fine grid empty, so they use a coarse one.
	constexpr uint32_t largeSceneGridSize = 1000;
	constexpr uint32_t smallSceneGridSize = 8;
}
