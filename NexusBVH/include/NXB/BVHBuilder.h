#pragma once
#include "BVH.h"
#include "AABB.h"
#include "Triangle.h"
#include "BuildConfig.h"

namespace NXB
{
	/* \brief Builds a binary BVH from a list of primitives
	 *
	 * \param primitives The primitives the BVH will be built from (one of AABB or Triangle)
	 * \param primCount The number of primitives
	 * \param buildConfig The build configuration
	 * \param buildMetrics The build metrics. If different from nullptr, kernel execution time will be measured which results in a slower build
	 *
	 * \returns A pointer to the device instance of the newly created binary BVH
	 */
	template <typename PrimT>
	BVH2* BuildBinary(PrimT* primitives, uint32_t primCount, BuildConfig buildConfig = BuildConfig(), BVHBuildMetrics* buildMetrics = nullptr);

	/* \brief Converts a binary BVH into a compressed wide BVH
	 *
	 * \param binaryBVH The binary BVH to be converted
	 *
	 * \returns A pointer to the device instance of the newly created compressed wide BVH
	 */

	BVH8* ConvertToWideBVH(BVH2* binaryBVH, BVHBuildMetrics* buildMetrics = nullptr);

	/*
	 * \brief Free the device instance of the binary BVH
	 */
	void FreeBVH(BVH2* bvh2);

	/*
	 * \brief Free the device instance of the wide BVH
	 */
	void FreeBVH(BVH8* wideBVH);

}