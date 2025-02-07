#pragma once
#include "BVH.h"
#include "AABB.h"
#include "Triangle.h"
#include "BVHBuildMetrics.h"

namespace NXB
{
	class BVHBuilder
	{
	public:
		/* \brief Builds a binary BVH from a list of primitives
		 *
		 * \param primitives The primitives the BVH will be built from
		 * \param primCount The number of primitives
		 * 
		 * \returns A pointer to the device instance of the newly created binary BVH
		 */
		BVH2* BuildBinary(AABB* primitives, uint32_t primCount, BVHBuildMetrics* buildMetrics = nullptr);

		/* \brief Builds a binary BVH from a list of primitives
		 *
		 * \param primitives The primitives the BVH will be built from
		 * \param primCount The number of primitives
		 * 
		 * \returns A pointer to the device instance of the newly created binary BVH
		 */
		BVH2* BuildBinary(Triangle* primitives, uint32_t primCount, BVHBuildMetrics* buildMetrics = nullptr);

		/* \brief Converts a binary BVH into a compressed wide BVH
		 *
		 * \param binaryBVH The binary BVH to be converted
		 * 
		 * \returns A pointer to the device instance of the newly created compressed wide BVH
		 */

		BVH8* ConvertToWideBVH(BVH2* binaryBVH, BVHBuildMetrics* buildMetrics = nullptr);


	private:
	};

	/*
	 * \brief Free the device instance of the binary BVH
	 */
	void FreeBVH(BVH2* bvh2);

	/*
	 * \brief Free the device instance of the wide BVH
	 */
	void FreeBVH(BVH8* wideBVH);

}