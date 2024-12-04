#pragma once
#include "BVH/BVH.h"

namespace NXB
{
	class BVHBuilder
	{
	public:
		/* \brief Builds a BVH from a list of primitives
		 *
		 * \param primitives The primitives the BVH will be built from
		 * \param primCount The number of primitives
		 * \param primType The type of primitives. 
		 * One of NXB::PrimType::AABB or NXB::PrimType::Triangle.
		 * Each AABB should be composed of two float3 and each triangle of three float3
		 * 
		 * \returns A pointer to the device instance of the newly created binary BVH
		 */
		BVH2* BuildBinary(float3* primitives, uint32_t primCount, PrimType primType);

		/* \brief Converts a binary BVH into a compressed wide BVH
		 *
		 * \param binaryBVH The binary BVH to be converted
		 * 
		 * \returns A pointer to the device instance of the newly created compressed wide BVH
		 */
		BVH8* ConvertToWideBVH(BVH2* binaryBVH);

		/*
		 * \brief Free the device instance of the binary BVH
		 */
		void FreeBVH(BVH2* bindaryBVH);

		/*
		 * \brief Free the device instance of the wide BVH
		 */
		void FreeBVH(BVH8* wideBVH);

	private:
	};
}