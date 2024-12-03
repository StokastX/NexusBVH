#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include "Math/AABB.h"

namespace NXB
{
	/* \brief Compute a list of 64-bit Morton codes
	 *
	 * \param bounds The list of AABBs whose centroid will be used to generate the keys
	 * \param mortonCodes the list of mortonCodes to be filled
	 * \param size The number of centroids
	 */
	__global__ void ComputeMortonCodes(AABB* bounds, uint64_t* mortonCodes, uint32_t size);

	/* \brief One sweep radix sort for 64-bit Morton codes
	 *
	 * \param mortonCodes The list of Morton codes to be sorted
	 * \param primIds The list of primitive indices
	 * \param size The number of Morton codes
	 */
	void RadixSort(uint64_t* mortonCodes, uint32_t* primIds, uint32_t size);
}