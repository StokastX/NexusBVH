#pragma once
#include "BVH.h"
#include "AABB.h"
#include "DeviceBuffer.h"
#include "Error.h"
#include "MemoryPool.h"
#include "Triangle.h"
#include "BuildConfig.h"

namespace NXB
{
	/*
	 * Error handling: every function below reports a CUDA failure by throwing
	 * NXB::CudaError, and none of them terminate the process. A build that throws has
	 * already released everything it allocated, so catching it leaks nothing and the
	 * caller is free to retry with fewer primitives.
	 *
	 * The primitive pointers are DEVICE pointers, and the BVH2 / BVH8 handles returned
	 * hold device pointers too. NXB::DeviceBuffer is the RAII way to produce the former;
	 * the latter are released with FreeDeviceBVH.
	 */

	/* \brief Builds a binary BVH from a list of primitives
	 *
	 * \param primitives The primitives the BVH will be built from (AABB or Triangle)
	 * \param primCount The number of primitives
	 * \param buildConfig The build configuration
	 * \param buildMetrics The build metrics. If different from nullptr, kernel execution times will be measured which results in a slower build
	 *
	 * \returns A pointer to the device instance of the newly created binary BVH
	 */
	template <typename PrimT>
	BVH2 BuildBVH2(PrimT* primitives, uint32_t primCount, BuildConfig buildConfig = BuildConfig(), BVHBuildMetrics* buildMetrics = nullptr);

	/* \brief Builds a compressed wide BVH from a list of primitives
	 *
	 * \param primitives The primitives the BVH will be built from (AABB or Triangle)
	 * \param primCount The number of primitives
	 * \param buildConfig The build configuration
	 * \param buildMetrics The build metrics. If different from nullptr, kernel execution times will be measured which results in a slower build
	 *
	 * \returns A pointer to the device instance of the newly created binary BVH
	 */
	template <typename PrimT>
	BVH8 BuildBVH8(PrimT* primitives, uint32_t primCount, BuildConfig buildConfig = BuildConfig(), BVHBuildMetrics* buildMetrics = nullptr);


	/* \brief Transfer BVH to CPU
	 *
	 * \param binaryBVH The binary BVH to be transferred
	 *
	 * \returns A CPU BVH instance
	 */
	BVH2 ToHost(BVH2 deviceBvh);

	/* \brief Transfer BVH to CPU
	 *
	 * \param wideBVH The compressed wide BVH to be transferred
	 *
	 * \returns A CPU BVH instance
	 */
	BVH8 ToHost(BVH8 deviceBvh);

	/*
	 * \brief Free the host instance of the binary BVH
	 */
	void FreeHostBVH(BVH2 hostBvh);

	/*
	 * \brief Free the host instance of the compressed wide BVH
	 */
	void FreeHostBVH(BVH8 hostBvh);

	/*
	 * \brief Free the device instance of the binary BVH
	 */
	void FreeDeviceBVH(BVH2 deviceBvh);

	/*
	 * \brief Free the device instance of the compressed wide BVH
	 */
	void FreeDeviceBVH(BVH8 deviceBvh);

}