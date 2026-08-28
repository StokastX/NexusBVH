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
	 * The primitive pointers are DEVICE pointers. NXB::DeviceBuffer is the RAII way to
	 * produce them.
	 *
	 * The BVH2 / BVH8 returned OWN their device memory and release it when they go out of
	 * scope, on the stream they were built on and back into the pool they came from. They
	 * are move only. Pass bvh.View() into a kernel, bvh.ToHost() to read the hierarchy
	 * back on the host, and bvh.Release() to opt out and take the arrays over yourself.
	 */

	/* \brief Builds a binary BVH from a list of primitives
	 *
	 * \param primitives The primitives the BVH will be built from (AABB or Triangle)
	 * \param primCount The number of primitives
	 * \param buildConfig The build configuration
	 * \param buildMetrics Per step kernel timings. Filled in when non-null, at a cost of
	 *        roughly 0.03 to 0.05 ms of CUDA event calls -- under 1% of a build at a million
	 *        primitives, closer to 7% at 200k, where a build is short enough for host side
	 *        work to show. The build itself is not serialized to measure it.
	 *        SAH cost is not part of this -- see NXB/BVHCost.h.
	 *
	 * \returns The newly built binary BVH, owning its device memory
	 */
	template <typename PrimT>
	BVH2 BuildBVH2(PrimT* primitives, uint32_t primCount, BuildConfig buildConfig = BuildConfig(), BVHBuildMetrics* buildMetrics = nullptr);

	/* \brief Builds a compressed wide BVH from a list of primitives
	 *
	 * \param primitives The primitives the BVH will be built from (AABB or Triangle)
	 * \param primCount The number of primitives
	 * \param buildConfig The build configuration
	 * \param buildMetrics Per step kernel timings. Filled in when non-null, at a cost of
	 *        roughly 0.03 to 0.05 ms of CUDA event calls -- under 1% of a build at a million
	 *        primitives, closer to 7% at 200k, where a build is short enough for host side
	 *        work to show. The build itself is not serialized to measure it.
	 *        SAH cost is not part of this -- see NXB/BVHCost.h.
	 *
	 * \returns The newly built wide BVH, owning its device memory
	 */
	template <typename PrimT>
	BVH8 BuildBVH8(PrimT* primitives, uint32_t primCount, BuildConfig buildConfig = BuildConfig(), BVHBuildMetrics* buildMetrics = nullptr);


}