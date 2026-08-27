#pragma once

#include <cuda_runtime.h>

#include "BVH.h"

namespace NXB
{
	/*
	 * \brief Surface Area Heuristic cost of a finished BVH
	 *
	 * The cost model is the usual one: C_T per traversal step and C_I per
	 * ray-primitive intersection, each node weighted by the probability a random ray
	 * that hits the root also hits it, i.e. the ratio of surface areas.
	 *
	 * This is an analysis of a tree, not a step of building one, so it is a call the
	 * caller makes when a number is wanted rather than something a build produces on
	 * the side. It launches a kernel over every node and synchronizes to hand the
	 * result back, which is why it is not free and why no build path runs it.
	 *
	 * \param stream The stream the kernel and its scratch allocation are issued on
	 * \param pool   The pool the scratch allocation is taken from, as in BuildConfig
	 *
	 * \returns The SAH cost, or 0 for an empty BVH
	 */
	float ComputeSAHCost(const BVH2& bvh, cudaStream_t stream = 0, cudaMemPool_t pool = nullptr);

	float ComputeSAHCost(const BVH8& bvh, cudaStream_t stream = 0, cudaMemPool_t pool = nullptr);
}
