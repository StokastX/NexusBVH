#pragma once

#include <cassert>
#include <math.h>

#include "AABB.h"
#include "BVH.h"

/*
 * Ray traversal of a BVH built by this library.
 *
 * Opt in: nothing else in NXB includes this header, and the builder does not need it.
 * Everything here is a header only template, so a consumer's own kernels can call it --
 * the compiled archive resolves its device symbols at archive time and cannot export
 * them. Include this from a .cu; it also compiles as ordinary host code, which is what
 * the tests traverse with.
 *
 * The library owns node traversal and nothing else. Primitive intersection is a callback,
 * so the ray type, the hit record and the primitive test stay yours. Two level traversal
 * (a TLAS over instance bounds, built with BuildBVH2<AABB>) is the same callback calling
 * TraverseBVH2 again on the instance's own BVH -- see the example at the bottom.
 */

namespace NXB
{
	/*
	 * Distance reported for a box the ray does not reach, and the value to seed tMax with
	 * for a ray with no far limit. Finite rather than INFINITY so that it survives the
	 * multiplications in the slab test without producing NaN.
	 */
	inline constexpr float RayMiss = 1e30f;

	/* \brief Slab test
	 *
	 * \param invDir Component wise reciprocal of the ray direction. A zero direction
	 *        component gives an infinity, which is handled: an origin strictly inside that
	 *        slab produces -inf and +inf, leaving the axis unbounded, which is what a ray
	 *        parallel to the slab should do.
	 *
	 *        The exception is an origin sitting exactly on a slab plane, where one term is
	 *        0 * inf = NaN. fminf and fmaxf drop it in favour of the other operand, both
	 *        infinities of the same sign, and the box reports a miss. That case is a tie
	 *        the slab test cannot resolve, and no epsilon fixes it without costing every
	 *        other ray -- offset such rays, or accept it.
	 *
	 * \returns Distance at which the ray enters the box, clamped to [tMin, tMax], or
	 *          RayMiss if it does not overlap the box within that interval. A ray starting
	 *          inside the box reports tMin.
	 */
	__host__ __device__ inline float IntersectAABB(const AABB& box, float3 origin, float3 invDir, float tMin, float tMax)
	{
		float tx1 = (box.bMin.x - origin.x) * invDir.x;
		float tx2 = (box.bMax.x - origin.x) * invDir.x;
		float tEnter = fminf(tx1, tx2);
		float tExit = fmaxf(tx1, tx2);

		float ty1 = (box.bMin.y - origin.y) * invDir.y;
		float ty2 = (box.bMax.y - origin.y) * invDir.y;
		tEnter = fmaxf(tEnter, fminf(ty1, ty2));
		tExit = fminf(tExit, fmaxf(ty1, ty2));

		float tz1 = (box.bMin.z - origin.z) * invDir.z;
		float tz2 = (box.bMax.z - origin.z) * invDir.z;
		tEnter = fmaxf(tEnter, fminf(tz1, tz2));
		tExit = fminf(tExit, fmaxf(tz1, tz2));

		tEnter = fmaxf(tEnter, tMin);
		tExit = fminf(tExit, tMax);

		return tExit >= tEnter ? tEnter : RayMiss;
	}

	/* \brief Traverses a binary BVH, calling leafFn for every leaf the ray reaches
	 *
	 * Children are visited nearest entry distance first, and a child is skipped once its
	 * entry distance passes tMax. Ordering is a performance property and not a guarantee
	 * about the callback: leafFn is called for every leaf that survives the current tMax,
	 * not only for the closest one, so the caller keeps its own hit record.
	 *
	 * \param nodes The node array. bvh.View().nodes, or Host::nodes.data() on the host.
	 * \param rootIdx Where the root sits in that array, i.e. RootIdx(). Passing InvalidIdx
	 *        (an empty BVH) traverses nothing and reports no early out.
	 * \param invDir Component wise reciprocal of the ray direction. See IntersectAABB.
	 * \param tMin Near clip. Distances below it are never reported.
	 * \param tMax Far clip, read before every box test and so the pruning bound. leafFn
	 *        lowers it on a hit; the caller sees the final value.
	 * \param leafFn bool(uint32_t primIdx, float& tMax). Returns false to stop traversal
	 *        immediately, which is how an any hit or shadow query terminates. It may lower
	 *        tMax and must never raise it -- a raised bound reopens subtrees that have
	 *        already been culled, and the result is silently missing intersections.
	 *
	 * \returns false if leafFn stopped the traversal early, true if it ran to completion.
	 *
	 * StackSize bounds the depth this can descend to. The builder puts no bound on tree
	 * depth, so a pathological scene can exceed it; the assert catches that in a debug
	 * build, and a release build silently drops subtrees. Raise it rather than guess.
	 */
	template <uint32_t StackSize = 32, typename LeafFn>
	__host__ __device__ inline bool TraverseBVH2(const BVH2::Node* nodes, uint32_t rootIdx,
		float3 origin, float3 invDir, float tMin, float& tMax, LeafFn leafFn)
	{
		if (rootIdx == InvalidIdx)
			return true;

		uint32_t stack[StackSize];
		uint32_t stackPtr = 0;

		BVH2::Node node = nodes[rootIdx];

		while (true)
		{
			if (node.leftChild == InvalidIdx)
			{
				// Leaf: rightChild is the primitive index
				if (!leafFn(node.rightChild, tMax))
					return false;
			}
			else
			{
				BVH2::Node left = nodes[node.leftChild];
				BVH2::Node right = nodes[node.rightChild];

				float dLeft = IntersectAABB(left.bounds, origin, invDir, tMin, tMax);
				float dRight = IntersectAABB(right.bounds, origin, invDir, tMin, tMax);

				bool leftFirst = dLeft <= dRight;
				float dNear = leftFirst ? dLeft : dRight;
				float dFar = leftFirst ? dRight : dLeft;

				if (dNear != RayMiss)
				{
					if (dFar != RayMiss)
					{
						assert(stackPtr < StackSize && "BVH2 deeper than StackSize");
						stack[stackPtr++] = leftFirst ? node.rightChild : node.leftChild;
					}
					node = leftFirst ? left : right;
					continue;
				}
			}

			// Pop until a node is still worth entering. A node is pushed against the tMax
			// of the moment, and leafFn may have tightened it since, so the test has to be
			// repeated here or a closer hit does not cull what was already queued.
			while (true)
			{
				if (stackPtr == 0)
					return true;

				node = nodes[stack[--stackPtr]];

				if (IntersectAABB(node.bounds, origin, invDir, tMin, tMax) != RayMiss)
					break;
			}
		}
	}

	// Convenience overload over a view handed out by BVH2::View()
	template <uint32_t StackSize = 32, typename LeafFn>
	__host__ __device__ inline bool TraverseBVH2(const BVH2::DeviceView& bvh,
		float3 origin, float3 invDir, float tMin, float& tMax, LeafFn leafFn)
	{
		return TraverseBVH2<StackSize>(bvh.nodes, bvh.RootIdx(), origin, invDir, tMin, tMax, leafFn);
	}

	// Host side overload over a copy pulled back with BVH2::ToHost()
	template <uint32_t StackSize = 32, typename LeafFn>
	inline bool TraverseBVH2(const BVH2::Host& bvh,
		float3 origin, float3 invDir, float tMin, float& tMax, LeafFn leafFn)
	{
		return TraverseBVH2<StackSize>(bvh.nodes.data(), bvh.RootIdx(), origin, invDir, tMin, tMax, leafFn);
	}
}

/*
 * Closest hit:
 *
 *     float tMax = NXB::RayMiss;
 *     uint32_t hitPrim = NXB::InvalidIdx;
 *     NXB::TraverseBVH2(bvh, o, invD, 0.0f, tMax,
 *         [&](uint32_t primIdx, float& tMax)
 *         {
 *             float t;
 *             if (IntersectTriangle(prims[primIdx], o, d, tMax, t))
 *             {
 *                 tMax = t;
 *                 hitPrim = primIdx;
 *             }
 *             return true;
 *         });
 *
 * Any hit, by returning false on the first one:
 *
 *     bool occluded = !NXB::TraverseBVH2(bvh, o, invD, 0.0f, tMax,
 *         [&](uint32_t primIdx, float& tMax) { return !IntersectTriangle(...); });
 *
 * Two level, where the TLAS was built with BuildBVH2<AABB> over instance world bounds.
 * Do not normalize the transformed direction: leaving it unnormalized is what keeps t in
 * the same units on both sides of the transform, so tMax carries across unchanged.
 *
 *     NXB::TraverseBVH2(tlas, o, invD, 0.0f, tMax,
 *         [&](uint32_t instIdx, float& tMax)
 *         {
 *             float3 lo = TransformPoint(instances[instIdx].invTransform, o);
 *             float3 ld = TransformVector(instances[instIdx].invTransform, d);
 *             NXB::TraverseBVH2(blas[instIdx], lo, 1.0f / ld, 0.0f, tMax, leafFn);
 *             return true;
 *         });
 */
