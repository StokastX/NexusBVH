#pragma once

#include <string>
#include <vector>

#include "NXB/AABB.h"
#include "NXB/BVH.h"
#include "NXB/Triangle.h"

namespace NXB::Test
{
	/*
	 * Structural checks over a built hierarchy.
	 *
	 * These are deliberately framework agnostic: they collect what went wrong and hand
	 * it back, and the test case decides how to report it. Keeping them free of any
	 * CHECK macro is what lets them be reused from elsewhere -- the traversal tests
	 * will want ReferenceSceneBounds and the BVH2 walker just as much as these do.
	 */
	struct ValidationResult
	{
		std::vector<std::string> errors;

		bool Ok() const { return errors.empty(); }
		void Add(const std::string& message) { errors.push_back(message); }
		void Append(const ValidationResult& other)
		{
			errors.insert(errors.end(), other.errors.begin(), other.errors.end());
		}
	};


	// Parent bounds are the merge of the child bounds, so containment is exact up to
	// the rounding of the merge itself
	bool Contains(const AABB& parent, const AABB& child);

	AABB ReferenceSceneBounds(const std::vector<Triangle>& prims);
	AABB ReferenceSceneBounds(const std::vector<AABB>& prims);

	// Per primitive bounds, so the BVH8 walk can compare a node against what it contains
	// without caring which PrimT the tree was built from
	std::vector<AABB> PrimBounds(const std::vector<Triangle>& prims);
	std::vector<AABB> PrimBounds(const std::vector<AABB>& prims);

	/*
	 * Walks the hierarchy from the root and checks that it is a well formed binary tree
	 * covering every primitive exactly once.
	 *
	 * There is no way to spot a malformed hierarchy from the outside: a BVH2 whose
	 * indices are scrambled still has the right node count, and a SAH cost is happy to
	 * be computed over a broken tree. So the walk checks every index in range, every
	 * parent box containing its children, every primitive referenced exactly once, and
	 * no node reached twice.
	 *
	 * Takes a host side BVH2, i.e. the result of BVH2::ToHost.
	 */
	ValidationResult ValidateBVH2(const BVH2::Host& hostBvh);

	// Checks the scene bounds the builder reported against a host computed reference
	ValidationResult ValidateSceneBounds(const AABB& reported, const AABB& expected);

	/*
	 * Checks that a primitive index list references every primitive in [0, primCount)
	 * exactly once.
	 */
	ValidationResult ValidatePrimIdxPermutation(const std::vector<uint32_t>& primIdx, uint32_t primCount);

	/*
	 * Walks the wide hierarchy from the root, decoding each node with the independent
	 * decoder in BVH8Decode.h.
	 *
	 * Checks every node reachable exactly once, meta and imask agreeing, unused slots
	 * decoding as empty, every primitive slot referenced by exactly one leaf, and -- the
	 * part a node count alone cannot see -- that each quantized child box contains the
	 * true bounds of what sits below it, without being more than a quantization cell
	 * larger than it, and that each node grid spans its node in 255 cells.
	 *
	 * Takes a host side BVH8, i.e. the result of BVH8::ToHost.
	 */
	ValidationResult ValidateBVH8(const BVH8::Host& hostBvh, const std::vector<AABB>& primBounds);
}
