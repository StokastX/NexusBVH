#include "BVHChecks.h"

#include <cmath>

namespace NXB::Test
{
	bool Contains(const AABB& parent, const AABB& child)
	{
		const float epsilon = 1e-4f;
		return child.bMin.x >= parent.bMin.x - epsilon && child.bMax.x <= parent.bMax.x + epsilon
			&& child.bMin.y >= parent.bMin.y - epsilon && child.bMax.y <= parent.bMax.y + epsilon
			&& child.bMin.z >= parent.bMin.z - epsilon && child.bMax.z <= parent.bMax.z + epsilon;
	}


	AABB ReferenceSceneBounds(const std::vector<Triangle>& prims)
	{
		AABB bounds;
		bounds.Clear();
		for (const Triangle& tri : prims)
			bounds.Grow(tri.Bounds());
		return bounds;
	}

	AABB ReferenceSceneBounds(const std::vector<AABB>& prims)
	{
		AABB bounds;
		bounds.Clear();
		for (const AABB& box : prims)
			bounds.Grow(box);
		return bounds;
	}


	ValidationResult ValidateBVH2(const BVH2& hostBvh)
	{
		ValidationResult result;

		const uint32_t nodeCount = hostBvh.nodeCount;
		const uint32_t primCount = hostBvh.primCount;

		if (nodeCount != primCount * 2 - 1)
		{
			result.Add("node count is " + std::to_string(nodeCount) + ", expected " + std::to_string(primCount * 2 - 1));
			return result;
		}

		std::vector<uint32_t> primRefCount(primCount, 0);
		std::vector<bool> visited(nodeCount, false);
		std::vector<uint32_t> stack;

		uint32_t innerCount = 0;
		uint32_t leafCount = 0;
		uint32_t containmentErrors = 0;

		// The root is the last node: BuildBVH8 seeds its work list with nodeCount - 1
		stack.push_back(nodeCount - 1);

		while (!stack.empty())
		{
			uint32_t nodeIdx = stack.back();
			stack.pop_back();

			if (nodeIdx >= nodeCount)
			{
				result.Add("node index " + std::to_string(nodeIdx) + " out of range");
				return result;
			}
			if (visited[nodeIdx])
			{
				result.Add("node " + std::to_string(nodeIdx) + " reached twice (cycle or shared subtree)");
				return result;
			}
			visited[nodeIdx] = true;

			const BVH2::Node& node = hostBvh.nodes[nodeIdx];

			if (node.leftChild == INVALID_IDX)
			{
				leafCount++;
				if (node.rightChild >= primCount)
				{
					result.Add("leaf primitive index " + std::to_string(node.rightChild) + " out of range");
					return result;
				}
				primRefCount[node.rightChild]++;
				continue;
			}

			innerCount++;
			if (node.leftChild >= nodeCount || node.rightChild >= nodeCount)
			{
				result.Add("child index out of range at node " + std::to_string(nodeIdx));
				return result;
			}

			const uint32_t children[2] = { node.leftChild, node.rightChild };
			for (uint32_t child : children)
			{
				if (!Contains(node.bounds, hostBvh.nodes[child].bounds))
					containmentErrors++;
				stack.push_back(child);
			}
		}

		if (containmentErrors)
			result.Add(std::to_string(containmentErrors) + " child boxes not contained in their parent");

		if (leafCount != primCount)
			result.Add(std::to_string(leafCount) + " leaves for " + std::to_string(primCount) + " primitives");

		if (innerCount + leafCount != nodeCount)
			result.Add(std::to_string(innerCount + leafCount) + " nodes reached, " + std::to_string(nodeCount) + " allocated");

		uint32_t unreferenced = 0;
		uint32_t duplicated = 0;
		for (uint32_t i = 0; i < primCount; ++i)
		{
			if (primRefCount[i] == 0)
				unreferenced++;
			else if (primRefCount[i] > 1)
				duplicated++;
		}
		if (unreferenced)
			result.Add(std::to_string(unreferenced) + " primitives never referenced");
		if (duplicated)
			result.Add(std::to_string(duplicated) + " primitives referenced more than once");

		if (!std::isfinite(hostBvh.bounds.bMin.x) || hostBvh.bounds.bMin.x > hostBvh.bounds.bMax.x)
			result.Add("scene bounds were not read back");

		return result;
	}


	ValidationResult ValidateSceneBounds(const AABB& reported, const AABB& expected)
	{
		ValidationResult result;

		const float epsilon = 1e-3f;
		bool matches = std::fabs(reported.bMin.x - expected.bMin.x) < epsilon
			&& std::fabs(reported.bMin.y - expected.bMin.y) < epsilon
			&& std::fabs(reported.bMin.z - expected.bMin.z) < epsilon
			&& std::fabs(reported.bMax.x - expected.bMax.x) < epsilon
			&& std::fabs(reported.bMax.y - expected.bMax.y) < epsilon
			&& std::fabs(reported.bMax.z - expected.bMax.z) < epsilon;

		if (!matches)
			result.Add("scene bounds do not match the primitives they were built from");

		return result;
	}


	ValidationResult ValidatePrimIdxPermutation(const std::vector<uint32_t>& primIdx, uint32_t primCount)
	{
		ValidationResult result;

		std::vector<uint32_t> refCount(primCount, 0);
		uint32_t outOfRange = 0;
		for (uint32_t idx : primIdx)
		{
			if (idx >= primCount)
				outOfRange++;
			else
				refCount[idx]++;
		}

		uint32_t unreferenced = 0;
		uint32_t duplicated = 0;
		for (uint32_t i = 0; i < primCount; ++i)
		{
			if (refCount[i] == 0)
				unreferenced++;
			else if (refCount[i] > 1)
				duplicated++;
		}

		if (outOfRange)
			result.Add(std::to_string(outOfRange) + " primitive indices out of range");
		if (unreferenced)
			result.Add(std::to_string(unreferenced) + " primitives missing from the index list");
		if (duplicated)
			result.Add(std::to_string(duplicated) + " primitives duplicated in the index list");

		return result;
	}
}
