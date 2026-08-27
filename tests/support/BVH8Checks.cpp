#include "BVHChecks.h"

#include <cmath>
#include <string>

#include "BVH8Decode.h"

namespace NXB::Test
{
	namespace
	{
		float Axis(const float3& v, uint32_t axis)
		{
			return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
		}

		/*
		 * Walks the wide hierarchy, checking each node against the exact bounds of the
		 * subtree below it. Visit returns those exact bounds, built bottom up from the
		 * primitives, so a parent can be compared with what it actually contains rather
		 * than with what the encoder claimed.
		 */
		struct Bvh8Walk
		{
			const BVH8::Host& bvh;
			const std::vector<AABB>& primBounds;
			ValidationResult& result;

			std::vector<DecodedNode> decoded;
			std::vector<bool> nodeVisited;
			std::vector<uint32_t> primSlotRefCount;

			uint32_t containmentErrors = 0;
			uint32_t tightnessErrors = 0;
			uint32_t gridErrors = 0;
			uint32_t metaErrors = 0;
			uint32_t originErrors = 0;
			uint32_t sentinelErrors = 0;
			bool aborted = false;

			void Abort(const std::string& message)
			{
				if (!aborted)
					result.Add(message);
				aborted = true;
			}

			AABB Visit(uint32_t nodeIdx);
			void CheckSlotBounds(const DecodedNode& node, const DecodedChild& child, const AABB& exact);
			void CheckGrid(const DecodedNode& node, const AABB& exact);
		};

		void Bvh8Walk::CheckSlotBounds(const DecodedNode& node, const DecodedChild& child, const AABB& exact)
		{
			if (!Contains(child.bounds, exact))
			{
				containmentErrors++;
				return;
			}

			// Quantizing rounds the box outwards by at most one cell per side. Two cells
			// of slack keeps float noise out of it while still catching a slot whose
			// quantized bounds have drifted off the child entirely.
			const float3 cell = node.CellSize();
			for (uint32_t axis = 0; axis < 3; axis++)
			{
				const float slack = 2.0f * Axis(cell, axis) + 1e-4f;
				if (Axis(child.bounds.bMin, axis) < Axis(exact.bMin, axis) - slack
					|| Axis(child.bounds.bMax, axis) > Axis(exact.bMax, axis) + slack)
				{
					tightnessErrors++;
					return;
				}
			}
		}

		void Bvh8Walk::CheckGrid(const DecodedNode& node, const AABB& exact)
		{
			const float3 cell = node.CellSize();

			for (uint32_t axis = 0; axis < 3; axis++)
			{
				const float origin = Axis(node.p, axis);
				const float lo = Axis(exact.bMin, axis);
				const float extent = Axis(exact.bMax, axis) - lo;
				const float epsilon = 1e-4f * std::fmax(1.0f, std::fabs(lo));

				// The grid origin is the corner of the box the node covers
				if (std::fabs(origin - lo) > epsilon)
					originErrors++;

				if (extent <= 0.0f)
					continue;

				// e comes from CeilLog2(extent / 255), so 255 cells span the node and,
				// because CeilLog2 rounds up to a power of two, no more than twice it
				const float span = 255.0f * Axis(cell, axis);
				if (span < extent - epsilon || span > 2.0f * extent + epsilon)
					gridErrors++;
			}
		}

		AABB Bvh8Walk::Visit(uint32_t nodeIdx)
		{
			AABB exact;
			exact.Clear();

			if (aborted)
				return exact;

			if (nodeIdx >= bvh.nodes.size())
			{
				Abort("node index " + std::to_string(nodeIdx) + " out of range");
				return exact;
			}
			if (nodeVisited[nodeIdx])
			{
				Abort("node " + std::to_string(nodeIdx) + " reached twice (cycle or shared subtree)");
				return exact;
			}
			nodeVisited[nodeIdx] = true;

			const DecodedNode& node = decoded[nodeIdx];

			for (uint32_t slot = 0; slot < 8; slot++)
			{
				if (SlotOccupied(node, slot))
					continue;

				// An unused slot has to decode as an empty box, or a traversal that ignores
				// meta would intersect uninitialized memory
				for (uint32_t axis = 0; axis < 3; axis++)
				{
					if (node.qlo[axis][slot] != 0xff || node.qhi[axis][slot] != 0x00)
						sentinelErrors++;
				}
			}

			for (const DecodedChild& child : node.children)
			{
				AABB childExact;
				childExact.Clear();

				if (child.isInner)
				{
					// Inner slots carry 001 in the high bits and 24 + slot in the low five
					if ((node.meta[child.slot] >> 5) != 1 || (node.meta[child.slot] & 0x1f) != 24 + child.slot)
						metaErrors++;

					childExact = Visit(child.index);
					if (aborted)
						return exact;
				}
				else
				{
					if (child.primCount == 0)
					{
						metaErrors++;
						continue;
					}
					if (child.index + child.primCount > bvh.primCount)
					{
						Abort("leaf primitive slot " + std::to_string(child.index) + " out of range");
						return exact;
					}

					for (uint32_t i = 0; i < child.primCount; i++)
					{
						const uint32_t primSlot = child.index + i;
						primSlotRefCount[primSlot]++;

						const uint32_t primIdx = bvh.primIdx[primSlot];
						if (primIdx >= primBounds.size())
						{
							Abort("primIdx[" + std::to_string(primSlot) + "] out of range");
							return exact;
						}
						childExact.Grow(primBounds[primIdx]);
					}
				}

				CheckSlotBounds(node, child, childExact);
				exact.Grow(childExact);
			}

			if (node.children.empty())
			{
				Abort("node " + std::to_string(nodeIdx) + " has no occupied slots");
				return exact;
			}

			CheckGrid(node, exact);
			return exact;
		}
	}


	std::vector<AABB> PrimBounds(const std::vector<Triangle>& prims)
	{
		std::vector<AABB> bounds;
		bounds.reserve(prims.size());
		for (const Triangle& tri : prims)
			bounds.push_back(tri.Bounds());
		return bounds;
	}

	std::vector<AABB> PrimBounds(const std::vector<AABB>& prims)
	{
		return prims;
	}


	ValidationResult ValidateBVH8(const BVH8::Host& hostBvh, const std::vector<AABB>& primBounds)
	{
		ValidationResult result;

		if (hostBvh.nodes.empty() || hostBvh.primIdx.empty())
		{
			result.Add("BVH8 handle is empty");
			return result;
		}
		if (hostBvh.primCount != primBounds.size())
		{
			result.Add("BVH8 primCount does not match the scene it was built from");
			return result;
		}

		Bvh8Walk walk{ hostBvh, primBounds, result };
		const uint32_t nodeCount = (uint32_t)hostBvh.nodes.size();

		walk.decoded.reserve(nodeCount);
		for (uint32_t i = 0; i < nodeCount; i++)
			walk.decoded.push_back(DecodeNode(hostBvh.nodes[i]));

		walk.nodeVisited.assign(nodeCount, false);
		walk.primSlotRefCount.assign(hostBvh.primCount, 0);

		const AABB exact = walk.Visit(0);
		if (walk.aborted)
			return result;

		if (walk.containmentErrors)
			result.Add(std::to_string(walk.containmentErrors) + " quantized child boxes do not contain their child");
		if (walk.tightnessErrors)
			result.Add(std::to_string(walk.tightnessErrors) + " quantized child boxes are more than a cell too large");
		if (walk.gridErrors)
			result.Add(std::to_string(walk.gridErrors) + " node grids do not span their node in 255 cells");
		if (walk.originErrors)
			result.Add(std::to_string(walk.originErrors) + " node grid origins are not at the node's lower corner");
		if (walk.metaErrors)
			result.Add(std::to_string(walk.metaErrors) + " slots have a malformed meta field");
		if (walk.sentinelErrors)
			result.Add(std::to_string(walk.sentinelErrors) + " unused slots do not decode as an empty box");

		uint32_t reached = 0;
		for (bool visited : walk.nodeVisited)
			reached += visited ? 1 : 0;
		if (reached != nodeCount)
			result.Add(std::to_string(reached) + " nodes reached, " + std::to_string(nodeCount) + " reported");

		uint32_t unusedSlots = 0;
		uint32_t duplicatedSlots = 0;
		for (uint32_t count : walk.primSlotRefCount)
		{
			if (count == 0)
				unusedSlots++;
			else if (count > 1)
				duplicatedSlots++;
		}
		if (unusedSlots)
			result.Add(std::to_string(unusedSlots) + " primitive slots never referenced by a leaf");
		if (duplicatedSlots)
			result.Add(std::to_string(duplicatedSlots) + " primitive slots referenced by more than one leaf");

		result.Append(ValidateSceneBounds(hostBvh.bounds, exact));

		return result;
	}
}
