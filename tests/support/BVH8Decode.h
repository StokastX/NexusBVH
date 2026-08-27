#pragma once

#include <cstdint>
#include <vector>

#include "NXB/AABB.h"
#include "NXB/BVH.h"

namespace NXB::Test
{
	/*
	 * Host side decoder for the compressed wide node, following Ylitie et al.
	 *
	 * This reads the 80 raw bytes of a BVH8::Node by offset rather than casting to
	 * BVH8::NodeExplicit, and reimplements the dequantization instead of calling the
	 * builder's InvPow2. Both are deliberate: the decoder is meant to be an independent
	 * oracle, and reusing the encoder's own helpers would hide a bug inside them. It also
	 * means the tests pin the two layouts as byte identical, which nothing else does.
	 */

	struct DecodedChild
	{
		uint32_t slot;

		// Dequantized child box, i.e. what a traversal kernel would test a ray against
		AABB bounds;

		bool isInner;

		// Inner children: index into BVH8::nodes. Leaves: index into BVH8::primIdx, and
		// primCount consecutive entries from there.
		uint32_t index;
		uint32_t primCount;
	};

	struct DecodedNode
	{
		float3 p;
		uint8_t e[3];
		uint8_t imask;
		uint32_t childBaseIdx;
		uint32_t primBaseIdx;
		uint8_t meta[8];

		// Quantized bounds of every slot, occupied or not
		uint8_t qlo[3][8];
		uint8_t qhi[3][8];

		std::vector<DecodedChild> children;

		// Size of one quantization cell per axis, 2^(e - 127). A dequantized box is at
		// most one cell per side larger than the box it encodes.
		float3 CellSize() const;
	};

	DecodedNode DecodeNode(const BVH8::Node& node);

	bool SlotOccupied(const DecodedNode& node, uint32_t slot);
}
