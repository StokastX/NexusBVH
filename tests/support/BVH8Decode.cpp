#include "BVH8Decode.h"

#include <cmath>
#include <cstring>

namespace NXB::Test
{
	namespace
	{
		/*
		 * Byte offsets of BVH8::NodeExplicit inside the 80 byte node, spelled out so the
		 * decoder does not inherit the layout from the header it is checking.
		 */
		constexpr size_t offsetP = 0;
		constexpr size_t offsetE = 12;
		constexpr size_t offsetImask = 15;
		constexpr size_t offsetChildBaseIdx = 16;
		constexpr size_t offsetPrimBaseIdx = 20;
		constexpr size_t offsetMeta = 24;
		constexpr size_t offsetQlo = 32;
		constexpr size_t offsetQhi = 56;

		static_assert(sizeof(BVH8::Node) == 80, "BVH8::Node is no longer 80 bytes");
		static_assert(sizeof(BVH8::NodeExplicit) == sizeof(BVH8::Node),
			"BVH8::Node and BVH8::NodeExplicit must stay byte identical");

		template <typename T>
		T ReadAt(const unsigned char* bytes, size_t offset)
		{
			T value;
			std::memcpy(&value, bytes + offset, sizeof(T));
			return value;
		}

		uint32_t PopCount(uint32_t x)
		{
			uint32_t count = 0;
			for (; x; x &= x - 1)
				count++;
			return count;
		}
	}


	float3 DecodedNode::CellSize() const
	{
		return make_float3(std::ldexp(1.0f, (int32_t)e[0] - 127),
			std::ldexp(1.0f, (int32_t)e[1] - 127),
			std::ldexp(1.0f, (int32_t)e[2] - 127));
	}


	bool SlotOccupied(const DecodedNode& node, uint32_t slot)
	{
		return node.meta[slot] != 0;
	}


	DecodedNode DecodeNode(const BVH8::Node& node)
	{
		const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&node);

		DecodedNode decoded = {};
		decoded.p = ReadAt<float3>(bytes, offsetP);
		decoded.imask = ReadAt<uint8_t>(bytes, offsetImask);
		decoded.childBaseIdx = ReadAt<uint32_t>(bytes, offsetChildBaseIdx);
		decoded.primBaseIdx = ReadAt<uint32_t>(bytes, offsetPrimBaseIdx);

		for (uint32_t axis = 0; axis < 3; axis++)
			decoded.e[axis] = ReadAt<uint8_t>(bytes, offsetE + axis);

		for (uint32_t slot = 0; slot < 8; slot++)
			decoded.meta[slot] = ReadAt<uint8_t>(bytes, offsetMeta + slot);

		for (uint32_t axis = 0; axis < 3; axis++)
		{
			for (uint32_t slot = 0; slot < 8; slot++)
			{
				decoded.qlo[axis][slot] = ReadAt<uint8_t>(bytes, offsetQlo + axis * 8 + slot);
				decoded.qhi[axis][slot] = ReadAt<uint8_t>(bytes, offsetQhi + axis * 8 + slot);
			}
		}

		const float3 cell = decoded.CellSize();
		const float origin[3] = { decoded.p.x, decoded.p.y, decoded.p.z };
		const float scale[3] = { cell.x, cell.y, cell.z };

		for (uint32_t slot = 0; slot < 8; slot++)
		{
			if (!SlotOccupied(decoded, slot))
				continue;

			DecodedChild child = {};
			child.slot = slot;

			float lo[3], hi[3];
			for (uint32_t axis = 0; axis < 3; axis++)
			{
				lo[axis] = origin[axis] + decoded.qlo[axis][slot] * scale[axis];
				hi[axis] = origin[axis] + decoded.qhi[axis][slot] * scale[axis];
			}
			child.bounds.bMin = make_float3(lo[0], lo[1], lo[2]);
			child.bounds.bMax = make_float3(hi[0], hi[1], hi[2]);

			child.isInner = (decoded.imask >> slot) & 1;
			if (child.isInner)
			{
				// Children are stored in slot order, so the offset from the base is the
				// number of inner slots below this one
				child.index = decoded.childBaseIdx + PopCount(decoded.imask & ((1u << slot) - 1));
				child.primCount = 0;
			}
			else
			{
				// High 3 bits of meta hold the primitive count in unary, low 5 the offset
				child.index = decoded.primBaseIdx + (decoded.meta[slot] & 0x1f);
				child.primCount = PopCount(decoded.meta[slot] >> 5);
			}

			decoded.children.push_back(child);
		}

		return decoded;
	}
}
