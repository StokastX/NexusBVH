#pragma once

#include <cstdint>
#include <type_traits>

#include <vector_types.h>

#include "AABB.h"

namespace NXB
{
	inline constexpr uint32_t InvalidIdx = ~0u;

	struct BVH2
	{
		struct Node
		{
			AABB bounds;

			// leftChild = InvalidIdx if leaf node
			uint32_t leftChild;

			// rightChild = primIdx if leaf node
			uint32_t rightChild;
		};

		/* rief What a kernel receives
		 *
		 * Kernel parameters are bitwise copied into parameter space, so they have to be
		 * trivially copyable -- which an owning type is not. This is the value the owner
		 * will hand out once BVH2 itself owns its memory; the pointer inside it belongs
		 * to that owner and does not outlive it.
		 */
		struct DeviceView
		{
			Node* nodes;
			uint32_t nodeCount;
			uint32_t primCount;

			// Root bounds
			AABB bounds;
		};

		DeviceView View() const { return DeviceView{ nodes, nodeCount, primCount, bounds }; }

		Node* nodes;
		uint32_t nodeCount;
		uint32_t primCount;

		// Root bounds
		AABB bounds;
	};

	static_assert(std::is_trivially_copyable<BVH2::DeviceView>::value,
		"BVH2::DeviceView is passed by value into kernels and must stay trivially copyable");

	// Compressed wide BVH (See Ylitie et al.)
	struct BVH8
	{
		struct NodeExplicit
		{
			// Origin point of the local grid
			float3 p;

			// Scale of the grid
			uint8_t e[3];

			// 8-bit mask to indicate which of the children are internal nodes
			uint8_t imask = 0;

			// Index of the first child
			uint32_t childBaseIdx = 0;

			// Index of the first triangle
			uint32_t primBaseIdx = 0;

			// Field encoding the indexing information of every child
			uint8_t meta[8];

			// Quantized origin of the childs' AABBs
			uint8_t qlox[8], qloy[8], qloz[8];

			// Quantized end point of the childs' AABBs
			uint8_t qhix[8], qhiy[8], qhiz[8];

		};

		struct Node
		{
			// P (12 bytes), e (3 bytes), imask (1 byte)
			float4 p_e_imask;

			// Child base index (4 bytes), triangle base index (4 bytes), meta (8 bytes)
			float4 childidx_tridx_meta;

			// qlox (8 bytes), qloy (8 bytes)
			float4 qlox_qloy;

			// qloz (8 bytes), qlix (8 bytes)
			float4 qloz_qhix;

			// qliy (8 bytes), qliz (8 bytes)
			float4 qhiy_qhiz;
		};

		// See BVH2::DeviceView
		struct DeviceView
		{
			Node* nodes;
			uint32_t nodeCount;

			uint32_t* primIdx;
			uint32_t primCount;

			// Root bounds
			AABB bounds;
		};

		DeviceView View() const { return DeviceView{ nodes, nodeCount, primIdx, primCount, bounds }; }

		Node* nodes;
		uint32_t nodeCount;

		uint32_t* primIdx;
		uint32_t primCount;

		// Root bounds
		AABB bounds;
	};

	static_assert(std::is_trivially_copyable<BVH8::DeviceView>::value,
		"BVH8::DeviceView is passed by value into kernels and must stay trivially copyable");
}
