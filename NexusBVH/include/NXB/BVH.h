#pragma once

#include <iostream>
#include <cstdint>
#include "Math/CudaMath.h"
#include "AABB.h"

#define INVALID_IDX (uint32_t)(-1)

namespace NXB
{
	enum struct PrimType: unsigned char
	{
		AABB,
		TRIANGLE
	};

	struct BVH2
	{
		struct Node
		{
			AABB bounds;

			// leftChild = INVALID_IDX if leaf node
			uint32_t leftChild;

			// rightChild = primIdx if leaf node
			uint32_t rightChild;
		};

		Node* nodes;
		uint32_t nodeCount;
		uint32_t primCount;

		// Root bounds
		AABB bounds;
	};

	// Compressed wide BVH (See Ylitie et al.)
	struct BVH8
	{
		using byte = unsigned char;

		struct NodeExplicit
		{
			// Origin point of the local grid
			float3 p;

			// Scale of the grid
			byte e[3];

			// 8-bit mask to indicate which of the children are internal nodes
			byte imask = 0;

			// Index of the first child
			uint32_t childBaseIdx = 0;

			// Index of the first triangle
			uint32_t primBaseIdx = 0;

			// Field encoding the indexing information of every child
			byte meta[8];

			// Quantized origin of the childs' AABBs
			byte qlox[8], qloy[8], qloz[8];

			// Quantized end point of the childs' AABBs
			byte qhix[8], qhiy[8], qhiz[8];

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

		Node* nodes;
		uint32_t nodeCount;

		uint32_t* primIdx;
		uint32_t primCount;

		// Root bounds
		AABB bounds;
	};
}
