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
