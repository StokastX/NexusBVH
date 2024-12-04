#pragma once

#include <iostream>
#include "Math/CudaMath.h"
#include "Math/AABB.h"

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

			// Either the index of the left child node, or the first index in the triangles list
			union
			{
				uint32_t leftChild;
				uint32_t firsPrimId;
			};
			// Number of triangles in the node (0 if not leaf)
			uint32_t primCount;
		};

		Node* nodes;
		uint32_t nodeCount;

		uint32_t* primIds;
		uint32_t primCount;
	};

	// Compressed wide BVH (See Ylitie et al.)
	struct BVH8
	{
		struct Node
		{
			using byte = unsigned char;

			// P (12 bytes), e (3 bytes), imask (1 byte)
			float4 p_e_imask;

			// Index of the first child
			uint32_t childBaseIdx = 0;

			// Index of the first triangle
			uint32_t triangleBaseIdx = 0;

			// Field encoding the indexing information of every child
			byte meta[8];

			// Quantized origin of the childs' AABBs
			byte qlox[8], qloy[8], qloz[8];

			// Quantized end point of the childs' AABBs
			byte qhix[8], qhiy[8], qhiz[8];
		};

		Node* nodes;
		uint32_t nodeCount;

		uint32_t* primIds;
		uint32_t primCount;
	};
}
