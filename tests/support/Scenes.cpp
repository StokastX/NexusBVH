#include "Scenes.h"

#include <random>

namespace NXB::Test
{
	std::vector<Triangle> GenerateTriangles(uint32_t primCount, uint32_t gridSize)
	{
		std::vector<Triangle> triangles(primCount);
		const float cellSize = 10.0f / gridSize;

		// For reproductibility
		std::mt19937 gen(12345);
		std::uniform_int_distribution<uint32_t> cellDist(0, gridSize - 1);
		std::uniform_real_distribution<float> offsetDist(0.1f * cellSize, 0.9f * cellSize);
		std::uniform_real_distribution<float> edgeDist(-0.4f * cellSize, 0.4f * cellSize);

		for (uint32_t i = 0; i < primCount; ++i)
		{
			float baseX = cellDist(gen) * cellSize;
			float baseY = cellDist(gen) * cellSize;
			float baseZ = cellDist(gen) * cellSize;

			// First vertex is randomly positioned within the cell
			float3 v0 = { baseX + offsetDist(gen), baseY + offsetDist(gen), baseZ + offsetDist(gen) };

			// Other vertices are offset relative to v0 to ensure valid triangles
			float3 v1 = { v0.x + edgeDist(gen), v0.y + edgeDist(gen), v0.z + edgeDist(gen) };
			float3 v2 = { v0.x + edgeDist(gen), v0.y + edgeDist(gen), v0.z + edgeDist(gen) };

			triangles[i] = { v0, v1, v2 };
		}
		return triangles;
	}

	std::vector<AABB> GenerateAABBs(uint32_t primCount, uint32_t gridSize)
	{
		std::vector<Triangle> triangles = GenerateTriangles(primCount, gridSize);

		std::vector<AABB> boxes(primCount);
		for (uint32_t i = 0; i < primCount; ++i)
			boxes[i] = triangles[i].Bounds();

		return boxes;
	}
}
