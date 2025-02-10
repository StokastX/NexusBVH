#include <iostream>
#include "NXB/BVHBuilder.h"
#include "CudaUtils.h"
#include <vector>
#include <random>

#define TRIANGLE_COUNT 10'000'000
#define GRID_SIZE 1000

int main(void)
{
	std::vector<NXB::Triangle> triangles(TRIANGLE_COUNT);

	// For a completely random triangle generation
    //std::mt19937 gen(12345);
	//std::uniform_real_distribution<float> posDist(-10.0f, 10.0f);
    //std::uniform_real_distribution<float> sizeDist(0.1f, 1.0f);

	//for (uint32_t i = 0; i < TRIANGLE_COUNT; ++i) {
	//	float3 center = { posDist(gen), posDist(gen), posDist(gen) };
	//	float size = sizeDist(gen);

	//	float3 v0 = { center.x + size * (posDist(gen) - 0.5f), center.y + size * (posDist(gen) - 0.5f), center.z + size * (posDist(gen) - 0.5f) };
	//	float3 v1 = { center.x + size * (posDist(gen) - 0.5f), center.y + size * (posDist(gen) - 0.5f), center.z + size * (posDist(gen) - 0.5f) };
	//	float3 v2 = { center.x + size * (posDist(gen) - 0.5f), center.y + size * (posDist(gen) - 0.5f), center.z + size * (posDist(gen) - 0.5f) };

    //  triangles[i] = {v0, v1, v2};
	//}

	// For a slightly less random triangle generation
	const float cellSize = 10.0f / GRID_SIZE;

	// For reproductibility
    std::mt19937 gen(12345);
    std::uniform_int_distribution<size_t> cellDist(0, GRID_SIZE - 1);
    std::uniform_real_distribution<float> offsetDist(0.1f * cellSize, 0.9f * cellSize);
	std::uniform_real_distribution<float> edgeDist(-0.4f * cellSize, 0.4f * cellSize);

	NXB::AABB bounds;
	bounds.Clear();
	for (size_t i = 0; i < TRIANGLE_COUNT; ++i) {
		size_t x = cellDist(gen);
		size_t y = cellDist(gen);
		size_t z = cellDist(gen);

		float baseX = x * cellSize;
		float baseY = y * cellSize;
		float baseZ = z * cellSize;

		// First vertex is randomly positioned within the cell
		float3 v0 = { baseX + offsetDist(gen), baseY + offsetDist(gen), baseZ + offsetDist(gen) };

		// Other vertices are offset relative to v0 to ensure valid triangles
		float3 v1 = { v0.x + edgeDist(gen), v0.y + edgeDist(gen), v0.z + edgeDist(gen) };
		float3 v2 = { v0.x + edgeDist(gen), v0.y + edgeDist(gen), v0.z + edgeDist(gen) };

		// Ensure v1 and v2 are not too close to v0 (avoid degeneracy)
		if (length(v1 - v0) < 0.1f * cellSize) v1.x += 0.2f * cellSize;
		if (length(v2 - v0) < 0.1f * cellSize) v2.y += 0.2f * cellSize;

		triangles[i] = { v0, v1, v2 };
		NXB::AABB primBounds(v0, v1, v2);
		bounds.Grow(primBounds);
	}

	NXB::Triangle* dTriangles = CudaMemory::Allocate<NXB::Triangle>(TRIANGLE_COUNT);
	CudaMemory::Copy<NXB::Triangle>(dTriangles, triangles.data(), TRIANGLE_COUNT, cudaMemcpyHostToDevice);

	NXB::BuildConfig buildConfig;
	buildConfig.prioritizeSpeed = true;

	//NXB::BenchmarkBuild(NXB::BuildBinary<NXB::Triangle>, 10, 100, dTriangles, TRIANGLE_COUNT, buildConfig);

	std::cout << "========== Building BVH ==========" << std::endl << std::endl;
	NXB::BVHBuildMetrics buildMetrics;
	NXB::BVH2 dBvh = NXB::BuildBinary(dTriangles, TRIANGLE_COUNT, buildConfig, &buildMetrics);
	NXB::BVH2 bvh = NXB::ToHost(dBvh);

	double cost = 0.0f;
	for (uint32_t i = 0; i < bvh.nodeCount; ++i)
	{
		NXB::BVH2::Node node = bvh.nodes[i];
		if (node.leftChild == INVALID_IDX)
		{
			cost += 2 * (double)node.bounds.Area() / bvh.bounds.Area();
		}
		else
			cost += 3 * (double)node.bounds.Area() / bvh.bounds.Area();
	}

	std::cout << "Cost: " << cost << std::endl;

	std::cout << "Primitive count: " << bvh.primCount << std::endl;
	std::cout << "Node count: " << bvh.nodeCount << std::endl << std::endl;

	std::cout << "---------- TIMINGS ----------" << std::endl << std::endl;
	std::cout << "Mesh bounds: " << buildMetrics.computeSceneBoundsTime << " ms" << std::endl;
	std::cout << "Morton codes: " << buildMetrics.computeMortonCodesTime << " ms" << std::endl;
	std::cout << "Radix sort: " << buildMetrics.radixSortTime << " ms" << std::endl;
	std::cout << "Binary BVH building: " << buildMetrics.bvhBuildTime << " ms" << std::endl;
	std::cout << "Total build time: " << buildMetrics.totalTime << " ms" << std::endl;

	std::cout << std::endl << "BVH cost: " << buildMetrics.bvhCost << std::endl << std::endl;

	std::cout << std::endl << "========== Building done ==========" << std::endl << std::endl;

	NXB::FreeDeviceBVH(dBvh);
	NXB::FreeHostBVH(bvh);

	CudaMemory::Free(dTriangles);

	return EXIT_SUCCESS;
}