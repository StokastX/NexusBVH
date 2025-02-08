#include <iostream>
#include "NXB/BVHBuilder.h"
#include "CudaUtils.h"
#include <vector>
#include <random>

#define TRIANGLE_COUNT 10000000
#define GRID_SIZE 100

int main(void)
{
	std::vector<NXB::Triangle> triangles;

	// For a completely random triangle generation
	//std::random_device rd;
    //std::mt19937 gen(rd());
	//std::uniform_real_distribution<float> posDist(-10.0f, 10.0f);
    //std::uniform_real_distribution<float> sizeDist(0.1f, 1.0f);

	//for (uint32_t i = 0; i < TRIANGLE_COUNT; ++i) {
	//	float3 center = { posDist(gen), posDist(gen), posDist(gen) };
	//	float size = sizeDist(gen);

	//	float3 v0 = { center.x + size * (posDist(gen) - 0.5f), center.y + size * (posDist(gen) - 0.5f), center.z + size * (posDist(gen) - 0.5f) };
	//	float3 v1 = { center.x + size * (posDist(gen) - 0.5f), center.y + size * (posDist(gen) - 0.5f), center.z + size * (posDist(gen) - 0.5f) };
	//	float3 v2 = { center.x + size * (posDist(gen) - 0.5f), center.y + size * (posDist(gen) - 0.5f), center.z + size * (posDist(gen) - 0.5f) };

	//	triangles.push_back({ v0, v1, v2 });
	//}

	// For a slightly less random triangle generation
	const float cellSize = 10.0f / GRID_SIZE;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> cellDist(0, GRID_SIZE - 1);
    std::uniform_real_distribution<float> offsetDist(0.1f * cellSize, 0.9f * cellSize);

    for (size_t i = 0; i < TRIANGLE_COUNT; ++i) {
        size_t x = cellDist(gen);
        size_t y = cellDist(gen);
        size_t z = cellDist(gen);

        float baseX = x * cellSize;
        float baseY = y * cellSize;
        float baseZ = z * cellSize;

        float3 v0 = {baseX + offsetDist(gen), baseY + offsetDist(gen), baseZ + offsetDist(gen)};
        float3 v1 = {baseX + offsetDist(gen), baseY + offsetDist(gen), baseZ + offsetDist(gen)};
        float3 v2 = {baseX + offsetDist(gen), baseY + offsetDist(gen), baseZ + offsetDist(gen)};

        triangles.push_back({v0, v1, v2});
    }

	NXB::BVHBuilder bvhBuilder;

	NXB::Triangle* dTriangles = CudaMemory::Allocate<NXB::Triangle>(TRIANGLE_COUNT);
	CudaMemory::Copy<NXB::Triangle>(dTriangles, triangles.data(), TRIANGLE_COUNT, cudaMemcpyHostToDevice);

	NXB::BVHBuildMetrics buildMetrics = NXB::BenchmarkBuild(
    [&bvhBuilder](NXB::Triangle* triangles, uint32_t count, NXB::BVHBuildMetrics* metrics) {
        return bvhBuilder.BuildBinary(triangles, count, metrics);
    },
    1, 2,
    (NXB::Triangle*)dTriangles, TRIANGLE_COUNT);

	CudaMemory::Free(dTriangles);

	return EXIT_SUCCESS;

	//std::cout << "========== Building BVH ==========" << std::endl << std::endl;
	//NXB::BVHBuildMetrics buildMetrics;
	//NXB::BVH2* dBvh = bvhBuilder.BuildBinary(dTriangles, TRIANGLE_COUNT, &buildMetrics);

	//NXB::BVH2 bvh;
	//CudaMemory::Copy(&bvh, dBvh, 1, cudaMemcpyDeviceToHost);

	//std::cout << "Primitive count: " << bvh.primCount << std::endl;
	//std::cout << "Node count: " << bvh.nodeCount << std::endl << std::endl;

	//std::cout << "---------- TIMINGS ----------" << std::endl << std::endl;
	//std::cout << "Triangle bounds: " << buildMetrics.computeTriangleBoundsTime << " ms" << std::endl;
	//std::cout << "Mesh bounds: " << buildMetrics.computeSceneBoundsTime << " ms" << std::endl;
	//std::cout << "Morton codes: " << buildMetrics.computeMortonCodesTime << " ms" << std::endl;
	//std::cout << "Radix sort: " << buildMetrics.radixSortTime << " ms" << std::endl;
	//std::cout << "Clusters init: " << buildMetrics.initClustersTime << " ms" << std::endl;
	//std::cout << "Binary BVH building: " << buildMetrics.bvhBuildTime << " ms" << std::endl;
	//std::cout << "Total build time: " << buildMetrics.totalTime << " ms" << std::endl << std::endl;

	//std::cout << "========== Building done ==========" << std::endl << std::endl;

	//NXB::FreeBVH(dBvh);
}