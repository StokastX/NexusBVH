#include <iostream>
#include "NXB/BVHBuilder.h"
#include "CudaUtils.h"
#include <vector>

int main(void)
{
	NXB::Triangle t3(
		make_float3(40.0f, -40.0f, 40.0f),
		make_float3(40.0f, -40.0f, -40.0f),
		make_float3(-40.0f, -40.0f, -40.0f)
	);

	NXB::Triangle t1(
		make_float3(-40.0f, -40.0f, 40.0f),
		make_float3(40.0f, -40.0f, 40.0f),
		make_float3(-40.0f, -40.0f, -40.0f)
	);

	NXB::Triangle t2(
		make_float3(40.0f, 40.0f, 40.0f),
		make_float3(40.0f, 40.0f, -40.0f),
		make_float3(-40.0f, 40.0f, -40.0f)
	);

	NXB::Triangle t0(
		make_float3(-40.0f, 40.0f, 40.0f),
		make_float3(40.0f, 40.0f, 40.0f),
		make_float3(-40.0f, 40.0f, -40.0f)
	);

	NXB::Triangle triangles[4] = { t0, t1, t2, t3 };

	NXB::BVHBuilder bvhBuilder;

	NXB::Triangle* dTriangles = CudaMemory::Allocate<NXB::Triangle>(4);
	CudaMemory::Copy<NXB::Triangle>(dTriangles, triangles, 4, cudaMemcpyHostToDevice);

	std::cout << "========== Building BVH ==========" << std::endl << std::endl;
	NXB::BVHBuildMetrics buildMetrics;
	NXB::BVH2* dBvh = bvhBuilder.BuildBinary(dTriangles, 4, &buildMetrics);

	NXB::BVH2 bvh;
	CudaMemory::Copy(&bvh, dBvh, 1, cudaMemcpyDeviceToHost);

	std::cout << "Primitive count: " << bvh.primCount << std::endl;
	std::cout << "Node count: " << bvh.nodeCount << std::endl << std::endl;

	std::cout << "---------- TIMINGS ----------" << std::endl << std::endl;
	std::cout << "Triangle bounds: " << buildMetrics.computeTriangleBoundsTime << " ms" << std::endl;
	std::cout << "Mesh bounds: " << buildMetrics.computeSceneBoundsTime << " ms" << std::endl;
	std::cout << "Morton codes: " << buildMetrics.computeMortonCodesTime << " ms" << std::endl;
	std::cout << "Radix sort: " << buildMetrics.radixSortTime << " ms" << std::endl;
	std::cout << "Clusters init: " << buildMetrics.initClustersTime << " ms" << std::endl;
	std::cout << "Binary BVH building: " << buildMetrics.bvhBuildTime << " ms" << std::endl;
	std::cout << "Total build time: " << buildMetrics.totalTime << " ms" << std::endl << std::endl;

	std::cout << "========== Building done ==========" << std::endl << std::endl;

	bvhBuilder.FreeBVH(dBvh);

	CudaMemory::Free(dTriangles);

	return EXIT_SUCCESS;
}