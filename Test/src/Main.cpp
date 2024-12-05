#include <iostream>
#include "BVHBuilder.h"
#include "Cuda/CudaUtils.h"

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

	NXB::Triangle* dTriangles = NXB::CudaMemory::Allocate<NXB::Triangle>(4);
	NXB::CudaMemory::Copy<NXB::Triangle>(dTriangles, triangles, 4, cudaMemcpyHostToDevice);

	std::cout << "Building BVH" << std::endl;
	NXB::BVH2* bvh = bvhBuilder.BuildBinary(dTriangles, 4);
	std::cout << "Building done" << std::endl;

	bvhBuilder.FreeBVH(bvh);

	NXB::CudaMemory::Free(dTriangles);

	return EXIT_SUCCESS;
}