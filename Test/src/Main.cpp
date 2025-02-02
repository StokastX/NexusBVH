#include <iostream>
#include "BVHBuilder.h"
#include "Cuda/CudaUtils.h"
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

	NXB::Triangle* dTriangles = NXB::CudaMemory::Allocate<NXB::Triangle>(4);
	NXB::CudaMemory::Copy<NXB::Triangle>(dTriangles, triangles, 4, cudaMemcpyHostToDevice);

	NXB::BVH2* dBvh = bvhBuilder.BuildBinary(dTriangles, 4);

	NXB::BVH2 bvh;
	NXB::CudaMemory::Copy(&bvh, dBvh, 1, cudaMemcpyDeviceToHost);

	std::vector<uint32_t> primIdx(bvh.primCount);
	std::vector<NXB::BVH2::Node> nodes(bvh.primCount * 2);
	NXB::CudaMemory::Copy(primIdx.data(), bvh.primIdx, bvh.primCount, cudaMemcpyDeviceToHost);
	NXB::CudaMemory::Copy(nodes.data(), bvh.nodes, bvh.primCount * 2, cudaMemcpyDeviceToHost);

	bvhBuilder.FreeBVH(dBvh);

	NXB::CudaMemory::Free(dTriangles);

	return EXIT_SUCCESS;
}