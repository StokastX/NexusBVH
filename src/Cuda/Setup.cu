#include "Setup.h"

#include <cub/device/device_radix_sort.cuh>
#include <device_launch_parameters.h>

#include "CudaUtils.h"
#include "BuilderUtils.h"

__global__ void NXB::ComputePrimBounds(BuildState buildState, Triangle* primitives)
{
	uint32_t primIdx = blockDim.x * blockIdx.x + threadIdx.x;

	if (primIdx >= buildState.primCount)
		return;

	Triangle triangle = primitives[primIdx];

	buildState.primBounds[primIdx] = triangle.Bounds();
}

__global__ void NXB::ComputeSceneBounds(BuildState buildState)
{
	uint32_t primIdx = blockDim.x * blockIdx.x + threadIdx.x;

	// Make sure to initialize scene bounds
	if (primIdx == 0)
		buildState.sceneBounds->Clear();

	__syncthreads();

	if (primIdx >= buildState.primCount)
		return;

	// Shared bounds to parallelize atomic operations across thread blocks before updating the global scene bounds
	__shared__ AABB sharedBounds;

	// Clear shared AABB
	if (threadIdx.x == 0)
		sharedBounds.Clear();

	__syncthreads();

	AABB primBounds = buildState.primBounds[primIdx];
	AtomicGrow(&sharedBounds, primBounds);

	__syncthreads();

	// Scene bounds update
	if (threadIdx.x == 0)
		AtomicGrow(buildState.sceneBounds, sharedBounds);
}

__global__ void NXB::ComputeMortonCodes(BuildState buildState)
{
	uint32_t primIdx = blockDim.x * blockIdx.x + threadIdx.x;

	if (primIdx > buildState.primCount)
		return;

	AABB primBounds = buildState.primBounds[primIdx];
	AABB *sceneBounds = buildState.sceneBounds;
	float3 centroid = primBounds.Centroid();

	// TODO: change division into mutliplication? (store the inverse scene bounds)
	uint64_t mortonCode = MortonCode((centroid - sceneBounds->bMin) / (sceneBounds->bMax - sceneBounds->bMin));
	buildState.mortonCodes[primIdx] = mortonCode;

	// Initialize primitive indices as well
	buildState.primIdx[primIdx] = primIdx;
}

void NXB::RadixSort(BuildState buildState)
{
	using byte = unsigned char;

	size_t tempStorageBytes = 0;
	void* tempStorage = nullptr;

	uint64_t* mortonCodesSorted = CudaMemory::AllocAsync<uint64_t>(buildState.primCount);
	uint32_t* primIdxSorted = CudaMemory::AllocAsync<uint32_t>(buildState.primCount);

	// Get the temporary storage size necessary to perform radix sorting
	cub::DeviceRadixSort::SortPairs(
		tempStorage,	// NULL
		tempStorageBytes,
		buildState.mortonCodes,
		mortonCodesSorted,
		buildState.primIdx,
		primIdxSorted,
		buildState.primCount,
		0,
		64
	);

	tempStorage = CudaMemory::AllocAsync<byte>(tempStorageBytes);

	// Perform radix sorting
	cub::DeviceRadixSort::SortPairs(
		tempStorage,
		tempStorageBytes,
		buildState.mortonCodes,
		mortonCodesSorted,
		buildState.primIdx,
		primIdxSorted,
		buildState.primCount,
		0,
		64
	);

	//CudaMemory::FreeAsync(buildState.mortonCodes);
	//CudaMemory::FreeAsync(buildState.primIdx);

	buildState.mortonCodes = mortonCodesSorted;
	buildState.primIdx = primIdxSorted;

	uint64_t hostMC[4];
	CudaMemory::Copy<uint64_t>(hostMC, mortonCodesSorted, 4, cudaMemcpyDeviceToHost);
	std::cout << hostMC[0] << std::endl;
	std::cout << hostMC[1] << std::endl;
	std::cout << hostMC[2] << std::endl;
	std::cout << hostMC[3] << std::endl;
}
