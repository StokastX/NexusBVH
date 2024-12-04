#include "Setup.h"

#include <cub/device/device_radix_sort.cuh>
#include <device_launch_parameters.h>

#include "BuilderUtils.h"

__global__ void NXB::ComputePrimBounds(BuildState buildState, float3* primitives)
{
	uint32_t primIdx = blockDim.x * blockIdx.x + threadIdx.x;

	if (primIdx >= buildState.primCount)
		return;

	float3 v0 = primitives[3 * primIdx];
	float3 v1 = primitives[3 * primIdx + 1];
	float3 v2 = primitives[3 * primIdx + 2];

	AABB primBounds(v0, v1, v2);

	buildState.primBounds[primIdx] = primBounds;
}

__global__ void NXB::ComputeSceneBounds(BuildState buildState, float3* primitives)
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

void NXB::RadixSort(uint64_t* mortonCodes, uint32_t* primIds, uint32_t size)
{
	size_t tempStorageSize = 0;
	uint64_t* mortonCodesSorted = nullptr;
	uint32_t* primIdsSorted = nullptr;
	cub::DeviceRadixSort::SortPairs(nullptr, tempStorageSize, mortonCodes, mortonCodesSorted, primIds, primIdsSorted, size, 0, 64);
}
