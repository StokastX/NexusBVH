#include "Setup.h"

#include <cub/device/device_radix_sort.cuh>
#include <device_launch_parameters.h>

#include "CudaUtils.h"
#include "BuilderUtils.h"

__global__ void NXB::ComputeSceneBounds(BuildState buildState, Triangle* primitives)
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

	BVH2::Node node;
	node.bounds = primitives[primIdx].Bounds();
	node.leftChild = INVALID_IDX;
	node.rightChild = primIdx;
	buildState.nodes[primIdx] = node;

	AtomicGrow(&sharedBounds, node.bounds);

	__syncthreads();

	// Scene bounds update
	if (threadIdx.x == 0)
		AtomicGrow(buildState.sceneBounds, sharedBounds);
}

__global__ void NXB::ComputeSceneBounds(BuildState buildState, AABB* primitives)
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

	BVH2::Node node;
	node.bounds = primitives[primIdx];
	node.leftChild = INVALID_IDX;
	node.rightChild = primIdx;
	buildState.nodes[primIdx] = node;

	AtomicGrow(&sharedBounds, node.bounds);

	__syncthreads();

	// Scene bounds update
	if (threadIdx.x == 0)
		AtomicGrow(buildState.sceneBounds, sharedBounds);
}

__global__ void NXB::ComputeMortonCodes(BuildState buildState)
{
	uint32_t primIdx = blockDim.x * blockIdx.x + threadIdx.x;

	if (primIdx >= buildState.primCount)
		return;

	AABB primBounds = buildState.nodes[primIdx].bounds;
	AABB *sceneBounds = buildState.sceneBounds;
	float3 centroid = primBounds.Centroid();

	uint64_t mortonCode = MortonCode((centroid - sceneBounds->bMin) / (sceneBounds->bMax - sceneBounds->bMin));
	buildState.mortonCodes[primIdx] = mortonCode;

	// Initialize cluster indices as well
	buildState.clusterIdx[primIdx] = primIdx;
}

void NXB::RadixSort(BuildState& buildState, BVHBuildMetrics* buildMetrics)
{
	using byte = unsigned char;

	size_t tempStorageBytes = 0;
	void* tempStorage = nullptr;

	uint64_t* mortonCodesSorted = CudaMemory::AllocAsync<uint64_t>(buildState.primCount);
	uint32_t* clusteridxSorted = CudaMemory::AllocAsync<uint32_t>(buildState.primCount);

	cub::DoubleBuffer<uint64_t> keysBuffer(buildState.mortonCodes, mortonCodesSorted);
	cub::DoubleBuffer<uint32_t> valuesBuffer(buildState.clusterIdx, clusteridxSorted);

	// Get the temporary storage size necessary to perform radix sorting
	cub::DeviceRadixSort::SortPairs(
		tempStorage,	// NULL
		tempStorageBytes,
		keysBuffer,
		valuesBuffer,
		buildState.primCount,
		1,
		64
	);

	tempStorage = CudaMemory::AllocAsync<byte>(tempStorageBytes);

	cudaEvent_t start, stop;
	if (buildMetrics)
	{
		CUDA_CHECK(cudaEventCreate(&start);
		CUDA_CHECK(cudaEventCreate(&stop);
		CUDA_CHECK(cudaDeviceSynchronize()));
		CUDA_CHECK(cudaEventRecord(start)));
	}

	// Perform radix sorting
	cub::DeviceRadixSort::SortPairs(
		tempStorage,	// NULL
		tempStorageBytes,
		keysBuffer,
		valuesBuffer,
		buildState.primCount,
		1,
		64
	);

	if (buildMetrics)
	{
		CUDA_CHECK(cudaEventRecord(stop));
		CUDA_CHECK(cudaEventSynchronize(stop));
		CUDA_CHECK(cudaEventElapsedTime(&buildMetrics->radixSortTime, start, stop));
		CUDA_CHECK(cudaEventDestroy(start));
		CUDA_CHECK(cudaEventDestroy(stop));
	}

	buildState.mortonCodes = keysBuffer.Current();
	buildState.clusterIdx = valuesBuffer.Current();

	CudaMemory::FreeAsync(tempStorage);

	CudaMemory::FreeAsync(keysBuffer.Alternate());
	CudaMemory::FreeAsync(valuesBuffer.Alternate());
}
