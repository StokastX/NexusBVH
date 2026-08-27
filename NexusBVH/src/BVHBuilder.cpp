#include "NXB/BVHBuilder.h"

#include "Math/Math.h"
#include "Cuda/CudaUtils.h"
#include "Cuda/Launch.h"
#include "Cuda/BinaryBuilder.h"
#include "Cuda/WideConverter.h"
#include "Cuda/Setup.h"
#include "Cuda/Eval.h"

namespace NXB
{
	/*
	 * \brief Evaluates the SAH cost of a finished BVH into *dst
	 *
	 * Only ever called when build metrics are requested, so the synchronization needed
	 * to hand the result back to the host is acceptable here.
	 */
	template <typename BvhT>
	void EvaluateCost(void (*costKernel)(BvhT, float*), const BvhT& bvh, uint32_t nodeCount, uint32_t blockSize, float* dst, cudaStream_t stream)
	{
		float* cost = CudaMemory::AllocAsync<float>(1, stream);
		CudaMemory::MemsetAsync(cost, 0, sizeof(float), stream);

		Launch(costKernel, DivideRoundUp(nodeCount, blockSize), blockSize, stream, bvh, cost);

		CudaMemory::CopyAsync(dst, cost, 1, cudaMemcpyDeviceToHost, stream);
		CudaMemory::FreeAsync(cost, stream);

		CUDA_CHECK(cudaStreamSynchronize(stream));
	}


	/*
	 * \brief Steps 2 to 4 of the pipeline: Morton codes, radix sort, H-PLOC merging
	 *
	 * buildState and bvh are taken by reference rather than by value: every copy below
	 * is asynchronous and nothing is synchronized until the end of BuildBVH2, so any
	 * host memory the driver reads from or writes into has to outlive that
	 * synchronization. A by-value parameter or a returned local would already be gone.
	 */
	template <typename McT>
	void BuildBVH2Impl(BVH2BuildState& buildState, BVH2& bvh, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics)
	{
		cudaStream_t stream = buildConfig.stream;
		const uint32_t blockSize = 64;
		const uint32_t nodeCount = buildState.primCount * 2 - 1;

		buildState.parentIdx = CudaMemory::AllocAsync<uint32_t>(buildState.primCount, stream);
		buildState.clusterIdx = CudaMemory::AllocAsync<uint32_t>(buildState.primCount, stream);
		buildState.clusterCount = CudaMemory::AllocAsync<uint32_t>(1, stream);
		McT* mortonCodes = CudaMemory::AllocAsync<McT>(buildState.primCount, stream);

		// Init parent ids to -1
		CudaMemory::MemsetAsync(buildState.parentIdx, INVALID_IDX, sizeof(uint32_t) * buildState.primCount, stream);
		CudaMemory::CopyAsync(buildState.clusterCount, &buildState.primCount, 1, cudaMemcpyHostToDevice, stream);


		// Step 2: Compute morton codes
		// ===============================================================================
		{
			StepTimer timer(MetricPtr(buildMetrics, &BVHBuildMetrics::computeMortonCodesTime), stream);

			uint32_t gridSize = CudaUtils::GetGridSizeFullOccupancy((void*)ComputeMortonCodesKernel<McT>, blockSize);
			Launch(ComputeMortonCodesKernel<McT>, gridSize, blockSize, stream, buildState, mortonCodes);
		}
		// ===============================================================================


		// Step 3: Sort morton codes
		// ===============================================================================
		RadixSort<McT>(buildState, mortonCodes, stream, buildMetrics);
		// ===============================================================================


		// Step 4: HPLOC binary BVH building
		// ===============================================================================
		{
			StepTimer timer(MetricPtr(buildMetrics, &BVHBuildMetrics::bvhBuildTime), stream);

			// RadixSort swapped both double buffers, so mortonCodes and
			// buildState.clusterIdx point somewhere else than they did above
			uint32_t gridSize = DivideRoundUp(buildState.primCount, blockSize);
			Launch(BuildBVH2Kernel<McT>, gridSize, blockSize, stream, buildState, mortonCodes);
		}
		// ===============================================================================


		bvh.primCount = buildState.primCount;
		bvh.nodeCount = nodeCount;
		bvh.nodes = buildState.nodes;
		CudaMemory::CopyAsync<AABB>(&bvh.bounds, buildState.sceneBounds, 1, cudaMemcpyDeviceToHost, stream);

		CudaMemory::FreeAsync(buildState.parentIdx, stream);
		CudaMemory::FreeAsync(buildState.clusterIdx, stream);
		CudaMemory::FreeAsync(buildState.clusterCount, stream);
		CudaMemory::FreeAsync(mortonCodes, stream);

		if (buildMetrics)
		{
			buildMetrics->totalTime = buildMetrics->computeSceneBoundsTime + buildMetrics->computeMortonCodesTime
				+ buildMetrics->radixSortTime + buildMetrics->bvhBuildTime;

			// The cost kernel takes bvh by value and divides by its bounds area, so the
			// readback above has to have landed before the launch
			CUDA_CHECK(cudaStreamSynchronize(stream));
			EvaluateCost(ComputeBVH2CostKernel, bvh, nodeCount, blockSize, &buildMetrics->bvh2Cost, stream);
		}
	}


	template<typename PrimT>
	BVH2 BuildBVH2(PrimT* primitives, uint32_t primCount, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics)
	{
		cudaStream_t stream = buildConfig.stream;

		BVH2 bvh;
		bvh.nodes = nullptr;
		bvh.nodeCount = 0;
		bvh.primCount = 0;
		bvh.bounds.Clear();

		// Without this guard, primCount * 2 - 1 underflows to 0xFFFFFFFF and the
		// allocation below fails inside CUDA_CHECK, which exits the process.
		if (primCount == 0)
			return bvh;

		uint32_t nodeCount = primCount * 2 - 1;
		BVH2BuildState buildState;
		buildState.primCount = primCount;
		buildState.sceneBounds = CudaMemory::AllocAsync<AABB>(1, stream);
		buildState.nodes = CudaMemory::AllocAsync<BVH2::Node>(nodeCount, stream);

		// Clear scene bounds
		AABB sceneBounds;
		sceneBounds.Clear();
		CudaMemory::CopyAsync(buildState.sceneBounds, &sceneBounds, 1, cudaMemcpyHostToDevice, stream);

		const uint32_t blockSize = 64;

		// Step 1: Compute scene bounding box
		// ===============================================================================
		{
			StepTimer timer(MetricPtr(buildMetrics, &BVHBuildMetrics::computeSceneBoundsTime), stream);

			uint32_t gridSize = CudaUtils::GetGridSizeFullOccupancy((void*)ComputeSceneBoundsKernel<PrimT>, blockSize);
			Launch(ComputeSceneBoundsKernel<PrimT>, gridSize, blockSize, stream, buildState, primitives);
		}
		// ===============================================================================


		// Step 2 - 4: Build BVH
		// ==============================================================================
		if (buildConfig.prioritizeSpeed)
			BuildBVH2Impl<uint32_t>(buildState, bvh, buildConfig, buildMetrics);
		else
			BuildBVH2Impl<uint64_t>(buildState, bvh, buildConfig, buildMetrics);
		// ==============================================================================

		CudaMemory::FreeAsync(buildState.sceneBounds, stream);

		// bvh.bounds, and the host locals the copies above read from, are only settled
		// once the stream has drained. Only this stream is synchronized: unrelated work
		// the caller has in flight elsewhere keeps running.
		CUDA_CHECK(cudaStreamSynchronize(stream));

		return bvh;
	}

	template <typename PrimT>
	BVH8 BuildBVH8(PrimT* primitives, uint32_t primCount, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics)
	{
		cudaStream_t stream = buildConfig.stream;

		BVH8 bvh8;
		bvh8.nodes = nullptr;
		bvh8.nodeCount = 0;
		bvh8.primIdx = nullptr;
		bvh8.primCount = 0;
		bvh8.bounds.Clear();

		if (primCount == 0)
			return bvh8;

		BVH2 bvh2 = BuildBVH2<PrimT>(primitives, primCount, buildConfig, buildMetrics);

		BVH8BuildState buildState;
		buildState.bvh2Nodes = bvh2.nodes;
		buildState.primCount = bvh2.primCount;

		// Worst case senario for a BVH8 built with H-PLOC collapsing: node count = (4n - 1) / 7.
		// This occurs when each internal node in the level above the leaves contains only two leaf nodes
		buildState.bvh8Nodes = CudaMemory::AllocAsync<BVH8::Node>(DivideRoundUp(4 * buildState.primCount - 1, 7), stream);
		buildState.primIdx = CudaMemory::AllocAsync<uint32_t>(buildState.primCount, stream);
		buildState.nodeCounter = CudaMemory::AllocAsync<uint32_t>(1, stream);
		buildState.leafCounter = CudaMemory::AllocAsync<uint32_t>(1, stream);
		buildState.workCounter = CudaMemory::AllocAsync<uint32_t>(1, stream);
		buildState.workAllocCounter = CudaMemory::AllocAsync<uint32_t>(1, stream);
		buildState.indexPairs = CudaMemory::AllocAsync<uint64_t>(buildState.primCount, stream);

		// Init index pairs
		CudaMemory::MemsetAsync(buildState.indexPairs, INVALID_IDX, sizeof(uint64_t) * buildState.primCount, stream);
		// Set first index pair to root of bvh2 and root of bvh8
		uint64_t firstPair = ((uint64_t)bvh2.nodeCount - 1) << 32;
		CudaMemory::CopyAsync(buildState.indexPairs, &firstPair, 1, cudaMemcpyHostToDevice, stream);
		uint32_t nodeCount = 1;
		CudaMemory::CopyAsync(buildState.nodeCounter, &nodeCount, 1, cudaMemcpyHostToDevice, stream);
		CudaMemory::CopyAsync(buildState.workAllocCounter, &nodeCount, 1, cudaMemcpyHostToDevice, stream);

		CudaMemory::MemsetAsync(buildState.workCounter, 0, sizeof(uint32_t), stream);
		CudaMemory::MemsetAsync(buildState.leafCounter, 0, sizeof(uint32_t), stream);

		const uint32_t blockSize = 256;

		// Step 5: BVH8 collapse
		// ===============================================================================
		{
			StepTimer timer(MetricPtr(buildMetrics, &BVHBuildMetrics::bvh8ConversionTime), stream);

			uint32_t gridSize = DivideRoundUp(buildState.primCount, blockSize);
			Launch(BuildBVH8Kernel, gridSize, blockSize, stream, buildState);
		}
		// ===============================================================================

		bvh8.nodes = buildState.bvh8Nodes;
		bvh8.primIdx = buildState.primIdx;
		bvh8.bounds = bvh2.bounds;
		bvh8.primCount = buildState.primCount;
		CudaMemory::CopyAsync<uint32_t>(&bvh8.nodeCount, buildState.nodeCounter, 1, cudaMemcpyDeviceToHost, stream);

		if (buildMetrics)
		{
			buildMetrics->totalTime += buildMetrics->bvh8ConversionTime;

			// Both the grid size and averageChildPerNode below need the node count the
			// collapse produced, so the readback above has to have landed
			CUDA_CHECK(cudaStreamSynchronize(stream));

			EvaluateCost(ComputeBVH8CostKernel, bvh8, bvh8.nodeCount, blockSize, &buildMetrics->bvh8Cost, stream);

			// Warning: this formula is only valid if a leaf node contains exactly one primitive
			// Should be (totalNodes - 1) / internalNodes
			buildMetrics->averageChildPerNode = (float)(bvh8.primCount + bvh8.nodeCount - 1) / bvh8.nodeCount;
		}

		// Stream-ordered, so the collapse kernel above is guaranteed to be done reading
		// the BVH2 nodes before they are released
		CudaMemory::FreeAsync(bvh2.nodes, stream);
		CudaMemory::FreeAsync(buildState.nodeCounter, stream);
		CudaMemory::FreeAsync(buildState.leafCounter, stream);
		CudaMemory::FreeAsync(buildState.workCounter, stream);
		CudaMemory::FreeAsync(buildState.workAllocCounter, stream);
		CudaMemory::FreeAsync(buildState.indexPairs, stream);

		CUDA_CHECK(cudaStreamSynchronize(stream));

		return bvh8;
	}

	BVH2 ToHost(BVH2 deviceBvh)
	{
		BVH2 hostBVH;
		hostBVH.primCount = deviceBvh.primCount;
		hostBVH.nodeCount = deviceBvh.nodeCount;
		hostBVH.bounds = deviceBvh.bounds;
		hostBVH.nodes = new BVH2::Node[deviceBvh.nodeCount];
		CudaMemory::Copy(hostBVH.nodes, deviceBvh.nodes, deviceBvh.nodeCount, cudaMemcpyDeviceToHost);
		return hostBVH;
	}

	void FreeHostBVH(BVH2 hostBvh)
	{
		delete[] hostBvh.nodes;
	}

	void FreeDeviceBVH(BVH2 deviceBvh)
	{
		CudaMemory::Free(deviceBvh.nodes);
	}

	void FreeDeviceBVH(BVH8 deviceBvh)
	{
		CudaMemory::Free(deviceBvh.nodes);
		CudaMemory::Free(deviceBvh.primIdx);
	}

	template BVH2 BuildBVH2<Triangle>(Triangle* primitives, uint32_t primCount, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics);
	template BVH2 BuildBVH2<AABB>(AABB* primitives, uint32_t primCount, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics);

	template BVH8 BuildBVH8<Triangle>(Triangle* primitives, uint32_t primCount, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics);
	template BVH8 BuildBVH8<AABB>(AABB* primitives, uint32_t primCount, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics);
}
