#include "NXB/BVHBuilder.h"

#include <memory>

#include "NXB/DeviceBuffer.h"

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
	template <typename ViewT>
	void EvaluateCost(void (*costKernel)(ViewT, float*), ViewT bvh, uint32_t nodeCount, uint32_t blockSize, float* dst, const BuildConfig& buildConfig)
	{
		cudaStream_t stream = buildConfig.stream;
		DeviceBuffer<float> cost(1, stream, buildConfig.pool);
		cost.FillBytes(0);

		Launch(costKernel, DivideRoundUp(nodeCount, blockSize), blockSize, stream, bvh, cost.Get());

		cost.DownloadAsync(dst, 1);

		NXB_CUDA_CHECK(cudaStreamSynchronize(stream));
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

		// Scratch for the three steps below. The build state keeps raw pointers because it
		// is passed by value into kernels, which an owning type cannot be.
		DeviceBuffer<uint32_t> parentIdx(buildState.primCount, stream, buildConfig.pool);
		DeviceBuffer<uint32_t> clusterIdx(buildState.primCount, stream, buildConfig.pool);
		DeviceBuffer<uint32_t> clusterCount(1, stream, buildConfig.pool);
		DeviceBuffer<McT> mortonCodes(buildState.primCount, stream, buildConfig.pool);

		buildState.parentIdx = parentIdx.Get();
		buildState.clusterIdx = clusterIdx.Get();
		buildState.clusterCount = clusterCount.Get();

		// Init parent ids to -1, i.e. every byte set to 0xff
		parentIdx.FillBytes(0xff);
		clusterCount.UploadAsync(&buildState.primCount, 1);


		// Step 2: Compute morton codes
		// ===============================================================================
		{
			StepTimer timer(MetricPtr(buildMetrics, &BVHBuildMetrics::computeMortonCodesTime), stream);

			uint32_t gridSize = CudaUtils::GetGridSizeFullOccupancy((void*)ComputeMortonCodesKernel<McT>, blockSize);
			Launch(ComputeMortonCodesKernel<McT>, gridSize, blockSize, stream, buildState, mortonCodes.Get());
		}
		// ===============================================================================


		// Step 3: Sort morton codes
		// ===============================================================================
		RadixSort<McT>(buildState, mortonCodes, clusterIdx, buildConfig, buildMetrics);
		// ===============================================================================


		// Step 4: HPLOC binary BVH building
		// ===============================================================================
		{
			StepTimer timer(MetricPtr(buildMetrics, &BVHBuildMetrics::bvhBuildTime), stream);

			// RadixSort may have swapped both double buffers, so mortonCodes and
			// buildState.clusterIdx can point somewhere else than they did above
			uint32_t gridSize = DivideRoundUp(buildState.primCount, blockSize);
			Launch(BuildBVH2Kernel<McT>, gridSize, blockSize, stream, buildState, mortonCodes.Get());
		}
		// ===============================================================================


		bvh.primCount = buildState.primCount;
		bvh.nodeCount = nodeCount;
		bvh.nodes = buildState.nodes;
		CopyToHostAsync(&bvh.bounds, buildState.sceneBounds, 1, stream);

		if (buildMetrics)
		{
			buildMetrics->totalTime = buildMetrics->computeSceneBoundsTime + buildMetrics->computeMortonCodesTime
				+ buildMetrics->radixSortTime + buildMetrics->bvhBuildTime;

			// The cost kernel takes bvh by value and divides by its bounds area, so the
			// readback above has to have landed before the launch
			NXB_CUDA_CHECK(cudaStreamSynchronize(stream));
			EvaluateCost(ComputeBVH2CostKernel, bvh.View(), nodeCount, blockSize, &buildMetrics->bvh2Cost, buildConfig);
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
		// allocation below throws for what is a legitimate, if empty, request.
		if (primCount == 0)
			return bvh;

		uint32_t nodeCount = primCount * 2 - 1;
		BVH2BuildState buildState;
		buildState.primCount = primCount;
		DeviceBuffer<AABB> sceneBoundsBuffer(1, stream, buildConfig.pool);
		DeviceBuffer<BVH2::Node> nodes(nodeCount, stream, buildConfig.pool);

		buildState.sceneBounds = sceneBoundsBuffer.Get();
		buildState.nodes = nodes.Get();

		// Clear scene bounds
		AABB sceneBounds;
		sceneBounds.Clear();
		sceneBoundsBuffer.UploadAsync(&sceneBounds, 1);

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

		// bvh.bounds, and the host locals the copies above read from, are only settled
		// once the stream has drained. Only this stream is synchronized: unrelated work
		// the caller has in flight elsewhere keeps running.
		NXB_CUDA_CHECK(cudaStreamSynchronize(stream));

		// bvh.nodes already points at it, and the caller releases it with FreeDeviceBVH.
		// After the synchronize, so that a throw there still frees it.
		nodes.Release();

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

		// The BVH2 is scratch from here on: the collapse reads it, then it is released
		DeviceBuffer<BVH2::Node> bvh2Nodes = DeviceBuffer<BVH2::Node>::Adopt(bvh2.nodes, bvh2.nodeCount, stream);

		BVH8BuildState buildState;
		buildState.bvh2Nodes = bvh2Nodes.Get();
		buildState.primCount = bvh2.primCount;

		// Worst case senario for a BVH8 built with H-PLOC collapsing: node count = (4n - 1) / 7.
		// This occurs when each internal node in the level above the leaves contains only two leaf nodes
		DeviceBuffer<BVH8::Node> bvh8Nodes(DivideRoundUp(4 * buildState.primCount - 1, 7), stream, buildConfig.pool);
		DeviceBuffer<uint32_t> primIdx(buildState.primCount, stream, buildConfig.pool);
		DeviceBuffer<uint32_t> nodeCounter(1, stream, buildConfig.pool);
		DeviceBuffer<uint32_t> leafCounter(1, stream, buildConfig.pool);
		DeviceBuffer<uint32_t> workCounter(1, stream, buildConfig.pool);
		DeviceBuffer<uint32_t> workAllocCounter(1, stream, buildConfig.pool);
		DeviceBuffer<uint64_t> indexPairs(buildState.primCount, stream, buildConfig.pool);

		buildState.bvh8Nodes = bvh8Nodes.Get();
		buildState.primIdx = primIdx.Get();
		buildState.nodeCounter = nodeCounter.Get();
		buildState.leafCounter = leafCounter.Get();
		buildState.workCounter = workCounter.Get();
		buildState.workAllocCounter = workAllocCounter.Get();
		buildState.indexPairs = indexPairs.Get();

		// Init index pairs to -1, i.e. every byte set to 0xff
		indexPairs.FillBytes(0xff);
		// Set first index pair to root of bvh2 and root of bvh8
		uint64_t firstPair = ((uint64_t)bvh2.nodeCount - 1) << 32;
		indexPairs.UploadAsync(&firstPair, 1);
		uint32_t nodeCount = 1;
		nodeCounter.UploadAsync(&nodeCount, 1);
		workAllocCounter.UploadAsync(&nodeCount, 1);

		workCounter.FillBytes(0);
		leafCounter.FillBytes(0);

		const uint32_t blockSize = 256;

		// Step 5: BVH8 collapse
		// ===============================================================================
		{
			StepTimer timer(MetricPtr(buildMetrics, &BVHBuildMetrics::bvh8ConversionTime), stream);

			uint32_t gridSize = DivideRoundUp(buildState.primCount, blockSize);
			Launch(BuildBVH8Kernel, gridSize, blockSize, stream, buildState);
		}
		// ===============================================================================

		bvh8.nodes = bvh8Nodes.Get();
		bvh8.primIdx = primIdx.Get();
		bvh8.bounds = bvh2.bounds;
		bvh8.primCount = buildState.primCount;
		nodeCounter.DownloadAsync(&bvh8.nodeCount, 1);

		if (buildMetrics)
		{
			buildMetrics->totalTime += buildMetrics->bvh8ConversionTime;

			// Both the grid size and averageChildPerNode below need the node count the
			// collapse produced, so the readback above has to have landed
			NXB_CUDA_CHECK(cudaStreamSynchronize(stream));

			EvaluateCost(ComputeBVH8CostKernel, bvh8.View(), bvh8.nodeCount, blockSize, &buildMetrics->bvh8Cost, buildConfig);

			// Warning: this formula is only valid if a leaf node contains exactly one primitive
			// Should be (totalNodes - 1) / internalNodes
			buildMetrics->averageChildPerNode = (float)(bvh8.primCount + bvh8.nodeCount - 1) / bvh8.nodeCount;
		}

		// The frees at the end of this scope are stream ordered, so the collapse kernel is
		// guaranteed to be done reading the BVH2 nodes by the time they go.
		NXB_CUDA_CHECK(cudaStreamSynchronize(stream));

		bvh8Nodes.Release();
		primIdx.Release();

		return bvh8;
	}

	BVH2 ToHost(BVH2 deviceBvh)
	{
		BVH2 hostBvh = deviceBvh;

		std::unique_ptr<BVH2::Node[]> nodes(new BVH2::Node[deviceBvh.nodeCount]);
		CopyToHost(nodes.get(), deviceBvh.nodes, deviceBvh.nodeCount);

		hostBvh.nodes = nodes.release();
		return hostBvh;
	}

	BVH8 ToHost(BVH8 deviceBvh)
	{
		BVH8 hostBvh = deviceBvh;

		std::unique_ptr<BVH8::Node[]> nodes(new BVH8::Node[deviceBvh.nodeCount]);
		std::unique_ptr<uint32_t[]> primIdx(new uint32_t[deviceBvh.primCount]);
		CopyToHost(nodes.get(), deviceBvh.nodes, deviceBvh.nodeCount);
		CopyToHost(primIdx.get(), deviceBvh.primIdx, deviceBvh.primCount);

		hostBvh.nodes = nodes.release();
		hostBvh.primIdx = primIdx.release();
		return hostBvh;
	}

	void FreeHostBVH(BVH2 hostBvh)
	{
		delete[] hostBvh.nodes;
	}

	void FreeHostBVH(BVH8 hostBvh)
	{
		delete[] hostBvh.nodes;
		delete[] hostBvh.primIdx;
	}

	/*
	 * The arrays always come from the async allocator, so they are released back to it
	 * rather than with cudaFree. cudaFree does free them, but when the build allocated
	 * from a BuildConfig::pool it returns them to the driver behind that pool's back,
	 * leaving cudaMemPoolAttrUsedMemCurrent permanently overstated.
	 *
	 * These entry points take no stream. The null stream is safe regardless of the one the
	 * build ran on, because a build synchronizes its stream before handing the BVH over,
	 * so nothing is still reading these arrays.
	 */
	void FreeDeviceBVH(BVH2 deviceBvh)
	{
		NXB_CUDA_CHECK(cudaFreeAsync(deviceBvh.nodes, 0));
		NXB_CUDA_CHECK(cudaStreamSynchronize(0));
	}

	void FreeDeviceBVH(BVH8 deviceBvh)
	{
		NXB_CUDA_CHECK(cudaFreeAsync(deviceBvh.nodes, 0));
		NXB_CUDA_CHECK(cudaFreeAsync(deviceBvh.primIdx, 0));
		NXB_CUDA_CHECK(cudaStreamSynchronize(0));
	}

	template BVH2 BuildBVH2<Triangle>(Triangle* primitives, uint32_t primCount, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics);
	template BVH2 BuildBVH2<AABB>(AABB* primitives, uint32_t primCount, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics);

	template BVH8 BuildBVH8<Triangle>(Triangle* primitives, uint32_t primCount, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics);
	template BVH8 BuildBVH8<AABB>(AABB* primitives, uint32_t primCount, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics);
}
