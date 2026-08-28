#include "NXB/BVHBuilder.h"

#include "NXB/DeviceBuffer.h"

#include "Math/Math.h"
#include "Cuda/CudaUtils.h"
#include "Cuda/Launch.h"
#include "Cuda/BinaryBuilder.h"
#include "Cuda/WideConverter.h"
#include "Cuda/Setup.h"

namespace NXB
{
	/*
	 * \brief Steps 2 to 4 of the pipeline: Morton codes, radix sort, H-PLOC merging
	 *
	 * buildState and view are taken by reference rather than by value: every copy below
	 * is asynchronous and nothing is synchronized until the end of BuildBVH2, so any
	 * host memory the driver reads from or writes into has to outlive that
	 * synchronization. A by-value parameter or a returned local would already be gone.
	 *
	 * It fills a view rather than a BVH2 because the owner cannot exist yet: the node
	 * buffer is still a local of BuildBVH2 here, and moving it in before the stream has
	 * drained would move it out from under the copies still in flight.
	 */
	template <typename McT>
	void BuildBVH2Impl(BVH2BuildState& buildState, BVH2::DeviceView& view, BuildConfig buildConfig, StepTimers& timers)
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
			StepTimer timer(timers, &BVHBuildMetrics::computeMortonCodesTime);

			uint32_t gridSize = CudaUtils::GetGridSizeFullOccupancy((void*)ComputeMortonCodesKernel<McT>, blockSize);
			Launch(ComputeMortonCodesKernel<McT>, gridSize, blockSize, stream, buildState, mortonCodes.Get());
		}
		// ===============================================================================


		// Step 3: Sort morton codes
		// ===============================================================================
		RadixSort<McT>(buildState, mortonCodes, clusterIdx, buildConfig, timers);
		// ===============================================================================


		// Step 4: HPLOC binary BVH building
		// ===============================================================================
		{
			StepTimer timer(timers, &BVHBuildMetrics::bvhBuildTime);

			// RadixSort may have swapped both double buffers, so mortonCodes and
			// buildState.clusterIdx can point somewhere else than they did above
			uint32_t gridSize = DivideRoundUp(buildState.primCount, blockSize);
			Launch(BuildBVH2Kernel<McT>, gridSize, blockSize, stream, buildState, mortonCodes.Get());
		}
		// ===============================================================================


		view.primCount = buildState.primCount;
		view.nodeCount = nodeCount;
		view.nodes = buildState.nodes;
		CopyToHostAsync(&view.bounds, buildState.sceneBounds, 1, stream);
	}


	template<typename PrimT>
	BVH2 BuildBVH2(PrimT* primitives, uint32_t primCount, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics)
	{
		cudaStream_t stream = buildConfig.stream;

		// Without this guard, primCount * 2 - 1 underflows to 0xFFFFFFFF and the
		// allocation below throws for what is a legitimate, if empty, request.
		if (primCount == 0)
			return BVH2();

		// Holds every step's event pair until the synchronize below, which is the only
		// point at which reading one back is free
		StepTimers timers(buildMetrics, stream);

		BVH2::DeviceView view = {};
		view.nodes = nullptr;
		view.nodeCount = 0;
		view.primCount = 0;
		view.bounds.Clear();

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
			StepTimer timer(timers, &BVHBuildMetrics::computeSceneBoundsTime);

			uint32_t gridSize = CudaUtils::GetGridSizeFullOccupancy((void*)ComputeSceneBoundsKernel<PrimT>, blockSize);
			Launch(ComputeSceneBoundsKernel<PrimT>, gridSize, blockSize, stream, buildState, primitives);
		}
		// ===============================================================================


		// Step 2 - 4: Build BVH
		// ==============================================================================
		if (buildConfig.prioritizeSpeed)
			BuildBVH2Impl<uint32_t>(buildState, view, buildConfig, timers);
		else
			BuildBVH2Impl<uint64_t>(buildState, view, buildConfig, timers);
		// ==============================================================================

		// view.bounds, and the host locals the copies above read from, are only settled
		// once the stream has drained. Only this stream is synchronized: unrelated work
		// the caller has in flight elsewhere keeps running.
		NXB_CUDA_CHECK(cudaStreamSynchronize(stream));

		timers.Flush();
		if (buildMetrics)
			buildMetrics->totalTime = buildMetrics->computeSceneBoundsTime + buildMetrics->computeMortonCodesTime
				+ buildMetrics->radixSortTime + buildMetrics->bvhBuildTime;

		// After the synchronize, so that a throw there releases the buffer rather than
		// handing it to an owner that is never returned
		return BVH2(std::move(nodes), view.primCount, view.bounds);
	}

	template <typename PrimT>
	BVH8 BuildBVH8(PrimT* primitives, uint32_t primCount, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics)
	{
		cudaStream_t stream = buildConfig.stream;

		if (primCount == 0)
			return BVH8();

		// Only the collapse is timed here; BuildBVH2 below flushes its own four steps
		StepTimers timers(buildMetrics, stream);

		/*
		 * The BVH2 is scratch from here on: the collapse reads it, and it is released when
		 * this owner goes out of scope at the end of the function. That free is stream
		 * ordered on the build stream, so the collapse kernel is guaranteed to be done
		 * reading the nodes by the time it happens.
		 */
		BVH2 bvh2 = BuildBVH2<PrimT>(primitives, primCount, buildConfig, buildMetrics);
		const BVH2::DeviceView bvh2View = bvh2.View();

		BVH8BuildState buildState;
		buildState.bvh2Nodes = bvh2View.nodes;
		buildState.primCount = bvh2View.primCount;

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
		uint64_t firstPair = ((uint64_t)bvh2View.nodeCount - 1) << 32;
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
			StepTimer timer(timers, &BVHBuildMetrics::bvh8ConversionTime);

			uint32_t gridSize = DivideRoundUp(buildState.primCount, blockSize);
			Launch(BuildBVH8Kernel, gridSize, blockSize, stream, buildState);
		}
		// ===============================================================================

		uint32_t bvh8NodeCount = 0;
		nodeCounter.DownloadAsync(&bvh8NodeCount, 1);

		// bvh8NodeCount is only settled once the readback above has landed
		NXB_CUDA_CHECK(cudaStreamSynchronize(stream));

		timers.Flush();
		if (buildMetrics)
			buildMetrics->totalTime += buildMetrics->bvh8ConversionTime;

		return BVH8(std::move(bvh8Nodes), std::move(primIdx), bvh8NodeCount,
			buildState.primCount, bvh2View.bounds);
	}

	template BVH2 BuildBVH2<Triangle>(Triangle* primitives, uint32_t primCount, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics);
	template BVH2 BuildBVH2<AABB>(AABB* primitives, uint32_t primCount, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics);

	template BVH8 BuildBVH8<Triangle>(Triangle* primitives, uint32_t primCount, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics);
	template BVH8 BuildBVH8<AABB>(AABB* primitives, uint32_t primCount, BuildConfig buildConfig, BVHBuildMetrics* buildMetrics);
}
