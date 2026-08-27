#include "NXB/BVHCost.h"

#include "NXB/DeviceBuffer.h"

#include "Math/Math.h"
#include "Cuda/Launch.h"
#include "Cuda/Eval.h"

namespace NXB
{
	namespace
	{
		// The block size the rest of the pipeline launches with
		constexpr uint32_t blockSize = 64;

		template <typename ViewT>
		float EvaluateCost(void (*costKernel)(ViewT, float*), ViewT bvh, uint32_t nodeCount,
			cudaStream_t stream, cudaMemPool_t pool)
		{
			DeviceBuffer<float> cost(1, stream, pool);
			cost.FillBytes(0);

			Launch(costKernel, DivideRoundUp(nodeCount, blockSize), blockSize, stream, bvh, cost.Get());

			// Download synchronizes, which is what makes the value safe to return
			float result = 0.0f;
			cost.Download(&result, 1);
			return result;
		}
	}

	float ComputeSAHCost(const BVH2& bvh, cudaStream_t stream, cudaMemPool_t pool)
	{
		if (bvh.Empty() || bvh.NodeCount() == 0)
			return 0.0f;

		return EvaluateCost(ComputeBVH2CostKernel, bvh.View(), bvh.NodeCount(), stream, pool);
	}

	float ComputeSAHCost(const BVH8& bvh, cudaStream_t stream, cudaMemPool_t pool)
	{
		if (bvh.Empty() || bvh.NodeCount() == 0)
			return 0.0f;

		return EvaluateCost(ComputeBVH8CostKernel, bvh.View(), bvh.NodeCount(), stream, pool);
	}
}
