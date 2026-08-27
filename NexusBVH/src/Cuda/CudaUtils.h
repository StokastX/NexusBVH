#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "NXB/Error.h"

namespace NXB
{
	namespace CudaUtils
	{
		inline uint32_t GetGridSizeFullOccupancy(const void* func, uint32_t blockSize)
		{
			// cudaGetDeviceProperties fills a large struct and costs far more than the
			// launch it is sizing. The SM count is the only field needed here and it
			// never changes for a given device, so query it once per device.
			static thread_local int32_t cachedDevice = -1;
			static thread_local int32_t smCount = 0;

			int32_t device = 0;
			NXB_CUDA_CHECK(cudaGetDevice(&device));
			if (device != cachedDevice)
			{
				NXB_CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
				cachedDevice = device;
			}

			int32_t blocksPerSM;
			NXB_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, func, blockSize, 0));
			return (uint32_t)(blocksPerSM * smCount);
		}
	}
}
