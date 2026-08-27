#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>

#define CUDA_CHECK(val)																		\
{																							\
	cudaError_t result = val;																\
	if (result) {																			\
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<		\
			__FILE__ << ":" << __LINE__ << " '" << #val << "' \n";							\
		cudaDeviceReset();																	\
		exit(99);																			\
	}																						\
}

namespace NXB
{
	// The async methods take the stream they are ordered on explicitly, with no
	// default: a build runs on the stream the caller picked in BuildConfig, and
	// silently falling back to the default stream would break that ordering.
	class CudaMemory
	{
	public:
		template<typename T>
		static T* Allocate(size_t count)
		{
			T* ptr;
			CUDA_CHECK(cudaMalloc((void**)&ptr, sizeof(T) * count));
			return ptr;
		}

		template<typename T>
		static T* AllocAsync(size_t count, cudaStream_t stream)
		{
			T* ptr;
			CUDA_CHECK(cudaMallocAsync((void**)&ptr, sizeof(T) * count, stream));
			return ptr;
		}

		template<typename T>
		static void Copy(T* dst, const T* src, size_t count, cudaMemcpyKind kind)
		{
			CUDA_CHECK(cudaMemcpy((void*)dst, (const void*)src, sizeof(T) * count, kind));
		}

		template<typename T>
		static void CopyAsync(T* dst, const T* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream)
		{
			CUDA_CHECK(cudaMemcpyAsync((void*)dst, (const void*)src, sizeof(T) * count, kind, stream));
		}

		// Warning: count is a BYTE count, unlike Copy/CopyAsync above
		static void Memset(void* dst, int32_t value, size_t count)
		{
			CUDA_CHECK(cudaMemset(dst, value, count));
		}

		// Warning: count is a BYTE count, unlike Copy/CopyAsync above
		static void MemsetAsync(void* dst, int32_t value, size_t count, cudaStream_t stream)
		{
			CUDA_CHECK(cudaMemsetAsync(dst, value, count, stream));
		}


		static void Free(void* ptr)
		{
			CUDA_CHECK(cudaFree(ptr));
		}

		static void FreeAsync(void* ptr, cudaStream_t stream)
		{
			CUDA_CHECK(cudaFreeAsync(ptr, stream));
		}
	};

	class CudaUtils
	{
	public:
		static uint32_t GetGridSizeFullOccupancy(const void* func, uint32_t blockSize)
		{
			// cudaGetDeviceProperties fills a large struct and costs far more than the
			// launch it is sizing. The SM count is the only field needed here and it
			// never changes for a given device, so query it once per device.
			static thread_local int32_t cachedDevice = -1;
			static thread_local int32_t smCount = 0;

			int32_t device = 0;
			CUDA_CHECK(cudaGetDevice(&device));
			if (device != cachedDevice)
			{
				CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
				cachedDevice = device;
			}

			int32_t blocksPerSM;
			CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, func, blockSize, 0));
			return (uint32_t)(blocksPerSM * smCount);
		}
	};
}
