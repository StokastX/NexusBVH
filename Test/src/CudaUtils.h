#pragma once
#include <cuda_runtime.h>

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

class CudaMemory
{
public:
	template<typename T>
	static T* Allocate(uint32_t count)
	{
		T* ptr;
		CUDA_CHECK(cudaMalloc((void**)&ptr, sizeof(T) * count));
		return ptr;
	}

	template<typename T>
	static T* AllocAsync(uint32_t count)
	{
		T* ptr;
		CUDA_CHECK(cudaMallocAsync((void**)&ptr, sizeof(T) * count, 0));
		return ptr;
	}

	template<typename T>
	static void Copy(T* dst, T* src, uint32_t count, cudaMemcpyKind kind)
	{
		CUDA_CHECK(cudaMemcpy((void*)dst, (void*)src, sizeof(T) * count, kind));
	}

	template<typename T>
	static void CopyAsync(T* dst, T* src, uint32_t count, cudaMemcpyKind kind)
	{
		CUDA_CHECK(cudaMemcpyAsync((void*)dst, (void*)src, sizeof(T) * count, kind));
	}

	static void Memset(void* dst, int32_t value, uint32_t count)
	{
		CUDA_CHECK(cudaMemset(dst, value, count));
	}

	static void MemsetAsync(void* dst, int32_t value, uint32_t count)
	{
		CUDA_CHECK(cudaMemsetAsync(dst, value, count));
	}


	static void Free(void* ptr)
	{
		CUDA_CHECK(cudaFree(ptr));
	}

	static void FreeAsync(void* ptr)
	{
		CUDA_CHECK(cudaFreeAsync(ptr, 0));
	}
};
