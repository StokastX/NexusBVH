#pragma once

#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

#include "CudaTestCheck.h"

namespace NXB::Test
{
	/*
	 * Owns one device allocation.
	 *
	 * Test cases bail out as soon as an invariant breaks, and pairing a cudaMalloc with
	 * a cudaFree by hand leaks the buffer every time they do. Over a suite that builds
	 * hundreds of BVHs that adds up to a device out-of-memory long before the run ends,
	 * which then looks like a builder bug rather than a test bug.
	 */
	template <typename T>
	class DeviceBuffer
	{
	public:
		DeviceBuffer() = default;

		explicit DeviceBuffer(size_t count)
		{
			if (count == 0)
				return;

			NXB_TEST_CUDA_CHECK(cudaMalloc((void**)&ptr, sizeof(T) * count));
			elemCount = count;
		}

		// Allocates and uploads in one step, which is what every case actually wants
		explicit DeviceBuffer(const std::vector<T>& host) : DeviceBuffer(host.size())
		{
			if (elemCount != 0)
				NXB_TEST_CUDA_CHECK(cudaMemcpy(ptr, host.data(), sizeof(T) * elemCount, cudaMemcpyHostToDevice));
		}

		~DeviceBuffer()
		{
			Release();
		}

		DeviceBuffer(const DeviceBuffer&) = delete;
		DeviceBuffer& operator=(const DeviceBuffer&) = delete;

		DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr), elemCount(other.elemCount)
		{
			other.ptr = nullptr;
			other.elemCount = 0;
		}

		DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
		{
			if (this != &other)
			{
				Release();
				ptr = other.ptr;
				elemCount = other.elemCount;
				other.ptr = nullptr;
				other.elemCount = 0;
			}
			return *this;
		}

		std::vector<T> ToHost() const
		{
			std::vector<T> host(elemCount);
			if (elemCount != 0)
				NXB_TEST_CUDA_CHECK(cudaMemcpy(host.data(), ptr, sizeof(T) * elemCount, cudaMemcpyDeviceToHost));
			return host;
		}

		T* Get() const { return ptr; }
		size_t Count() const { return elemCount; }

	private:
		void Release()
		{
			// A destructor must not throw, so a failing free is swallowed rather than
			// routed through NXB_TEST_CUDA_CHECK
			if (ptr != nullptr)
				cudaFree(ptr);

			ptr = nullptr;
			elemCount = 0;
		}

		T* ptr = nullptr;
		size_t elemCount = 0;
	};


	// Reads back a device range this class does not own, e.g. a buffer the builder
	// allocated and handed back inside a BVH handle
	template <typename T>
	std::vector<T> CopyToHost(const T* devicePtr, size_t count)
	{
		std::vector<T> host(count);
		if (count != 0)
			NXB_TEST_CUDA_CHECK(cudaMemcpy(host.data(), devicePtr, sizeof(T) * count, cudaMemcpyDeviceToHost));
		return host;
	}
}
