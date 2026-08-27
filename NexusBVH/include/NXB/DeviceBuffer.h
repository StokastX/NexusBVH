#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "Error.h"

namespace NXB
{
	/* \brief Owns one stream ordered device allocation
	 *
	 * Allocation is stream ordered (cudaMallocAsync) and never synchronizes. The entry
	 * points that touch host memory are split along that line:
	 *
	 *   - the std::vector constructor, Upload and Download synchronize the buffer's stream
	 *     before returning, so the pointer they produce is safe to hand to a build running
	 *     on any stream, and the host memory they touch need not outlive the call
	 *   - UploadAsync and DownloadAsync do not, and the host memory they touch has to stay
	 *     alive until the stream drains
	 *
	 * Move only.
	 */
	template <typename T>
	class DeviceBuffer
	{
	public:
		DeviceBuffer() = default;

		/* \param pool Where to allocate from. nullptr means the stream's default pool,
		 *        i.e. plain cudaMallocAsync. See NXB/MemoryPool.h for why a caller that
		 *        builds more than once wants to pass one.
		 */
		explicit DeviceBuffer(size_t count, cudaStream_t stream = 0, cudaMemPool_t pool = nullptr) : m_stream(stream)
		{
			if (count == 0)
				return;

			if (pool)
				NXB_CUDA_CHECK(cudaMallocFromPoolAsync((void**)&m_ptr, sizeof(T) * count, pool, stream));
			else
				NXB_CUDA_CHECK(cudaMallocAsync((void**)&m_ptr, sizeof(T) * count, stream));

			m_count = count;
		}

		explicit DeviceBuffer(const std::vector<T>& host, cudaStream_t stream = 0, cudaMemPool_t pool = nullptr)
			: DeviceBuffer(host.size(), stream, pool)
		{
			Upload(host.data(), host.size());
		}

		// Takes ownership of a pointer this class did not allocate
		static DeviceBuffer Adopt(T* devicePtr, size_t count, cudaStream_t stream = 0)
		{
			DeviceBuffer buffer;
			buffer.m_ptr = devicePtr;
			buffer.m_count = count;
			buffer.m_stream = stream;
			return buffer;
		}

		~DeviceBuffer() { Reset(); }

		DeviceBuffer(const DeviceBuffer&) = delete;
		DeviceBuffer& operator=(const DeviceBuffer&) = delete;

		DeviceBuffer(DeviceBuffer&& other) noexcept
			: m_ptr(other.m_ptr), m_count(other.m_count), m_stream(other.m_stream)
		{
			other.m_ptr = nullptr;
			other.m_count = 0;
		}

		DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
		{
			if (this != &other)
			{
				Reset();
				m_ptr = other.m_ptr;
				m_count = other.m_count;
				m_stream = other.m_stream;
				other.m_ptr = nullptr;
				other.m_count = 0;
			}
			return *this;
		}

		// Uploads and synchronizes, so host may be freed as soon as this returns
		void Upload(const T* host, size_t count)
		{
			UploadAsync(host, count);
			NXB_CUDA_CHECK(cudaStreamSynchronize(m_stream));
		}

		void Upload(const std::vector<T>& host) { Upload(host.data(), host.size()); }

		// Warning: stream ordered. host has to stay alive until the stream drains.
		void UploadAsync(const T* host, size_t count)
		{
			if (count == 0)
				return;

			NXB_CUDA_CHECK(cudaMemcpyAsync(m_ptr, host, sizeof(T) * count, cudaMemcpyHostToDevice, m_stream));
		}

		// Warning: stream ordered. host has to stay alive until the stream drains.
		void DownloadAsync(T* host, size_t count) const
		{
			if (count == 0)
				return;

			NXB_CUDA_CHECK(cudaMemcpyAsync(host, m_ptr, sizeof(T) * count, cudaMemcpyDeviceToHost, m_stream));
		}

		// Reads back and synchronizes
		void Download(T* host, size_t count) const
		{
			if (count == 0)
				return;

			DownloadAsync(host, count);
			NXB_CUDA_CHECK(cudaStreamSynchronize(m_stream));
		}

		std::vector<T> ToHost() const
		{
			std::vector<T> host(m_count);
			Download(host.data(), m_count);
			return host;
		}

		// Byte semantics, like cudaMemset: FillBytes(1) does not give a buffer full of ones
		void FillBytes(uint8_t value)
		{
			if (m_count == 0)
				return;

			NXB_CUDA_CHECK(cudaMemsetAsync(m_ptr, value, sizeof(T) * m_count, m_stream));
		}

		// Gives up ownership, for memory that outlives this object
		T* Release()
		{
			T* released = m_ptr;
			m_ptr = nullptr;
			m_count = 0;
			return released;
		}

		void Reset()
		{
			/*
			 * cudaFreeAsync returns the block to whichever pool it was taken from, so this
			 * needs no knowledge of the pool the constructor used.
			 *
			 * A destructor must not throw, so a failing free is discarded rather than
			 * checked. It can genuinely fail: freeing on a stream the caller has already
			 * destroyed gives cudaErrorInvalidResourceHandle, which is why this has to go
			 * through CudaDiscard rather than merely ignore the return value.
			 */
			if (m_ptr != nullptr)
				CudaDiscard(cudaFreeAsync(m_ptr, m_stream));

			m_ptr = nullptr;
			m_count = 0;
		}

		T* Get() const { return m_ptr; }
		size_t Count() const { return m_count; }
		cudaStream_t Stream() const { return m_stream; }

	private:
		T* m_ptr = nullptr;
		size_t m_count = 0;
		cudaStream_t m_stream = 0;
	};


	/*
	 * Readbacks from a device range this process does not hold a DeviceBuffer for, such as
	 * the arrays inside a BVH handle. ELEMENT counts, with T deduced on both sides.
	 */
	template <typename T>
	void CopyToHostAsync(T* host, const T* devicePtr, size_t count, cudaStream_t stream)
	{
		if (count == 0)
			return;

		NXB_CUDA_CHECK(cudaMemcpyAsync(host, devicePtr, sizeof(T) * count, cudaMemcpyDeviceToHost, stream));
	}

	template <typename T>
	void CopyToHost(T* host, const T* devicePtr, size_t count)
	{
		if (count == 0)
			return;

		NXB_CUDA_CHECK(cudaMemcpy(host, devicePtr, sizeof(T) * count, cudaMemcpyDeviceToHost));
	}

	template <typename T>
	std::vector<T> CopyToHost(const T* devicePtr, size_t count)
	{
		std::vector<T> host(count);
		CopyToHost(host.data(), devicePtr, count);
		return host;
	}
}
