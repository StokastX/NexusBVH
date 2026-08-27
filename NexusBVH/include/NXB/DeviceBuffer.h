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
	 * The build functions take device pointers and, before this, gave the caller nothing
	 * to make one with -- so every user wrote the same cudaMalloc / cudaMemcpy / cudaFree
	 * triple and leaked it on the first early return. This is that triple, written once.
	 *
	 * Allocation is stream ordered (cudaMallocAsync) and never synchronizes, which is what
	 * the builder needs for its scratch buffers. The entry points that touch host memory
	 * are split along that line:
	 *
	 *   - the std::vector constructor, Upload and Download synchronize the buffer's stream
	 *     before returning, so the pointer they produce is safe to hand to a build running
	 *     on any stream, and the host memory they read from need not outlive the call
	 *   - UploadAsync does not, and the host memory it reads from has to stay alive until
	 *     the stream drains
	 *
	 * Move only. The destructor cannot report a failing free, so it swallows it.
	 */
	template <typename T>
	class DeviceBuffer
	{
	public:
		DeviceBuffer() = default;

		explicit DeviceBuffer(size_t count, cudaStream_t stream = 0) : m_stream(stream)
		{
			if (count == 0)
				return;

			NXB_CUDA_CHECK(cudaMallocAsync((void**)&m_ptr, sizeof(T) * count, stream));
			m_count = count;
		}

		// Allocates and uploads in one step, which is what a caller with geometry in a
		// std::vector actually wants
		explicit DeviceBuffer(const std::vector<T>& host, cudaStream_t stream = 0)
			: DeviceBuffer(host.size(), stream)
		{
			Upload(host.data(), host.size());
		}

		/* \brief Takes ownership of an existing device pointer
		 *
		 * For memory this class did not allocate but should still release on the way out
		 * -- the node array inside a BVH handle, say.
		 */
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

		/* \brief Sets every byte of the buffer to value
		 *
		 * Byte semantics, like cudaMemset: FillBytes(1) does not give a buffer full of
		 * ones. There is deliberately no count parameter: the raw cudaMemset takes a byte
		 * count while its neighbours take element counts, and that asymmetry has already
		 * caused a bug here. Filling the whole buffer leaves nothing to get wrong.
		 */
		void FillBytes(uint8_t value)
		{
			if (m_count == 0)
				return;

			NXB_CUDA_CHECK(cudaMemsetAsync(m_ptr, value, sizeof(T) * m_count, m_stream));
		}

		// Gives up ownership, for memory that outlives this object -- the node array
		// handed back inside a BVH handle, which the caller frees with FreeDeviceBVH
		T* Release()
		{
			T* released = m_ptr;
			m_ptr = nullptr;
			m_count = 0;
			return released;
		}

		void Reset()
		{
			// A destructor must not throw, so a failing free is swallowed here rather
			// than routed through NXB_CUDA_CHECK
			if (m_ptr != nullptr)
				cudaFreeAsync(m_ptr, m_stream);

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
	 * Readbacks from a device range this process does not hold a DeviceBuffer for -- the
	 * arrays inside a BVH handle, which are raw pointers.
	 *
	 * These take an ELEMENT count and deduce T on both sides, so neither a stray sizeof
	 * nor a host/device mix-up compiles.
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
