#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "Error.h"

namespace NXB
{
	/* \brief A device memory pool the builds allocate from, reusable across builds
	 *
	 * CUDA's default pool is created with a release threshold of 0, so it returns every
	 * free byte to the driver at each synchronization -- and a build synchronizes before
	 * it returns. Repeated builds therefore re-acquire all of their memory from the driver
	 * every time, which measures as roughly half the wall time of a 1M primitive build.
	 *
	 * Handing one of these to BuildConfig::pool makes a build allocate from a pool that
	 * holds onto its memory instead, so the second and later builds find their buffers
	 * already reserved. Only reuse changes: a single cold build costs the same, and the
	 * BVH produced is identical either way.
	 *
	 * The pool is private to this object, so the process wide default pool -- and every
	 * allocation the caller makes through it -- is left alone. What this pool holds stays
	 * reserved until TrimTo releases it or the pool is destroyed, so a caller that needs
	 * the VRAM back has to ask.
	 *
	 * Move only. One pool per thread that builds: a pool may be used from several streams,
	 * but reuse across streams costs the driver a dependency it does not need within one.
	 */
	class MemoryPool
	{
	public:
		/* \param releaseThreshold Bytes the pool may keep reserved across a synchronization.
		 *        The default keeps everything, which is the point of using one.
		 * \param device The device to allocate on. Negative means the current device.
		 */
		explicit MemoryPool(uint64_t releaseThreshold = UINT64_MAX, int32_t device = -1)
		{
			if (device < 0)
				NXB_CUDA_CHECK(cudaGetDevice(&device));

			cudaMemPoolProps props = {};
			props.allocType = cudaMemAllocationTypePinned;
			props.handleTypes = cudaMemHandleTypeNone;
			props.location.type = cudaMemLocationTypeDevice;
			props.location.id = device;

			NXB_CUDA_CHECK(cudaMemPoolCreate(&m_pool, &props));

			// The pool is already created, so a failure here has to release it by hand:
			// the destructor does not run for an object whose constructor threw
			try
			{
				NXB_CUDA_CHECK(cudaMemPoolSetAttribute(m_pool, cudaMemPoolAttrReleaseThreshold, &releaseThreshold));
			}
			catch (...)
			{
				cudaMemPoolDestroy(m_pool);
				m_pool = nullptr;
				throw;
			}

			m_device = device;
		}

		~MemoryPool() { Reset(); }

		MemoryPool(const MemoryPool&) = delete;
		MemoryPool& operator=(const MemoryPool&) = delete;

		MemoryPool(MemoryPool&& other) noexcept
			: m_pool(other.m_pool), m_device(other.m_device)
		{
			other.m_pool = nullptr;
		}

		MemoryPool& operator=(MemoryPool&& other) noexcept
		{
			if (this != &other)
			{
				Reset();
				m_pool = other.m_pool;
				m_device = other.m_device;
				other.m_pool = nullptr;
			}
			return *this;
		}

		// What BuildConfig::pool wants
		cudaMemPool_t Handle() const { return m_pool; }

		// Releases everything the pool holds beyond bytesToKeep, down to what is still in
		// use. Call it when the VRAM is needed elsewhere; the next build re-acquires.
		void TrimTo(size_t bytesToKeep = 0)
		{
			if (m_pool)
				NXB_CUDA_CHECK(cudaMemPoolTrimTo(m_pool, bytesToKeep));
		}

		// Bytes the pool holds from the driver, in use or not
		uint64_t ReservedBytes() const { return Attribute(cudaMemPoolAttrReservedMemCurrent); }

		// Bytes currently handed out to live allocations
		uint64_t UsedBytes() const { return Attribute(cudaMemPoolAttrUsedMemCurrent); }

		/* rief The most bytes ever live at once since the last ResetPeakUsedBytes
		 *
		 * The peak working set of whatever ran in between, which for a build is every
		 * buffer it holds simultaneously. Worth more than ReservedBytes for checking that
		 * the allocations went where they were meant to: reserved is rounded up to the
		 * pool's chunk size and barely moves, this is exact to the byte.
		 */
		uint64_t PeakUsedBytes() const { return Attribute(cudaMemPoolAttrUsedMemHigh); }

		// Restarts the PeakUsedBytes measurement from what is live right now
		void ResetPeakUsedBytes()
		{
			if (!m_pool)
				return;

			uint64_t reset = 0;
			NXB_CUDA_CHECK(cudaMemPoolSetAttribute(m_pool, cudaMemPoolAttrUsedMemHigh, &reset));
		}

		int32_t Device() const { return m_device; }

		void Reset()
		{
			// A destructor must not throw, so a failing destroy is swallowed
			if (m_pool)
				cudaMemPoolDestroy(m_pool);

			m_pool = nullptr;
		}

	private:
		uint64_t Attribute(cudaMemPoolAttr attr) const
		{
			if (!m_pool)
				return 0;

			uint64_t value = 0;
			NXB_CUDA_CHECK(cudaMemPoolGetAttribute(m_pool, attr, &value));
			return value;
		}

		cudaMemPool_t m_pool = nullptr;
		int32_t m_device = -1;
	};
}