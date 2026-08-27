#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

#include "NXB/BVHBuildMetrics.h"
#include "NXB/Error.h"

namespace NXB
{
	/*
	 * \brief Type-checked wrapper around cudaLaunchKernel
	 *
	 * BVHBuilder.cpp is compiled by the host compiler, so kernels are launched through
	 * the runtime API rather than <<<>>>. That API takes a void* array and checks
	 * neither the number nor the types of the arguments. Deducing the parameter pack
	 * from the kernel pointer restores both at compile time.
	 *
	 * It also fixes the arguments at the point of the launch: the void* arrays this
	 * replaces were built once and reused across steps, so they silently picked up any
	 * mutation of the build state made in between (the buffer swap in RadixSort, for
	 * one). That happened to be the intended behaviour, but nothing said so.
	 */
	template <typename... Params, typename... Args>
	void Launch(void (*kernel)(Params...), uint32_t gridSize, uint32_t blockSize, cudaStream_t stream, Args&&... args)
	{
		static_assert(sizeof...(Params) == sizeof...(Args), "Wrong number of kernel arguments");
		static_assert((std::is_convertible_v<Args&&, Params> && ...), "Kernel argument types do not match the kernel signature");

		// Materialized as the kernel's own parameter types, so an implicit conversion
		// happens here rather than being reinterpreted by the driver
		std::tuple<Params...> params(std::forward<Args>(args)...);

		std::apply([&](Params&... param)
		{
			// One extra slot keeps the array from being zero-sized for a kernel that
			// takes no arguments. cudaLaunchKernel only reads as many entries as the
			// kernel signature declares.
			void* argPtrs[sizeof...(Params) + 1] = { (void*)&param..., nullptr };
			NXB_CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), gridSize, blockSize, argPtrs, 0, stream));
		}, params);
	}


	/*
	 * \brief Scoped CUDA event timer
	 *
	 * Writes the elapsed time of the enclosing scope into *dst, and does nothing at all
	 * when dst is nullptr, so a measured and an unmeasured build share a single code
	 * path instead of duplicating every launch across both branches of an if.
	 *
	 * Reading the result forces a synchronization, which is why passing build metrics
	 * makes a build measurably slower.
	 */
	class StepTimer
	{
	public:
		StepTimer(float* dst, cudaStream_t stream) : m_dst(dst), m_stream(stream)
		{
			if (!m_dst)
				return;

			NXB_CUDA_CHECK(cudaEventCreate(&m_start));

			// If anything below throws, the destructor never runs -- this object was never
			// fully constructed -- so the events created so far have to be released by hand
			try
			{
				NXB_CUDA_CHECK(cudaEventCreate(&m_stop));

				// Drain the stream before starting the clock. Without it the per-step
				// numbers silently borrow from each other whenever the host is free to run
				// ahead - which is exactly what BenchmarkBuild's back-to-back loop does.
				// Measured there, the radix sort reported 0.003 ms for 2M keys against a
				// true ~0.4 ms, with the difference credited to the neighbouring steps;
				// inserting any host-side work in the loop made the same build report the
				// sort correctly. The totals were right either way, the breakdown was not.
				//
				// The cost is that the steps are measured serialized rather than pipelined,
				// so the total reads higher than an untimed build actually takes. That is
				// the usual trade for per-kernel attribution, and it is only ever paid when
				// metrics are requested - StepTimer is inert otherwise.
				NXB_CUDA_CHECK(cudaStreamSynchronize(m_stream));
				NXB_CUDA_CHECK(cudaEventRecord(m_start, m_stream));
			}
			catch (...)
			{
				cudaEventDestroy(m_start);
				if (m_stop)
					cudaEventDestroy(m_stop);
				throw;
			}
		}

		~StepTimer() noexcept
		{
			if (!m_dst)
				return;

			// Deliberately unchecked. A destructor is noexcept, and this one also runs while
			// an exception from the timed scope unwinds, where a second one in flight would
			// terminate the process -- so NXB_CUDA_CHECK cannot be used here. Little is lost
			// by it: a real failure inside the scope was already reported by the launch that
			// raised it, and the only casualty here is one metrics number, which keeps
			// whatever value it came in with.
			if (cudaEventRecord(m_stop, m_stream) == cudaSuccess && cudaEventSynchronize(m_stop) == cudaSuccess)
				cudaEventElapsedTime(m_dst, m_start, m_stop);

			cudaEventDestroy(m_start);
			cudaEventDestroy(m_stop);
		}

		StepTimer(const StepTimer&) = delete;
		StepTimer& operator=(const StepTimer&) = delete;

	private:
		float* m_dst;
		cudaStream_t m_stream;
		cudaEvent_t m_start = nullptr;
		cudaEvent_t m_stop = nullptr;
	};


	/*
	 * \brief Address of one metrics field, or nullptr when metrics are not requested
	 *
	 * Turns the "is this build measured?" test into a single nullptr that StepTimer
	 * already knows how to ignore.
	 */
	inline float* MetricPtr(BVHBuildMetrics* buildMetrics, float BVHBuildMetrics::* field)
	{
		return buildMetrics ? &(buildMetrics->*field) : nullptr;
	}
}
