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
	 * \brief The event pairs of one build, read only once the build has synchronized
	 *
	 * cudaEventRecord is asynchronous: it enqueues a timestamp and returns. Only reading
	 * one back waits for the GPU. Recording every step's pair in stream order and reading
	 * them all after the single synchronize a build already does therefore measures each
	 * step exactly -- an event is timestamped when the GPU reaches it, so a pair brackets
	 * its own kernel no matter how far ahead the host has run -- without the per-step
	 * synchronization that used to serialize the pipeline.
	 *
	 * Inert when metrics were not requested: no event is created and every operation is a
	 * null check, which is what lets a measured and an unmeasured build share one code
	 * path.
	 */
	class StepTimers
	{
	public:
		StepTimers(BVHBuildMetrics* metrics, cudaStream_t stream)
			: m_metrics(metrics), m_stream(stream) { }

		~StepTimers() noexcept { Reset(); }

		StepTimers(const StepTimers&) = delete;
		StepTimers& operator=(const StepTimers&) = delete;

		BVHBuildMetrics* Metrics() const { return m_metrics; }
		cudaStream_t Stream() const { return m_stream; }

		/*
		 * \brief Reads every recorded pair into its field and releases the events
		 *
		 * The stream must have drained first. Reading an event the GPU has not reached
		 * yet reports cudaErrorNotReady and leaves the field at zero.
		 *
		 * Failures are discarded rather than thrown: by the time this runs the build has
		 * succeeded, and losing a finished BVH because a timer could not be read would be
		 * the worse trade. CudaDiscard also consumes the error, which matters because the
		 * next CUDA call in the process would otherwise inherit it.
		 */
		void Flush() noexcept
		{
			for (size_t i = 0; i < m_count; ++i)
			{
				const Record& record = m_records[i];
				CudaDiscard(cudaEventElapsedTime(&(m_metrics->*record.field), record.start, record.stop));
				CudaDiscard(cudaEventDestroy(record.start));
				CudaDiscard(cudaEventDestroy(record.stop));
			}
			m_count = 0;
		}

	private:
		friend class StepTimer;

		struct Record
		{
			float BVHBuildMetrics::* field;
			cudaEvent_t start;
			cudaEvent_t stop;
		};

		/*
		 * Called from ~StepTimer, so it can neither throw nor allocate. The array is
		 * sized well above the five steps the pipeline has; a build that somehow timed
		 * more than that would drop the extras rather than overrun it.
		 */
		void Add(float BVHBuildMetrics::* field, cudaEvent_t start, cudaEvent_t stop) noexcept
		{
			if (m_count == MaxSteps)
			{
				CudaDiscard(cudaEventDestroy(start));
				CudaDiscard(cudaEventDestroy(stop));
				return;
			}

			m_records[m_count++] = Record{ field, start, stop };
		}

		// Releases whatever Flush did not, which is everything when the build threw
		void Reset() noexcept
		{
			for (size_t i = 0; i < m_count; ++i)
			{
				CudaDiscard(cudaEventDestroy(m_records[i].start));
				CudaDiscard(cudaEventDestroy(m_records[i].stop));
			}
			m_count = 0;
		}

		static constexpr size_t MaxSteps = 8;

		BVHBuildMetrics* m_metrics;
		cudaStream_t m_stream;
		Record m_records[MaxSteps] = {};
		size_t m_count = 0;
	};


	/*
	 * \brief Brackets the enclosing scope with one event pair
	 *
	 * Records the start on construction and the stop on destruction, then hands both to
	 * the StepTimers that outlives it. Nothing is read here -- see StepTimers::Flush.
	 */
	class StepTimer
	{
	public:
		StepTimer(StepTimers& timers, float BVHBuildMetrics::* field)
			: m_timers(timers), m_field(field)
		{
			if (!timers.Metrics())
				return;

			NXB_CUDA_CHECK(cudaEventCreate(&m_start));

			// If anything below throws, the destructor never runs -- this object was never
			// fully constructed -- so the events created so far have to be released by hand
			try
			{
				NXB_CUDA_CHECK(cudaEventCreate(&m_stop));
				NXB_CUDA_CHECK(cudaEventRecord(m_start, timers.Stream()));
			}
			catch (...)
			{
				CudaDiscard(cudaEventDestroy(m_start));
				if (m_stop)
					CudaDiscard(cudaEventDestroy(m_stop));
				throw;
			}
		}

		~StepTimer() noexcept
		{
			if (!m_start)
				return;

			// Deliberately unchecked: a destructor is noexcept, and this one also runs
			// while an exception from the timed scope unwinds, where a second one in
			// flight would terminate. A failure inside the scope has already been
			// reported by the launch that raised it, and the only casualty here is one
			// metrics number.
			if (cudaEventRecord(m_stop, m_timers.Stream()) == cudaSuccess)
			{
				m_timers.Add(m_field, m_start, m_stop);
				return;
			}

			cudaGetLastError();
			CudaDiscard(cudaEventDestroy(m_start));
			CudaDiscard(cudaEventDestroy(m_stop));
		}

		StepTimer(const StepTimer&) = delete;
		StepTimer& operator=(const StepTimer&) = delete;

	private:
		StepTimers& m_timers;
		float BVHBuildMetrics::* m_field;
		cudaEvent_t m_start = nullptr;
		cudaEvent_t m_stop = nullptr;
	};
}
