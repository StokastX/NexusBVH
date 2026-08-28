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
	 * Construction opens the total's pair, so a build's span starts at its first stream
	 * operation. The three calls a build makes are, in order: TimeStep around each step,
	 * StopTotal as the last thing issued on the stream, and Flush after the synchronize.
	 *
	 * Inert when metrics were not requested: no event is created and every operation is a
	 * null check, which is what lets a measured and an unmeasured build share one code
	 * path.
	 */
	class StepTimers
	{
	public:
		StepTimers(BVHBuildMetrics* metrics, cudaStream_t stream)
			: m_metrics(metrics), m_stream(stream)
		{
			if (!m_metrics)
				return;

			// A failure here leaves the object inert rather than failing the build: the
			// timings are diagnostics, and losing them is not worth losing the BVH
			if (!OpenPair(m_totalStart, m_totalStop))
				m_metrics = nullptr;
		}

		~StepTimers() noexcept { Reset(); }

		StepTimers(const StepTimers&) = delete;
		StepTimers& operator=(const StepTimers&) = delete;

		/*
		 * \brief Times the scope the returned object lives in
		 *
		 * The guard is only reachable through here -- its constructor is private -- so it
		 * cannot outlive the StepTimers it reports into, and it is neither copyable nor
		 * movable, so it cannot leave the scope it was created in either. C++17 elides
		 * the copy on the way out, which is what lets it be immovable and still returned.
		 */
		class Scope;
		[[nodiscard]] Scope TimeStep(float BVHBuildMetrics::* field);

		/*
		 * \brief Closes the total's pair
		 *
		 * Must be the last thing the build issues on the stream: the total is measured
		 * rather than summed from the steps, so the difference between it and that sum is
		 * the launch overhead and idle time between kernels.
		 */
		void StopTotal() noexcept
		{
			if (!m_totalStart)
				return;

			if (cudaEventRecord(m_totalStop, m_stream) == cudaSuccess)
			{
				Add(&BVHBuildMetrics::totalTime, m_totalStart, m_totalStop);
				m_totalStart = nullptr;
				m_totalStop = nullptr;
				return;
			}

			cudaGetLastError();
		}

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
		friend class Scope;

		struct Record
		{
			float BVHBuildMetrics::* field;
			cudaEvent_t start;
			cudaEvent_t stop;
		};

		// Creates a pair and records the start. Returns false and leaves both null on
		// failure, which every caller treats as "do not time this".
		bool OpenPair(cudaEvent_t& start, cudaEvent_t& stop) noexcept
		{
			if (cudaEventCreate(&start) != cudaSuccess)
			{
				cudaGetLastError();
				start = nullptr;
				return false;
			}

			if (cudaEventCreate(&stop) != cudaSuccess || cudaEventRecord(start, m_stream) != cudaSuccess)
			{
				cudaGetLastError();
				CudaDiscard(cudaEventDestroy(start));
				if (stop)
					CudaDiscard(cudaEventDestroy(stop));
				start = nullptr;
				stop = nullptr;
				return false;
			}

			return true;
		}

		/*
		 * Called from ~Scope, so it can neither throw nor allocate. The array is sized
		 * well above the five steps and one total the pipeline has; a build that somehow
		 * timed more than that would drop the extras rather than overrun it.
		 */
		void Add(float BVHBuildMetrics::* field, cudaEvent_t start, cudaEvent_t stop) noexcept
		{
			if (m_count == MaxRecords)
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

			if (m_totalStart)
				CudaDiscard(cudaEventDestroy(m_totalStart));
			if (m_totalStop)
				CudaDiscard(cudaEventDestroy(m_totalStop));
			m_totalStart = nullptr;
			m_totalStop = nullptr;
		}

		static constexpr size_t MaxRecords = 8;

		BVHBuildMetrics* m_metrics;
		cudaStream_t m_stream;
		cudaEvent_t m_totalStart = nullptr;
		cudaEvent_t m_totalStop = nullptr;
		Record m_records[MaxRecords] = {};
		size_t m_count = 0;
	};


	/*
	 * \brief Brackets the enclosing scope with one event pair
	 *
	 * Records the start on construction and the stop on destruction, then hands both to
	 * the StepTimers that outlives it. Nothing is read here -- see StepTimers::Flush.
	 */
	class StepTimers::Scope
	{
	public:
		~Scope() noexcept
		{
			if (!m_start)
				return;

			// Deliberately unchecked: a destructor is noexcept, and this one also runs
			// while an exception from the timed scope unwinds, where a second one in
			// flight would terminate. A failure inside the scope has already been
			// reported by the launch that raised it, and the only casualty here is one
			// metrics number.
			if (cudaEventRecord(m_stop, m_timers.m_stream) == cudaSuccess)
			{
				m_timers.Add(m_field, m_start, m_stop);
				return;
			}

			cudaGetLastError();
			CudaDiscard(cudaEventDestroy(m_start));
			CudaDiscard(cudaEventDestroy(m_stop));
		}

		Scope(const Scope&) = delete;
		Scope& operator=(const Scope&) = delete;

	private:
		friend class StepTimers;

		Scope(StepTimers& timers, float BVHBuildMetrics::* field) noexcept
			: m_timers(timers), m_field(field)
		{
			if (timers.m_metrics)
				timers.OpenPair(m_start, m_stop);
		}

		StepTimers& m_timers;
		float BVHBuildMetrics::* m_field;
		cudaEvent_t m_start = nullptr;
		cudaEvent_t m_stop = nullptr;
	};

	inline StepTimers::Scope StepTimers::TimeStep(float BVHBuildMetrics::* field)
	{
		return Scope(*this, field);
	}
}
