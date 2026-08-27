#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace NXB
{
	/* \brief Thrown by every NexusBVH entry point when a CUDA call fails
	 *
	 * Every device allocation a build had made is released before the exception leaves the
	 * library, so a caught CudaError leaks nothing and the caller may retry -- with fewer
	 * primitives, say, after an out of memory.
	 */
	struct CudaError : std::runtime_error
	{
		CudaError(cudaError_t errorCode, const std::string& message)
			: std::runtime_error(message), code(errorCode) { }

		// So a caller can branch on the reason rather than parse the message
		cudaError_t code;
	};

	inline void CudaCheck(cudaError_t result, const char* expression, const char* file, int32_t line)
	{
		if (result == cudaSuccess)
			return;

		/*
		 * Consume the failure before leaving. It is otherwise left pending in the
		 * runtime's per thread error slot, and the next CUDA call that inspects that
		 * slot inherits it: cub::Debug() substitutes cudaGetLastError() for a call
		 * that returned success, so a leaked error here makes CUB read the following
		 * build's CurrentDevice() as -1, and its radix sort then asks for zero bytes
		 * of temporary storage and silently sorts nothing.
		 *
		 * The error is already carried by the exception, so nothing is lost by
		 * clearing it, and cudaGetLastError only reads and clears that slot -- it
		 * does not synchronize.
		 */
		cudaGetLastError();

		throw CudaError(result, std::string(cudaGetErrorName(result)) + " (" + cudaGetErrorString(result) + ")"
			+ " at " + file + ":" + std::to_string(line) + " in '" + expression + "'");
	}


	/*
	 * For the paths that must not throw -- destructors, and anything running during stack
	 * unwinding -- where the only options are to swallow the failure or to terminate.
	 *
	 * Swallowing the return value is NOT enough on its own. The failure also sits in the
	 * runtime's per thread error slot, and the next call that inspects that slot inherits
	 * it: cub::Debug() substitutes cudaGetLastError() for a call that returned success, so
	 * a leaked error here makes the following build read CurrentDevice() as -1 and fail
	 * with cudaErrorInvalidDevice from somewhere entirely unrelated. Consuming it is what
	 * keeps the failure local to the object that caused it.
	 */
	inline void CudaDiscard(cudaError_t result)
	{
		if (result != cudaSuccess)
			cudaGetLastError();
	}
}

#define NXB_CUDA_CHECK(val) ::NXB::CudaCheck((val), #val, __FILE__, __LINE__)
