#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace NXB::Test
{
	/*
	 * The library's own CUDA_CHECK calls exit(99). That is a reasonable-enough default
	 * for an application, but inside a test runner it takes the whole run down with the
	 * first failing case and reports nothing about the rest. Throwing instead lets the
	 * framework record one failed test and carry on.
	 */
	struct CudaError : std::runtime_error
	{
		explicit CudaError(const std::string& message) : std::runtime_error(message) { }
	};

	inline void CudaCheck(cudaError_t result, const char* expression, const char* file, int32_t line)
	{
		if (result == cudaSuccess)
			return;

		throw CudaError(std::string(cudaGetErrorName(result)) + " (" + cudaGetErrorString(result) + ")"
			+ " at " + file + ":" + std::to_string(line) + " in '" + expression + "'");
	}
}

#define NXB_TEST_CUDA_CHECK(val) ::NXB::Test::CudaCheck((val), #val, __FILE__, __LINE__)
