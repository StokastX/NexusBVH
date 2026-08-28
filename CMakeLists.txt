cmake_minimum_required(VERSION 3.24)

if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	message(FATAL_ERROR "In-source builds are not allowed. Please make a new build directory and run cmake from there. You may want to remove CMakeCache.txt")
endif()

# `native` is a convenience for whoever is developing NexusBVH, not a decision this project
# gets to make for a consumer. The guard has to sit BEFORE project(), because enabling the
# CUDA language writes a value into the cache -- after that, NOT DEFINED is never true again
# and a guard placed later is a silent no-op. Here it does what it reads like: an explicit
# -DCMAKE_CUDA_ARCHITECTURES=86, or a parent project that already made the choice, wins.
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# CUDA is enabled here rather than in the subdirectories so that there is one place the
# architecture decision can be guarded, and one find_package for every target to share.
# VERSION exists for write_basic_package_version_file; bump it when the public API moves.
project(NexusBVH VERSION 0.1.0 LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# Output directories and a default-on test suite are reasonable when someone is building
# NexusBVH, and rude when someone is merely consuming it: embedded, these would relocate
# the consumer's binaries and inject GPU tests into their `ctest`.
if (PROJECT_IS_TOP_LEVEL)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
endif()

option(NEXUSBVH_BUILD_TESTS "Build the test suite" ${PROJECT_IS_TOP_LEVEL})
option(NEXUSBVH_INSTALL "Generate install and package rules" ${PROJECT_IS_TOP_LEVEL})

add_subdirectory(NexusBVH)

if (NEXUSBVH_BUILD_TESTS)
	enable_testing()
	add_subdirectory(tests)
endif()
