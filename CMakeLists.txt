cmake_minimum_required(VERSION 3.22)

if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	message(FATAL_ERROR "In-source builds are not allowed. Please make a new build directory and run cmake from there. You may want to remove CMakeCache.txt")
endif()

project("NexusBVH")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries")

option(NEXUSBVH_BUILD_TESTS "Build the test project" ON)

add_subdirectory(NexusBVH)

if (NEXUSBVH_BUILD_TESTS)
	add_subdirectory(Test)
endif()
