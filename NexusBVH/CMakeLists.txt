# The NexusBVH library target.
#
# This file describes one target and nothing else. It deliberately does not call project(),
# does not enable a language, and does not assign to any CMAKE_* variable: those belong to
# whoever runs cmake. Everything the library needs is attached to the target instead, which
# is both what lets an embedding project keep its own settings and what gives
# install(EXPORT) something to write down.

# Included up here, not down in the install block: CMAKE_INSTALL_INCLUDEDIR is referenced
# by the $<INSTALL_INTERFACE:> generator expression below, and would otherwise be empty.
include(GNUInstallDirs)

set(src
	include/NXB/BVHBuilder.h
	include/NXB/DeviceBuffer.h
	include/NXB/Error.h
	include/NXB/MemoryPool.h
	include/NXB/BVHBuildMetrics.h
	include/NXB/BenchmarkReport.h
	include/NXB/BuildConfig.h
	include/NXB/BVH.h
	include/NXB/BVHCost.h
	include/NXB/Triangle.h
	include/NXB/AABB.h

	src/BVHBuilder.cpp
	src/BVHCost.cpp

	src/Cuda/BinaryBuilder.h
	src/Cuda/BinaryBuilder.cu
	src/Cuda/WideConverter.h
	src/Cuda/WideConverter.cu
	src/Cuda/Setup.h
	src/Cuda/Setup.cu
	src/Cuda/BuildState.h
	src/Cuda/BuilderUtils.h
	src/Cuda/CudaUtils.h
	src/Cuda/Launch.h
	src/Cuda/Eval.h
	src/Cuda/Eval.cu

	src/Math/CudaMath.h
	src/Math/Math.h
)

add_library(NexusBVH STATIC ${src})

# Consumers should spell the target the same way whether they reached it through
# add_subdirectory or find_package. The `::` also makes a typo fail at configure time --
# CMake requires a name containing it to be a target, where a bare name would be passed to
# the linker as -lNexusBVH and fail much later with a worse message.
add_library(NXB::NexusBVH ALIAS NexusBVH)

# STATIC on purpose, and not honouring BUILD_SHARED_LIBS: BVH2::Host hands std::vector
# across the API boundary, DeviceBuffer and MemoryPool are header-only templates
# instantiated in the consumer's TU, and nothing in include/NXB carries dllexport markup.
# A shared build would be an ABI contract this API is not shaped for.
set_target_properties(NexusBVH PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
	POSITION_INDEPENDENT_CODE ON
)

# include/ is the public surface. src/ must stay PRIVATE: it holds the vendored
# helper_math.h, which defines float3 operators and `uint`/`ushort` at global scope, so a
# consumer with its own vector math would get redefinition errors just from our headers.
target_include_directories(NexusBVH
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/src
)

# PUBLIC because the dependency is visible in the public headers, not just in our sources:
# AABB.h includes <cuda_runtime.h>, and DeviceBuffer.h / MemoryPool.h / Error.h call the
# CUDA runtime inline. So a consumer needs both the toolkit's include dir and cudart from
# nothing more than an #include. The imported target carries both and survives being
# written into an export set, which a raw ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} path
# does not.
target_link_libraries(NexusBVH
	PUBLIC
		CUDA::cudart
)

# PUBLIC: the public headers are C++17, so a consumer compiling at an older standard would
# fail. PRIVATE for CUDA: it pins our own nvcc dialect (the default moved from C++14 to
# C++17 between CUDA 11 and 12) without requiring that a consumer have CUDA enabled at all.
target_compile_features(NexusBVH
	PUBLIC
		cxx_std_17
	PRIVATE
		cuda_std_17
)

# CCCL (shipped with CUDA 13.x) hard-errors under MSVC's traditional preprocessor.
# nvcc drives cl.exe as the host compiler, so the flag has to go through -Xcompiler.
if (MSVC)
	target_compile_options(NexusBVH PRIVATE
		$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
	)
endif()

# CUDA optimizations. PRIVATE: how this library is compiled is not a requirement it gets to
# impose on a consumer's own CUDA sources. -rdc=true is not repeated here -- it is what
# CUDA_SEPARABLE_COMPILATION above already adds, and stating it twice let the two disagree
# under MinSizeRel, which matched none of the generator expressions.
target_compile_options(NexusBVH PRIVATE
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:
		--use_fast_math -G -g -lineinfo
	>
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:
		--use_fast_math
	>
)

# CUDA device link-time optimization (DLTO)
target_link_options(NexusBVH PRIVATE
	$<$<CONFIG:Release>:$<DEVICE_LINK:--dlink-time-opt>>
)

# Display tree files in Visual Studio
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}
	FILES ${src}
)

# ---------------------------------------------------------------------------------------
# Install and package
#
# The export set is generated from the usage requirements above, so it is only ever as
# complete as they are -- a requirement left PRIVATE simply does not appear in it.
# ---------------------------------------------------------------------------------------

if (NEXUSBVH_INSTALL)
	include(CMakePackageConfigHelpers)

	set(NXB_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/NexusBVH)

	install(TARGETS NexusBVH
		EXPORT NexusBVHTargets
		ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
	)

	install(DIRECTORY include/NXB
		DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
	)

	install(EXPORT NexusBVHTargets
		FILE NexusBVHTargets.cmake
		NAMESPACE NXB::
		DESTINATION ${NXB_INSTALL_CMAKEDIR}
	)

	# configure_package_config_file rather than configure_file: it is what expands
	# @PACKAGE_INIT@ into the block that derives the install prefix from the config file's
	# own location, which is what makes the installed package relocatable.
	configure_package_config_file(
		${PROJECT_SOURCE_DIR}/cmake/NexusBVHConfig.cmake.in
		${CMAKE_CURRENT_BINARY_DIR}/NexusBVHConfig.cmake
		INSTALL_DESTINATION ${NXB_INSTALL_CMAKEDIR}
	)

	# SameMinorVersion while the API is still moving: 0.1.1 satisfies a request for 0.1,
	# 0.2.0 does not.
	write_basic_package_version_file(
		${CMAKE_CURRENT_BINARY_DIR}/NexusBVHConfigVersion.cmake
		COMPATIBILITY SameMinorVersion
	)

	install(FILES
			${CMAKE_CURRENT_BINARY_DIR}/NexusBVHConfig.cmake
			${CMAKE_CURRENT_BINARY_DIR}/NexusBVHConfigVersion.cmake
		DESTINATION ${NXB_INSTALL_CMAKEDIR}
	)
endif()
