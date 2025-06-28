project(NexusBVH)

enable_language(CUDA)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

file(GLOB src
    include/NXB/BVHBuilder.h
    include/NXB/BVHBuildMetrics.h
    include/NXB/BuildConfig.h
    include/NXB/BVH.h
    include/NXB/Triangle.h
    include/NXB/AABB.h
    include/NXB/Math/CudaMath.h

    src/BVHBuilder.cpp

    src/Cuda/BinaryBuilder.h
    src/Cuda/BinaryBuilder.cu
    src/Cuda/WideConverter.h
    src/Cuda/WideConverter.cu
    src/Cuda/Setup.h
    src/Cuda/Setup.cu
    src/Cuda/BuildState.h
    src/Cuda/BuilderUtils.h
    src/Cuda/CudaUtils.h
    src/Cuda/Eval.h
    src/Cuda/Eval.cu

    src/Math/Math.h
)

add_library(NexusBVH STATIC ${src})
set_target_properties(NexusBVH PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(NexusBVH
    PUBLIC
		include
    PRIVATE
        src
)

# CUDA optimizations
target_compile_options(NexusBVH PRIVATE
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:
    --use_fast_math -G -g -lineinfo -rdc=true
  >
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:
    --use_fast_math -rdc=true
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

