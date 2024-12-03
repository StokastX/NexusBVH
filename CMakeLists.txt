project(NexusBVH)

enable_language(CUDA)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

file(GLOB src
    src/BVHBuilder.h
    src/BVHBuilder.cpp
    src/BVH/BVH.h

    src/Cuda/BinaryBuilder.h
    src/Cuda/BinaryBuilder.cu
    src/Cuda/WideConverter.h
    src/Cuda/WideConverter.cu
    src/Cuda/Setup.h
    src/Cuda/Setup.cu
    src/Cuda/BuilderUtils.h

    src/Math/CudaMath.h
    src/Math/AABB.h
)

add_library(NexusBVH STATIC ${src})

target_include_directories(NexusBVH
    PUBLIC
    src
)

# CUDA optimizations
target_compile_options(NexusBVH PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
	--use_fast_math
	-rdc=true
>)

# CUDA device link-time optimization (DLTO)
target_link_options(NexusBVH PRIVATE $<DEVICE_LINK:
	--dlink-time-opt
>)

# Display tree files in Visual Studio
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}
	FILES ${src}
)

