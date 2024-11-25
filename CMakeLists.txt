project(bvh-builder)

enable_language(CUDA)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

file(GLOB src
    src/BVHBuilder.h
    src/bvhbuilder.cpp
    src/bvhbuilder.cu
)

add_library(bvh-builder STATIC ${src})

target_include_directories(bvh-builder
    PUBLIC
    .
)

# CUDA optimizations
target_compile_options(bvh-builder PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
	--use_fast_math
	-rdc=true
>)

# CUDA device link-time optimization (DLTO)
target_link_options(bvh-builder PRIVATE $<DEVICE_LINK:
	--dlink-time-opt
>)

# Display tree files in Visual Studio
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}
	FILES ${src}
)

