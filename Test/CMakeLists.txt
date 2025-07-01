project(Test LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

file(GLOB src
	src/Main.cpp
	src/CudaUtils.h
)

add_executable(Test ${src})
target_link_libraries(Test
	NexusBVH
)

target_include_directories(Test
	PUBLIC
	src
)

# Set Nexus as startup project in Visual Studio
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT Test)

# Display tree files in Visual Studio
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}
	FILES ${src}
)

