project(Test)

if(NOT NEXUSBVH_BUILD_TESTS)

	enable_language(CUDA)
	set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

	file(GLOB src
		src/Main.cpp
	)

	add_executable(Test ${src})
	target_link_libraries(Test
		NexusBVH
	)

	target_include_directories(Test
		PUBLIC
		src
		../NexusBVH/src
	)

	target_include_directories(Test
		PUBLIC
		src
	)

	# CUDA optimizations
	target_compile_options(Test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
		--use_fast_math
		-rdc=true
	>)

	# Set Nexus as startup project in Visual Studio
	set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT Test)

	# Display tree files in Visual Studio
	source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}
		FILES ${src}
	)
endif()

