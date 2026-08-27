cmake_minimum_required(VERSION 3.24)
project(NexusBVHTests LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES native)

set(NEXUSBVH_TEST_PRIM_COUNT 200000 CACHE STRING
	"Primitive count used by the large scene test cases")

# Shared fixtures and invariant checks, kept free of any test framework so they can be
# reused from anywhere -- including, later, the traversal tests
file(GLOB supportSrc
	support/BVHChecks.h
	support/BVHChecks.cpp
	support/BVH8Checks.cpp
	support/BVH8Decode.h
	support/BVH8Decode.cpp
	support/Scenes.h
	support/Scenes.cpp
	support/TestConfig.h
)

add_library(NexusBVHTestSupport STATIC ${supportSrc})

target_link_libraries(NexusBVHTestSupport
	PUBLIC
		NexusBVH
		CUDA::cudart
)

target_include_directories(NexusBVHTestSupport
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(NexusBVHTestSupport
	PUBLIC
		NXB_TEST_PRIM_COUNT=${NEXUSBVH_TEST_PRIM_COUNT}
)


file(GLOB testSrc
	TestChecks.h
	TestMain.cpp
	TestApi.cpp
	TestBVH2.cpp
	TestBVH8.cpp
)

add_executable(NexusBVHTests ${testSrc})

target_link_libraries(NexusBVHTests
	PRIVATE
		NexusBVHTestSupport
)

# doctest.h forward declares std::tuple, which MSVC flags under C5285. It is vendored
# third party code, so silence it here rather than patching the header.
if (MSVC)
	target_compile_options(NexusBVHTests PRIVATE /wd5285)
endif()

# Registers every TEST_CASE as its own ctest entry, so a failure names the case and
# `ctest -R` can select one while iterating
include(${CMAKE_CURRENT_SOURCE_DIR}/vendor/doctest.cmake)

# Labels come from the doctest TEST_SUITE a case sits in ("fast" / "slow"), so the
# decision lives next to the test rather than in a filter expression here. The tiny
# scenes are the inner loop (`ctest -L fast`); the large ones are worth the wait before
# a commit but not on every edit.
doctest_discover_tests(NexusBVHTests
	ADD_LABELS 1
)

# ...but that runs every case in its OWN process, which structurally cannot catch state
# one case leaks into the next: a CUDA error left pending in the runtime's per thread slot,
# a context left in a bad state, a pool destroyed with allocations outstanding. Running the
# whole binary once, the way Visual Studio's F5 does, is the only thing that sees it.
add_test(NAME "All cases in one process" COMMAND NexusBVHTests)
set_tests_properties("All cases in one process" PROPERTIES LABELS "slow")

# Set the test runner as startup project in Visual Studio
set_property(DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT NexusBVHTests)

# Display tree files in Visual Studio
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}
	FILES ${supportSrc} ${testSrc}
)
