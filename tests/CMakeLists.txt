# CUDA is enabled and CUDAToolkit found once, at the top level, and DeviceTraversal.cu
# inherits both. This directory still declares neither LANGUAGES CUDA nor an
# architecture: doing so here would override the one the user asked for.
#
# The .cu is why NEXUSBVH_BUILD_TESTS now needs the CUDA language enabled in an embedding
# project. Nothing in it uses relocatable device code, so it needs no separable
# compilation of its own.

set(NEXUSBVH_TEST_PRIM_COUNT 200000 CACHE STRING
	"Primitive count used by the large scene test cases")

# Shared fixtures and invariant checks, kept free of any test framework so they can be
# reused from anywhere, including the traversal tests
set(supportSrc
	support/BVHChecks.h
	support/DeviceTraversal.h
	support/DeviceTraversal.cu
	support/PrimIntersect.h
	support/BVHChecks.cpp
	support/BVH8Checks.cpp
	support/BVH8Decode.h
	support/BVH8Decode.cpp
	support/Rays.h
	support/Rays.cpp
	support/Scenes.h
	support/Scenes.cpp
	support/TestConfig.h
	support/TraversalChecks.h
	support/TraversalChecks.cpp
)

add_library(NexusBVHTestSupport STATIC ${supportSrc})

target_link_libraries(NexusBVHTestSupport
	PUBLIC
		NXB::NexusBVH
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


set(testSrc
	TestChecks.h
	TestMain.cpp
	TestApi.cpp
	TestBVH2.cpp
	TestBVH8.cpp
	TestTraversal.cpp
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
set_property(DIRECTORY ${PROJECT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT NexusBVHTests)

# Display tree files in Visual Studio
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}
	FILES ${supportSrc} ${testSrc}
)
