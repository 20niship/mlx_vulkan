cmake_minimum_required(VERSION 3.20)
project(mkx CXX)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Vulkan REQUIRED)

add_library(mkx
    src/mkx/vulkan/vulkan_backend.cpp
)
target_include_directories(mkx PUBLIC src)
target_link_libraries(mkx PUBLIC Vulkan::Vulkan)

include(FetchContent)
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.11
)
FetchContent_MakeAvailable(doctest)

enable_testing()
add_executable(mkx_tests
    tests/test_phase0_graph.cpp
    tests/test_phase1_creation.cpp
    tests/test_phase2_elementwise.cpp
    tests/test_phase3_shapes.cpp
    tests/test_phase4_linalg.cpp
    tests/test_phase5_random.cpp
    tests/test_phase6_transforms.cpp
    tests/test_phase7_fast_kernel.cpp
    tests/test_phase7_kinematics.cpp
    tests/test_phase7_euler.cpp
    tests/test_phase7_euler_devmem.cpp
    tests/test_phase7_forward.cpp
    tests/test_phase7_collision.cpp
    tests/test_phase7_solver.cpp
    tests/test_graph_composite.cpp
)
target_include_directories(mkx_tests PRIVATE tests)
target_link_libraries(mkx_tests PRIVATE mkx doctest::doctest)
include(CTest)
include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(mkx_tests)
