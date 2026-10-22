cmake_minimum_required(VERSION 3.20)

project(HiResFlow VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(HIRESFLOW_BUILD_TESTS "Build HiResFlow tests" ON)
option(HIRESFLOW_ENABLE_ORT "Enable the ONNX Runtime backend when available" ON)

set(HIRESFLOW_DEPS_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/work/deps"
    CACHE PATH "Directory containing downloaded third-party development packages")
set(HIRESFLOW_ORT_ROOT
    "${HIRESFLOW_DEPS_DIR}/onnxruntime-linux-x64-gpu_cuda12-1.31.0"
    CACHE PATH "ONNX Runtime distribution root")

add_library(hiresflow_core
    src/calibration.cpp
    src/comparison.cpp
    src/fusion.cpp
    src/image.cpp
    src/metrics.cpp
    src/planner.cpp
    src/report.cpp
    src/runner.cpp
    src/synthetic_backend.cpp
    src/tiler.cpp
)

target_include_directories(hiresflow_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    SYSTEM PRIVATE
        "${HIRESFLOW_DEPS_DIR}/stb"
)

target_compile_options(hiresflow_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
)
set_source_files_properties(src/image.cpp PROPERTIES
    COMPILE_OPTIONS "$<$<CXX_COMPILER_ID:GNU,Clang>:-Wno-missing-field-initializers>"
)

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_sources(hiresflow_core PRIVATE src/gpu_memory.cpp)
    target_compile_definitions(hiresflow_core PUBLIC HIRESFLOW_HAS_CUDA=1)
    target_link_libraries(hiresflow_core PRIVATE CUDA::cudart)
endif()

add_executable(hiresflow src/main.cpp)
target_link_libraries(hiresflow PRIVATE hiresflow_core)

if(HIRESFLOW_ENABLE_ORT)
    find_path(HIRESFLOW_ORT_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        HINTS "${HIRESFLOW_ORT_ROOT}/include"
        NO_DEFAULT_PATH
    )
    find_library(HIRESFLOW_ORT_LIBRARY
        NAMES onnxruntime
        HINTS "${HIRESFLOW_ORT_ROOT}/lib"
        NO_DEFAULT_PATH
    )
    if(NOT HIRESFLOW_ORT_INCLUDE_DIR AND EXISTS "${HIRESFLOW_DEPS_DIR}/ort_include/onnxruntime_cxx_api.h")
        set(HIRESFLOW_ORT_INCLUDE_DIR "${HIRESFLOW_DEPS_DIR}/ort_include")
    endif()

    if(HIRESFLOW_ORT_INCLUDE_DIR AND HIRESFLOW_ORT_LIBRARY)
        add_library(hiresflow_ort src/onnx_backend.cpp)
        target_include_directories(hiresflow_ort PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${HIRESFLOW_ORT_INCLUDE_DIR}"
        )
        target_link_libraries(hiresflow_ort PUBLIC hiresflow_core PRIVATE "${HIRESFLOW_ORT_LIBRARY}")
        target_compile_definitions(hiresflow_ort PUBLIC HIRESFLOW_HAS_ORT=1)
        target_link_libraries(hiresflow PRIVATE hiresflow_ort)
        set_target_properties(hiresflow PROPERTIES
            BUILD_RPATH "${HIRESFLOW_ORT_ROOT}/lib"
            INSTALL_RPATH "$ORIGIN/../lib"
        )
    else()
        message(WARNING
            "ONNX Runtime not found under ${HIRESFLOW_ORT_ROOT}; "
            "building without the ONNX backend")
    endif()
endif()

if(HIRESFLOW_BUILD_TESTS)
    enable_testing()
    add_executable(hiresflow_tests tests/test_core.cpp)
    target_link_libraries(hiresflow_tests PRIVATE hiresflow_core)
    add_test(NAME hiresflow_core COMMAND hiresflow_tests)
endif()

install(TARGETS hiresflow RUNTIME DESTINATION bin)
install(TARGETS hiresflow_core ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
