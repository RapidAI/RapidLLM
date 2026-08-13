cmake_minimum_required(VERSION 3.20)
project(RapidLLM VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(RAPIDLLM_WITH_CUDA "Build NVIDIA .cu kernels" OFF)

if(MSVC)
  add_compile_options(/W3 /utf-8 /Zc:__cplusplus)
else()
  add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_package(Threads REQUIRED)

set(RAPIDLLM_SOURCES
  src/version.cpp
  src/ir/model_desc.cpp
  src/frontend/json_mini.cpp
  src/frontend/name_map.cpp
  src/frontend/safetensors.cpp
  src/frontend/hf_loader.cpp
  src/frontend/gguf_loader.cpp
  src/frontend/weight_store.cpp
  src/backend/cpu/cpu_device.cpp
  src/runtime/thread_pool.cpp
  src/runtime/dual_cache.cpp
  src/runtime/planner.cpp
  src/runtime/tokenizer.cpp
  src/runtime/sampler.cpp
  src/runtime/session.cpp
  src/kernels/ops.cpp
  src/kernels/cpu_caps.cpp
  src/kernels/ops_simd_avx2.cpp
  src/kernels/ops_simd_avx512.cpp
  src/kernels/ops_fused.cpp
  src/kernels/cuda/cuda_ref.cpp
  src/kernels/cuda/cuda_engine_stub.cpp
  src/api/rapidllm_c.cpp
)

add_library(rapidllm_core STATIC ${RAPIDLLM_SOURCES})
target_link_libraries(rapidllm_core PUBLIC Threads::Threads)
target_include_directories(rapidllm_core PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src
)
target_compile_definitions(rapidllm_core PUBLIC RAPIDLLM_VERSION="${PROJECT_VERSION}")
# ISA kernels live in their own TUs so AVX-512 opcodes are never emitted on the AVX2-only path.
if(MSVC)
  set_source_files_properties(src/kernels/ops_simd_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  set_source_files_properties(src/kernels/ops_simd_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
else()
  set_source_files_properties(src/kernels/ops_simd_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  set_source_files_properties(src/kernels/ops_simd_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512dq;-mavx512bw;-mavx512vl;-mfma;-mf16c")
endif()
if(RAPIDLLM_WITH_CUDA)
  enable_language(CUDA)
  target_sources(rapidllm_core PRIVATE
    src/kernels/cuda/gemv_dp4a.cu
    src/kernels/cuda/cuda_engine.cu
  )
  target_compile_definitions(rapidllm_core PUBLIC RAPIDLLM_WITH_CUDA=1)
  set_target_properties(rapidllm_core PROPERTIES CUDA_ARCHITECTURES "75;86;89")
  # Legacy default stream (stream 0) cannot be captured; per-thread stream can.
  target_compile_options(rapidllm_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(rapidllm_core PUBLIC CUDA::cudart CUDA::cublas)
endif()

add_executable(rapidllm src/cli/main.cpp)
target_link_libraries(rapidllm PRIVATE rapidllm_core)
set_target_properties(rapidllm PROPERTIES OUTPUT_NAME rapidllm)

add_executable(rapidllm_version src/cli/version_main.cpp)
target_link_libraries(rapidllm_version PRIVATE rapidllm_core)

set(FIXTURE_DIR ${CMAKE_BINARY_DIR}/fixtures)
add_custom_command(
  OUTPUT
    ${FIXTURE_DIR}/tiny_hybrid/config.json
    ${FIXTURE_DIR}/tiny_hybrid/model.safetensors
    ${FIXTURE_DIR}/tiny_hybrid/golden.json
    ${FIXTURE_DIR}/tiny_hybrid.gguf
    ${FIXTURE_DIR}/bad_fp8/config.json
    ${FIXTURE_DIR}/bad_gguf.gguf
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/python/goldens/tiny_hybrid.py
          --out ${FIXTURE_DIR}
  DEPENDS ${CMAKE_SOURCE_DIR}/python/goldens/tiny_hybrid.py
  COMMENT "Generating tiny hybrid fixtures and goldens"
  VERBATIM
)
add_custom_target(rapidllm_fixtures DEPENDS
  ${FIXTURE_DIR}/tiny_hybrid/golden.json
  ${FIXTURE_DIR}/tiny_hybrid.gguf
)

function(rapidllm_test name src)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE rapidllm_core)
  target_compile_definitions(${name} PRIVATE RAPIDLLM_FIXTURE_DIR="${FIXTURE_DIR}")
  add_dependencies(${name} rapidllm_fixtures)
endfunction()

rapidllm_test(test_ir tests/test_ir.cpp)
rapidllm_test(test_loader tests/test_loader.cpp)
rapidllm_test(test_hybrid tests/test_hybrid.cpp)
rapidllm_test(test_simd_bench tests/test_simd_bench.cpp)
rapidllm_test(test_nv tests/test_nv.cpp)
rapidllm_test(test_spec tests/test_spec.cpp)
rapidllm_test(test_cuda_decode tests/test_cuda_decode.cpp)

enable_testing()
add_test(NAME ir COMMAND test_ir)
add_test(NAME loader COMMAND test_loader)
add_test(NAME hybrid COMMAND test_hybrid)
add_test(NAME simd_bench COMMAND test_simd_bench)
add_test(NAME nv COMMAND test_nv)
add_test(NAME spec COMMAND test_spec)
add_test(NAME cuda_decode COMMAND test_cuda_decode)
