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
  src/backend/vulkan/vulkan_device.cpp
  src/runtime/thread_pool.cpp
  src/runtime/dual_cache.cpp
  src/runtime/planner.cpp
  src/runtime/tokenizer.cpp
  src/runtime/sampler.cpp
  src/runtime/session.cpp
  src/runtime/ngram_draft.cpp
  src/kernels/ops.cpp
  src/kernels/cpu_caps.cpp
  src/kernels/ops_simd_avx2.cpp
  src/kernels/ops_simd_avx512.cpp
  src/kernels/ops_fused.cpp
  src/kernels/ops_vision.cpp
  src/kernels/cuda/cuda_ref.cpp
  src/kernels/cuda/cuda_engine_stub.cpp
  src/api/rapidllm_c.cpp
  src/server/http_serve.cpp
)
if(EXISTS "${CMAKE_SOURCE_DIR}/src/runtime/image_io.cpp")
  list(APPEND RAPIDLLM_SOURCES src/runtime/image_io.cpp)
endif()

add_library(rapidllm_core STATIC ${RAPIDLLM_SOURCES})
target_link_libraries(rapidllm_core PUBLIC Threads::Threads)
if(NOT WIN32)
  target_link_libraries(rapidllm_core PUBLIC ${CMAKE_DL_LIBS})
endif()
target_include_directories(rapidllm_core PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/third_party
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
    src/kernels/cuda/gemv_q6_t2.cu
    src/kernels/cuda/gemv_q6_t2_1row.cu
    src/kernels/cuda/gemv_q6_t2_hbm.cu
    src/kernels/cuda/gemv_q4_t2_dual_hbm.cu
    src/kernels/cuda/gemv_q6_t4.cu
    src/kernels/cuda/gemv_t4_1row.cu
    src/kernels/cuda/gemv_t4_pipe.cu
    src/kernels/cuda/gemv_t4_q6_ku.cu
    src/kernels/cuda/gemv_t4_q6_t12tile.cu
    src/kernels/cuda/gemv_t4_q6_xs.cu
    src/kernels/cuda/gemv_t4_occ.cu
    src/kernels/cuda/gemv_t4_q6_pipe.cu
    src/kernels/cuda/gemv_t4_q8.cu
    src/kernels/cuda/gemv_t4_q6_2row.cu
    src/kernels/cuda/gemv_t4_q4_2row.cu
    src/kernels/cuda/gemv_t4_q4_sx.cu
    src/kernels/cuda/gemv_t4_q4_p3.cu
    src/kernels/cuda/gemv_t4_u2.cu
    src/kernels/cuda/gemv_t4_q4_r8.cu
    src/kernels/cuda/gemv_t4_q6_q8.cu
    src/kernels/cuda/gemv_t4_q4_x2.cu
    src/kernels/cuda/gemv_t4_q6_ho.cu
    src/kernels/cuda/gemv_t4_q6q4.cu
    src/kernels/cuda/gemv_t4_q4_tc.cu
    src/kernels/cuda/gemv_t4_q6_i8.cu
    src/kernels/cuda/gemv_t4_q4_p2.cu
    src/kernels/cuda/gemv_t4_q4_xs.cu
    src/kernels/cuda/gemv_t4_ur2.cu
    src/kernels/cuda/gemv_t4_q4_mmq.cu
    src/kernels/cuda/gemv_t4_q6_gm.cu
    src/kernels/cuda/gemv_t4_sk.cu
    src/kernels/cuda/gemv_t4_q6_ilp.cu
    src/kernels/cuda/gemv_q6_t1_2row.cu
    src/kernels/cuda/gemv_q4_t2_dual_1row.cu
    src/kernels/cuda/gemv_q6_t2_2row.cu
    src/kernels/cuda/gemv_q6_t2_splitk.cu
    src/kernels/cuda/mtp_attn_decode.cu
    src/kernels/cuda/gemv_q6_t12.cu
    src/kernels/cuda/gemv_q6_t12_1row.cu
    src/kernels/cuda/gemv_q6_t12_smemx.cu
    src/kernels/cuda/gemv_q4_t12_pipe.cu
    src/kernels/cuda/gemv_q4_t12_2row_pipe.cu
    src/kernels/cuda/gemv_q4_t16_pipe.cu
    src/kernels/cuda/gemv_q6_t16_1row.cu
    src/kernels/cuda/gemv_q4_t12_dual.cu
    src/kernels/cuda/gdn_t12.cu
    src/kernels/cuda/cuda_engine.cu
  )
  target_compile_definitions(rapidllm_core PUBLIC RAPIDLLM_WITH_CUDA=1)
  set(_fi_home "$ENV{FLASHINFER_HOME}")
  if(NOT _fi_home)
    set(_fi_home "/home/znsoft/vllm-env/lib/python3.12/site-packages/flashinfer/data")
  endif()
  if(EXISTS "${_fi_home}/include/flashinfer/attention/prefill.cuh")
    target_sources(rapidllm_core PRIVATE src/kernels/cuda/flashinfer_attn.cu)
    target_include_directories(rapidllm_core PRIVATE "${_fi_home}/include")
    if(EXISTS "${_fi_home}/cutlass/include")
      target_include_directories(rapidllm_core PRIVATE "${_fi_home}/cutlass/include")
    endif()
    if(EXISTS "${_fi_home}/cccl/libcudacxx/include")
      target_include_directories(rapidllm_core PRIVATE "${_fi_home}/cccl/libcudacxx/include")
    endif()
    target_compile_definitions(rapidllm_core PRIVATE RAPIDLLM_FLASHINFER=1)
    set_source_files_properties(src/kernels/cuda/flashinfer_attn.cu PROPERTIES
      CUDA_ARCHITECTURES "89")
    message(STATUS "FlashInfer prefill: ${_fi_home}")
  endif()
  set_target_properties(rapidllm_core PROPERTIES CUDA_ARCHITECTURES "75;86;89")
  # Legacy default stream (stream 0) cannot be captured; per-thread stream can.
  target_compile_options(rapidllm_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread;--use_fast_math;-Xptxas=-O3>)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(rapidllm_core PUBLIC CUDA::cudart CUDA::cublas)
  if(TARGET CUDA::cublasLt)
    target_link_libraries(rapidllm_core PUBLIC CUDA::cublasLt)
  endif()
endif()

add_executable(rapidllm src/cli/main.cpp)
target_link_libraries(rapidllm PRIVATE rapidllm_core)
if(WIN32)
  target_link_libraries(rapidllm PRIVATE ws2_32)
  target_link_libraries(rapidllm_core PUBLIC ws2_32)
endif()
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
rapidllm_test(test_batch tests/test_batch.cpp)
rapidllm_test(test_protocol tests/test_protocol.cpp)
rapidllm_test(test_ngram_draft tests/test_ngram_draft.cpp)
rapidllm_test(test_vulkan tests/test_vulkan.cpp)
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_vision.cpp")
  rapidllm_test(test_vision tests/test_vision.cpp)
endif()

enable_testing()
add_test(NAME ir COMMAND test_ir)
add_test(NAME loader COMMAND test_loader)
add_test(NAME hybrid COMMAND test_hybrid)
add_test(NAME simd_bench COMMAND test_simd_bench)
add_test(NAME nv COMMAND test_nv)
add_test(NAME spec COMMAND test_spec)
add_test(NAME batch COMMAND test_batch)
add_test(NAME cuda_decode COMMAND test_cuda_decode)
add_test(NAME protocol COMMAND test_protocol)
add_test(NAME ngram_draft COMMAND test_ngram_draft)
add_test(NAME vision COMMAND test_vision)
