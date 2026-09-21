cmake_minimum_required(VERSION 3.20)
project(memvanta_cpu VERSION 0.7.2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MEMVANTA_NATIVE "Enable -march=native for Release builds" ON)
option(MEMVANTA_BUILD_FUZZER "Build the Clang/libFuzzer GGUF parser target" OFF)
option(MEMVANTA_ENABLE_IPO "Enable Release interprocedural optimization when supported" ON)

# Keep the performance candidate compiler-driven and RSS-neutral: IPO/LTO can
# inline hot projection call boundaries without adding a persistent weight cache
# or changing the runtime memory model. Existing 7B same-runner A/B workflows
# remain the promotion gate for any throughput claim.
if(MEMVANTA_ENABLE_IPO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT MEMVANTA_IPO_SUPPORTED OUTPUT MEMVANTA_IPO_ERROR)
  if(MEMVANTA_IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
  else()
    message(WARNING "IPO/LTO unavailable; continuing without it: ${MEMVANTA_IPO_ERROR}")
  endif()
endif()

add_library(memvanta_core
    src/common.cpp
    src/mmap_file.cpp
    src/tensor_store.cpp
    src/lru_cache.cpp
    src/prefetcher.cpp
    src/prefetch_policy.cpp
    src/runtime.cpp
    src/matmul.cpp
    src/quant.cpp
    src/quant_kernels.cpp
    src/llama2.cpp
    src/gguf.cpp
    src/gguf_kernels.cpp
    src/worker_pool.cpp
    src/llama_model.cpp)

target_include_directories(memvanta_core PUBLIC include)
target_compile_options(memvanta_core PRIVATE -Wall -Wextra -Wpedantic $<$<CONFIG:Release>:-O3>)

# The 7B profile is dominated by the compact Q4/Q8 projection loops in
# gguf_kernels.cpp. Ask GCC/Clang to unroll only that Release translation unit
# so hot loops can expose more independent loads/FMAs without enabling
# fast-math or broad reassociation across the runtime. fno-math-errno removes
# unnecessary errno conservatism around scalar quantization helpers while
# preserving IEEE arithmetic and the existing correctness gates.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set_source_files_properties(src/gguf_kernels.cpp PROPERTIES
    COMPILE_OPTIONS "$<$<CONFIG:Release>:-funroll-loops>;$<$<CONFIG:Release>:-fno-math-errno>")
endif()

# ELF symbol interposition is not used by the static core library. Disabling
# semantic interposition gives the optimizer more freedom across hot projection
# call sites without changing numerical semantics or allocating additional RAM.
if(UNIX AND NOT APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(memvanta_core PRIVATE
    $<$<CONFIG:Release>:-fno-semantic-interposition>)
endif()

if(MEMVANTA_NATIVE)
  target_compile_options(memvanta_core PRIVATE $<$<CONFIG:Release>:-march=native>)
endif()

find_package(Threads REQUIRED)
find_package(OpenMP)
target_link_libraries(memvanta_core PUBLIC Threads::Threads)
if(OpenMP_CXX_FOUND)
  target_link_libraries(memvanta_core PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(memvanta_core PUBLIC MEMVANTA_USE_OPENMP=1)
endif()

add_executable(memvanta src/main.cpp)
target_link_libraries(memvanta PRIVATE memvanta_core)

add_executable(memvanta_bench benchmarks/streaming_bench.cpp)
target_link_libraries(memvanta_bench PRIVATE memvanta_core)

add_executable(memvanta_kernel_bench benchmarks/kernel_bench.cpp)
target_link_libraries(memvanta_kernel_bench PRIVATE memvanta_core)

add_executable(memvanta_llama_bench benchmarks/llama_style_bench.cpp)
target_link_libraries(memvanta_llama_bench PRIVATE memvanta_core)

enable_testing()

add_executable(memvanta_tests tests/tests.cpp)
target_link_libraries(memvanta_tests PRIVATE memvanta_core)
add_test(NAME memvanta_tests COMMAND memvanta_tests)

add_executable(memvanta_fp16_tests tests/fp16_tests.cpp)
target_link_libraries(memvanta_fp16_tests PRIVATE memvanta_core)
add_test(NAME memvanta_fp16_tests COMMAND memvanta_fp16_tests)

add_executable(memvanta_gguf_limits_tests tests/gguf_limits_tests.cpp)
target_link_libraries(memvanta_gguf_limits_tests PRIVATE memvanta_core)
add_test(NAME memvanta_gguf_limits_tests COMMAND memvanta_gguf_limits_tests)

add_executable(memvanta_concurrency_stress_tests tests/concurrency_stress_tests.cpp)
target_link_libraries(memvanta_concurrency_stress_tests PRIVATE memvanta_core)
add_test(NAME memvanta_concurrency_stress_tests COMMAND memvanta_concurrency_stress_tests)

add_executable(memvanta_prefetch_policy_tests tests/prefetch_policy_tests.cpp)
target_link_libraries(memvanta_prefetch_policy_tests PRIVATE memvanta_core)
add_test(NAME memvanta_prefetch_policy_tests COMMAND memvanta_prefetch_policy_tests)

set_tests_properties(
  memvanta_tests
  memvanta_fp16_tests
  memvanta_gguf_limits_tests
  memvanta_prefetch_policy_tests
  PROPERTIES TIMEOUT 120)

# The concurrency target intentionally performs repeated lifecycle stress and can
# take longer on contended hosted runners. Keep the full stress coverage while
# retaining a finite bound that still catches true hangs.
set_tests_properties(memvanta_concurrency_stress_tests PROPERTIES TIMEOUT 300)

add_executable(memvanta_real src/real_main.cpp)
target_link_libraries(memvanta_real PRIVATE memvanta_core)

add_executable(memvanta_real_bench benchmarks/real_model_bench.cpp)
target_link_libraries(memvanta_real_bench PRIVATE memvanta_core)

add_executable(memvanta_issue12_real_prefetch_ab benchmarks/issue12_real_prefetch_ab.cpp)
target_link_libraries(memvanta_issue12_real_prefetch_ab PRIVATE memvanta_core)

add_executable(memvanta_profile benchmarks/profile_bench.cpp)
target_link_libraries(memvanta_profile PRIVATE memvanta_core)

add_executable(memvanta_llama2c_bench benchmarks/llama2c_bench.cpp)
target_link_libraries(memvanta_llama2c_bench PRIVATE memvanta_core)

add_executable(memvanta_auto_tune benchmarks/auto_tune.cpp)
target_link_libraries(memvanta_auto_tune PRIVATE memvanta_core)

add_executable(memvanta_eval benchmarks/eval_bench.cpp)
target_link_libraries(memvanta_eval PRIVATE memvanta_core)

add_executable(memvanta_tokenize tools_tokenize.cpp)
target_link_libraries(memvanta_tokenize PRIVATE memvanta_core)

add_executable(memvanta_gguf_inspect tools_gguf_inspect.cpp)
target_link_libraries(memvanta_gguf_inspect PRIVATE memvanta_core)

add_executable(memvanta_tokenizer_tests tests/tokenizer_tests.cpp)
target_link_libraries(memvanta_tokenizer_tests PRIVATE memvanta_core)
add_test(NAME memvanta_tokenizer_tests COMMAND memvanta_tokenizer_tests
  ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer_fixtures/gpt2_smollm_tokenizer.gguf
  ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer_fixtures/llama_sentencepiece_tokenizer.gguf)
set_tests_properties(memvanta_tokenizer_tests PROPERTIES TIMEOUT 120)

# Several benchmark workflows intentionally build a narrow target set before
# running the repository-wide CTest gate. Keep memvanta_tests as the test-suite
# build anchor so targeted builds cannot leave newly registered executables
# missing.
add_dependencies(memvanta_tests
  memvanta_fp16_tests
  memvanta_gguf_limits_tests
  memvanta_concurrency_stress_tests
  memvanta_prefetch_policy_tests
  memvanta_tokenizer_tests)

if(MEMVANTA_BUILD_FUZZER)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "MEMVANTA_BUILD_FUZZER requires Clang/libFuzzer")
  endif()
  add_executable(memvanta_gguf_fuzz
    fuzz/gguf_fuzz.cpp
    src/gguf.cpp
    src/mmap_file.cpp)
  target_include_directories(memvanta_gguf_fuzz PRIVATE include)
  target_compile_options(memvanta_gguf_fuzz PRIVATE
    -fsanitize=fuzzer,address,undefined
    -fno-omit-frame-pointer
    -Wall -Wextra -Wpedantic)
  target_link_options(memvanta_gguf_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
endif()
