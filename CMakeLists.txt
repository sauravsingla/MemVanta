cmake_minimum_required(VERSION 3.20)
project(memvanta_cpu VERSION 0.7.2 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(MEMVANTA_NATIVE "Enable -march=native" ON)
add_library(memvanta_core
    src/mmap_file.cpp
    src/tensor_store.cpp
    src/lru_cache.cpp
    src/prefetcher.cpp
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
target_compile_options(memvanta_core PRIVATE -O3 -Wall -Wextra -Wpedantic)
# The GGUF projection kernels are dominated by small fixed-trip inner loops.
# Let the compiler unroll only this translation unit so the 7B FFN A/B gate
# can validate whether reduced loop/control overhead improves the Q4 hotspot
# without changing numerical behavior or broadening code-size impact globally.
set_source_files_properties(src/gguf_kernels.cpp PROPERTIES COMPILE_OPTIONS "-funroll-loops")
if(MEMVANTA_NATIVE)
  target_compile_options(memvanta_core PRIVATE -march=native)
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
add_executable(memvanta_tests tests/tests.cpp)
target_link_libraries(memvanta_tests PRIVATE memvanta_core)
enable_testing()
add_test(NAME memvanta_tests COMMAND memvanta_tests)

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

add_executable(memvanta_tokenizer_tests tests/tokenizer_tests.cpp)
target_link_libraries(memvanta_tokenizer_tests PRIVATE memvanta_core)
add_test(NAME memvanta_tokenizer_tests COMMAND memvanta_tokenizer_tests
  ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer_fixtures/gpt2_smollm_tokenizer.gguf
  ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer_fixtures/llama_sentencepiece_tokenizer.gguf)
