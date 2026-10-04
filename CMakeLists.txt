# strata-gemma - a Gemma 4 inference engine for one NVIDIA GPU, forked from Strata (github.com/Niko1221/Strata).
#
#   cmake -B build -DSTRATA_LLAMA_DIR=/path/to/llama.cpp -DCMAKE_CUDA_ARCHITECTURES=86
#   cmake --build build -j
#
# STRATA_LLAMA_DIR is a llama.cpp checkout (Strata pins 3cf03257, 2026-09-20). It supplies ggml's CPU library and
# the MMQ kernels of ggml-cuda (compiled into strata_mmq; ggml-cuda itself is never linked) and, for the server,
# the tokenizer, grammar and image preprocessing (CPU-only builds of libllama / libmtmd).
cmake_minimum_required(VERSION 3.24)
project(strata_gemma VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures" FORCE)
endif()
find_package(CUDAToolkit REQUIRED)

set(STRATA_LLAMA_DIR "" CACHE PATH "llama.cpp checkout (ggml sources)")
if(NOT STRATA_LLAMA_DIR OR NOT EXISTS ${STRATA_LLAMA_DIR}/ggml/src/ggml-cuda/mmq.cuh)
  message(FATAL_ERROR "set STRATA_LLAMA_DIR to a llama.cpp checkout (ggml/src/ggml-cuda/mmq.cuh not found)")
endif()

# llama.cpp, CPU only and static: ggml_type_size & co. for the MMQ glue; the tokenizer, the grammar sampler and
# json_schema_to_grammar (libllama + llama-common) and cpp-httplib for the server. No ggml-cuda: its MMQ kernels are
# compiled into strata_mmq below with the few host symbols they need, which a linked libggml-cuda would duplicate.
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE ON CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
foreach(_o LLAMA_BUILD_TESTS LLAMA_BUILD_TOOLS LLAMA_BUILD_EXAMPLES LLAMA_BUILD_SERVER LLAMA_BUILD_APP LLAMA_BUILD_UI
           LLAMA_OPENSSL LLAMA_SUBPROCESS LLAMA_BUILD_MTMD)
  set(${_o} OFF CACHE BOOL "" FORCE)
endforeach()
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
add_subdirectory(${STRATA_LLAMA_DIR} ${CMAKE_BINARY_DIR}/llama.cpp EXCLUDE_FROM_ALL)

set(_gg ${STRATA_LLAMA_DIR}/ggml)
set(_ggcuda ${_gg}/src/ggml-cuda)
set(STRATA_CUDA_FLAGS -use_fast_math -extended-lambda)   # as ggml-cuda compiles itself

# llama.cpp's MMQ for the prompt path's quantized products
add_library(strata_mmq STATIC src/gemma/mmq.cu src/gemma/ggml_cuda_host.cu ${_ggcuda}/quantize.cu)
foreach(_t q4_0 q5_0 q8_0 q4_k q5_k q6_k)
  target_sources(strata_mmq PRIVATE ${_ggcuda}/template-instances/mmq-instance-${_t}.cu)
endforeach()
target_include_directories(strata_mmq PRIVATE ${_gg}/include ${_gg}/src ${_ggcuda} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(strata_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${STRATA_CUDA_FLAGS}>)
target_link_libraries(strata_mmq PUBLIC ggml-base CUDA::cudart)

# the engine
add_library(strata_gemma STATIC
  src/gemma/model.cpp
  src/gemma/engine.cpp
  src/gemma/mtp.cpp
  src/gemma/kernels.cu
  src/gemma/image.cpp
  src/gemma/vision.cu
  src/kernels/cuda/native_mmvq.cu
  src/kernels/cuda/iq_kernels.cu)
target_include_directories(strata_gemma PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml ${STRATA_LLAMA_DIR}/vendor)
target_compile_options(strata_gemma PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${STRATA_CUDA_FLAGS}>)
target_link_libraries(strata_gemma PUBLIC strata_mmq CUDA::cudart CUDA::cublas)

add_executable(strata-gemma-parity src/gemma/parity_main.cpp)
target_link_libraries(strata-gemma-parity PRIVATE strata_gemma)

add_executable(strata-gemma-vision src/gemma/vision_main.cpp)
target_link_libraries(strata-gemma-vision PRIVATE strata_gemma)

add_executable(strata-gemma-server src/gemma/server_main.cpp)
target_include_directories(strata-gemma-server PRIVATE ${STRATA_LLAMA_DIR}/vendor ${STRATA_LLAMA_DIR}/common)
target_link_libraries(strata-gemma-server PRIVATE strata_gemma llama-common llama cpp-httplib)
