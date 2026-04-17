cmake_minimum_required(VERSION 3.24)
project(LuminaGI_CudaRef LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 4080 Laptop = Ada Lovelace (compute capability 8.9)
set(CMAKE_CUDA_ARCHITECTURES 89)

if(MSVC)
    add_compile_options(/W3)
endif()

# ---- hello_cuda: smoke test ----
add_executable(hello_cuda src/hello_cuda.cu)
set_target_properties(hello_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ---- cuda_ref: main path tracer (added in later tasks) ----
# add_executable(cuda_ref
#     src/main.cpp
#     src/scene_loader.cpp
#     src/pathtracer.cu
# )
