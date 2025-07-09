
file(GLOB   GGML_HEADERS_CUDA "../ggml-cuda/*.cuh")
list(APPEND GGML_HEADERS_CUDA "../../include/ggml-cuda.h")
file(GLOB   GGML_SOURCES_CUDA "../ggml-cuda/*.cu")
file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-mma*.cu")
list(APPEND GGML_SOURCES_CUDA ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/mmq*.cu")
list(APPEND GGML_SOURCES_CUDA ${SRCS})

if (GGML_CUDA_FA_ALL_QUANTS)
file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-vec*.cu")
list(APPEND GGML_SOURCES_CUDA ${SRCS})
add_compile_definitions(GGML_CUDA_FA_ALL_QUANTS)
else()
file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-vec*q4_0-q4_0.cu")
list(APPEND GGML_SOURCES_CUDA ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-vec*q8_0-q8_0.cu")
list(APPEND GGML_SOURCES_CUDA ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-vec*f16-f16.cu")
list(APPEND GGML_SOURCES_CUDA ${SRCS})
endif()

set(SDK_DIR CACHE PATH "Path to the SDK diredctory")

if(NOT DEFINED SDK_DIR)
    if(DEFINED ENV{SDK DIR})
	set(SDK_DIR $ENV{SDK_DIR}/)
    else()
	message(FATAL_ERROR "SDK_DIR must be set via -DSDK_DIR=or env variable")
    endif()
else()
    set(SDK_DIR ${SDK_DIR}/)
endif()
if(WIN32)
    set(CMAKE_CXX_COMPILER "clang-cl.exe")
else()
    set(CMAKE_CXX_COMPILER ${SDK_DIR}/bin/dlcc)
    set(CMAKE_C_COMPILER ${SDK_DIR}/bin/dlcc)
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_EXTENSIONS OFF)

set_source_files_properties(${GGML_SOURCES_CUDA} PROPERTIES LANGUAGE CXX)
foreach(SOURCE ${GGML_SOURCES_CUDA})
    #set(COMPILE_FLAGS "-I${SDK_DIR}/include -g3 -O3 -v --save-temps=.  -fPIC --offload-arch=dlgput64 -x cuda")
    #set(COMPILE_FLAGS "-I${SDK_DIR}/include -g3 -O0  --save-temps=. -fPIC --offload-arch=dlgput64 -x cuda")
    set(COMPILE_FLAGS "-I${SDK_DIR}/include -use_fast_math -mllvm -dlgpu-lower-xtpvn=true --save-temps=. -fPIC --offload-arch=dlgput64 -x cuda")
    set_property(SOURCE ${SOURCE} PROPERTY COMPILE_FLAGS ${COMPILE_FLAGS})
endforeach()


ggml_add_backend_library(ggml-dlcu
                    ${GGML_HEADERS_CUDA}
                    ${GGML_SOURCES_CUDA}
                    )

add_compile_definitions(GGML_USE_DLCU)

add_compile_definitions(GGML_CUDA_PEER_MAX_BATCH_SIZE=${GGML_CUDA_PEER_MAX_BATCH_SIZE})

if (GGML_CUDA_GRAPHS)
add_compile_definitions(GGML_CUDA_USE_GRAPHS)
endif()

if (GGML_CUDA_FORCE_MMQ)
add_compile_definitions(GGML_CUDA_FORCE_MMQ)
endif()

if (GGML_CUDA_FORCE_CUBLAS)
add_compile_definitions(GGML_CUDA_FORCE_CUBLAS)
endif()

if (GGML_CUDA_NO_VMM)
add_compile_definitions(GGML_CUDA_NO_VMM)
endif()

if (GGML_CUDA_F16 OR GGML_CUDA_DMMV_F16)
add_compile_definitions(GGML_CUDA_F16)
endif()

if (GGML_CUDA_NO_PEER_COPY)
add_compile_definitions(GGML_CUDA_NO_PEER_COPY)
endif()


set(CUDA_CXX_FLAGS "")

set(CUDA_FLAGS -use_fast_math)

if (GGML_FATAL_WARNINGS)
list(APPEND CUDA_FLAGS -Werror all-warnings)
endif()

if (GGML_ALL_WARNINGS AND NOT MSVC)
set(NVCC_CMD ${CMAKE_CUDA_COMPILER} .c)
if (NOT CMAKE_CUDA_HOST_COMPILER STREQUAL "")
    list(APPEND NVCC_CMD -ccbin ${CMAKE_CUDA_HOST_COMPILER})
endif()



set(CUDA_CCID "Clang")
set(CUDA_CCFULLVER "clang")


message("-- CUDA host compiler is ${CUDA_CCID} ${CUDA_CCVER}")

list(APPEND CUDA_CXX_FLAGS ${CXX_FLAGS} ${GF_CXX_FLAGS})  # This is passed to -Xcompiler later
endif()

if (NOT MSVC)
list(APPEND CUDA_CXX_FLAGS -Wno-pedantic)
endif()

list(JOIN   CUDA_CXX_FLAGS " " CUDA_CXX_FLAGS_JOINED)  # pass host compiler flags as a single argument

if (NOT CUDA_CXX_FLAGS_JOINED STREQUAL "")
list(APPEND CUDA_FLAGS -Xcompiler ${CUDA_CXX_FLAGS_JOINED})
endif()


target_compile_options(ggml-dlcu PRIVATE -I${SDK_DIR}/include  -fPIC --offload-arch=dlgput64 -x cuda)
target_link_libraries(ggml-dlcu PRIVATE curt cublas)

set_target_properties(ggml-dlcu PROPERTIES LINKER_LANGUAGE CXX)
