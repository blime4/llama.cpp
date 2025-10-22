
file(GLOB   GGML_HEADERS_CUDA "../ggml-cuda/*.cuh")
list(APPEND GGML_HEADERS_CUDA "../../include/ggml-cuda.h")
# Add DL-specific headers (in this directory)
list(APPEND GGML_HEADERS_CUDA "ggml-dl.cuh")
list(APPEND GGML_HEADERS_CUDA "dl-fattn.cuh")
list(APPEND GGML_HEADERS_CUDA "dl-fattn-golden.cuh")

file(GLOB   GGML_SOURCES_CUDA "../ggml-cuda/*.cu")
# Add DL-specific sources (in this directory)
list(APPEND GGML_SOURCES_CUDA "ggml-dl.cu")
list(APPEND GGML_SOURCES_CUDA "dl-fattn.cu")
list(APPEND GGML_SOURCES_CUDA "dl-fattn-golden.cu")

# TODO:  optimize this, cause DL do not fully support all the template instances
#        it will just waste time to compile all the template instances.
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
    # Set LIBRARY_PATH for dlcc compiler
    set(ENV{LIBRARY_PATH} "${SDK_DIR}/lib:$ENV{LIBRARY_PATH}")
    # Also add library path via linker flags
    link_directories(${SDK_DIR}/lib)
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_EXTENSIONS OFF)

set_source_files_properties(${GGML_SOURCES_CUDA} PROPERTIES LANGUAGE CXX)
foreach(SOURCE ${GGML_SOURCES_CUDA})
    #set(COMPILE_FLAGS "-I${SDK_DIR}/include -g3 -O3 -v --save-temps=.  -fPIC --offload-arch=dlgput64 -x cuda")
    #set(COMPILE_FLAGS "-I${SDK_DIR}/include -g3 -O0  --save-temps=. -fPIC --offload-arch=dlgput64 -x cuda")
    set(COMPILE_FLAGS "-I${SDK_DIR}/include -L${SDK_DIR}/lib -use_fast_math -mllvm -dlgpu-lower-xtpvn=true --save-temps=. -fPIC --offload-arch=dlgput64 -x cuda")
    # Add resource-dir only for Android platform
    if(ANDROID)
        set(COMPILE_FLAGS "${COMPILE_FLAGS} -resource-dir=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/14.0.7")
    endif()
    set_property(SOURCE ${SOURCE} PROPERTY COMPILE_FLAGS ${COMPILE_FLAGS})
endforeach()


ggml_add_backend_library(ggml-dlcu
                    ${GGML_HEADERS_CUDA}
                    ${GGML_SOURCES_CUDA}
                    )

add_compile_definitions(GGML_USE_DLCU)
add_compile_definitions(GGML_USE_DLTU)
add_compile_definitions(GGML_USE_DLFA)

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

if (NOT GGML_CUDA_FA)
    add_compile_definitions(GGML_CUDA_NO_FA)
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


# Set base compile options
set(DLCU_COMPILE_OPTIONS -I${SDK_DIR}/include -L${SDK_DIR}/lib -fPIC --offload-arch=dlgput64 -x cuda)
# Add resource-dir only for Android platform
if(ANDROID)
    list(APPEND DLCU_COMPILE_OPTIONS -resource-dir=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/14.0.7)
endif()
target_compile_options(ggml-dlcu PRIVATE ${DLCU_COMPILE_OPTIONS})

# Android cross-compilation: SDK library architecture handling
if(ANDROID)
    message(STATUS "Android cross-compilation: Checking SDK library architecture compatibility")

    # Check if aarch64 SDK libraries exist
    if(EXISTS "${SDK_DIR}/lib/libcurt.so")
        # Verify SDK library architecture
        execute_process(
            COMMAND file "${SDK_DIR}/lib/libcurt.so"
            OUTPUT_VARIABLE SDK_LIB_ARCH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        if(SDK_LIB_ARCH MATCHES "aarch64")
            message(STATUS "Found aarch64 SDK libraries, enabling full DLCU support")
            target_link_libraries(ggml-dlcu PRIVATE curt cublas cudnn)
        else()
            message(WARNING "SDK libraries are x86_64 (not ARM64), creating stub DLCU library")
            message(STATUS "Note: Using x86_64 SDK with amd64 Docker image for cross-compilation")
            message(STATUS "DLCU will run in CPU fallback mode on Android")
            message(STATUS "Detected SDK library architecture: ${SDK_LIB_ARCH}")
            target_compile_definitions(ggml-dlcu PRIVATE
                GGML_DLCU_ANDROID_STUB=1
                GGML_DLCU_NO_CUDA=1
            )
            target_link_libraries(ggml-dlcu PRIVATE log m)
        endif()
    else()
        message(WARNING "SDK CUDA libraries not found, creating stub DLCU library")
        target_compile_definitions(ggml-dlcu PRIVATE
            GGML_DLCU_ANDROID_STUB=1
            GGML_DLCU_NO_CUDA=1
        )
        target_link_libraries(ggml-dlcu PRIVATE log m)
    endif()

    # TODO: Remove this after Android SDK libraries are ready.
    # Set Android-specific linker flags to handle missing runtime libraries
    set_target_properties(ggml-dlcu PROPERTIES
        LINK_FLAGS "-L${SDK_DIR}/lib -Wl,--allow-shlib-undefined -Wl,--unresolved-symbols=ignore-all -Wl,--no-undefined-version -resource-dir=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/14.0.7"
    )

    # TODO: Remove this after Android SDK libraries are ready.
    # Remove problematic compiler flags for Android cross-compilation
    target_compile_options(ggml-dlcu PRIVATE
        -Wno-unused-command-line-argument
    )

    message(STATUS "Android DLCU library configuration completed")
else()
    # Standard configuration for non-Android platforms
    target_link_libraries(ggml-dlcu PRIVATE curt cublas cudnn)
endif()


set_target_properties(ggml-dlcu PROPERTIES LINKER_LANGUAGE CXX)
