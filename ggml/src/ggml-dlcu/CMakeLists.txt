
file(GLOB   GGML_HEADERS_CUDA "../ggml-cuda/*.cuh")
list(APPEND GGML_HEADERS_CUDA "../../include/ggml-cuda.h")
# Add DL-specific headers (in this directory)
list(APPEND GGML_HEADERS_CUDA "dl-mulmat.cuh")
list(APPEND GGML_HEADERS_CUDA "dl-fattn.cuh")
list(APPEND GGML_HEADERS_CUDA "fp16/dl-fp16.cuh")

file(GLOB   GGML_SOURCES_CUDA "../ggml-cuda/*.cu")
# Add DL-specific sources (in this directory)
file(GLOB   GGML_SOURCES_DLCU "*.cu")
list(APPEND GGML_SOURCES_CUDA ${GGML_SOURCES_DLCU})
file(GLOB   GGML_DLCU_FP16_SRCS "fp16/*.cu")
list(APPEND GGML_SOURCES_CUDA ${GGML_DLCU_FP16_SRCS})

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

set(SDK_DIR CACHE PATH "Path to the SDK directory")
set(SDK_AARCH64_DIR CACHE PATH "Path to the aarch64 SDK directory for Android linking")

if(NOT DEFINED SDK_DIR)
    if(DEFINED ENV{SDK_DIR})
	set(SDK_DIR $ENV{SDK_DIR})
    else()
	message(FATAL_ERROR "SDK_DIR must be set via -DSDK_DIR= or env variable")
    endif()
endif()

# For Android, also check for aarch64 SDK directory
if(ANDROID)
    if(NOT DEFINED SDK_AARCH64_DIR)
        if(DEFINED ENV{SDK_AARCH64_DIR})
            set(SDK_AARCH64_DIR $ENV{SDK_AARCH64_DIR})
        else()
            # Default to SDK_DIR if not specified
            set(SDK_AARCH64_DIR ${SDK_DIR})
            message(STATUS "SDK_AARCH64_DIR not specified, using SDK_DIR: ${SDK_AARCH64_DIR}")
        endif()
    endif()

    # Normalize the aarch64 SDK_DIR path
    get_filename_component(SDK_AARCH64_DIR "${SDK_AARCH64_DIR}" ABSOLUTE)
    message(STATUS "Using aarch64 SDK for linking: ${SDK_AARCH64_DIR}")
endif()

# Normalize the SDK_DIR path to avoid double slashes
get_filename_component(SDK_DIR "${SDK_DIR}" ABSOLUTE)

# Compiler configuration based on platform
if(WIN32)
    set(CMAKE_CXX_COMPILER "clang-cl.exe")
elseif(ANDROID)
    # For Android cross-compilation:
    # - C files will use NDK clang++ (set via CMAKE_C_COMPILER in run_llama.sh)
    # - CXX files (.cu files) will use SDK clang++ (set here)
    message(STATUS "Android cross-compilation detected")
    message(STATUS "Using NDK compilers for C code, SDK clang++ for CXX/CUDA code")
    message(STATUS "SDK_DIR: ${SDK_DIR}")
    message(STATUS "ANDROID_NDK: ${ANDROID_NDK}")

    # Override CXX compiler to use SDK clang++ for CUDA files
    # This will affect all CXX files in this target, including .cu files
    if(EXISTS "${SDK_DIR}/bin/clang++")
        set(CMAKE_CXX_COMPILER "${SDK_DIR}/bin/clang++")
        message(STATUS "Set CMAKE_CXX_COMPILER to: ${CMAKE_CXX_COMPILER}")
    else()
        message(WARNING "SDK clang++ not found at ${SDK_DIR}/bin/clang++")
    endif()

    # Set LIBRARY_PATH for SDK compiler when compiling CUDA code
    set(ENV{LIBRARY_PATH} "${SDK_DIR}/lib:$ENV{LIBRARY_PATH}")
    # Also add library path via linker flags
    link_directories(${SDK_DIR}/lib)
else()
    # For non-Android platforms, use dlcc as the main compiler
    set(CMAKE_CXX_COMPILER ${SDK_DIR}/bin/dlcc)
    set(CMAKE_C_COMPILER ${SDK_DIR}/bin/dlcc)
    # Set LIBRARY_PATH for dlcc compiler
    set(ENV{LIBRARY_PATH} "${SDK_DIR}/lib:$ENV{LIBRARY_PATH}")
    # Also add library path via linker flags
    link_directories(${SDK_DIR}/lib)
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_EXTENSIONS OFF)

# Detect GCC version and set C++ include paths for CUDA compilation
find_program(GCC_EXECUTABLE gcc)
if(GCC_EXECUTABLE)
    execute_process(
        COMMAND ${GCC_EXECUTABLE} -dumpversion
        OUTPUT_VARIABLE GCC_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REGEX REPLACE "([0-9]+).*" "\\1" GCC_MAJOR_VERSION ${GCC_VERSION})
    message("-- Detected GCC version: ${GCC_VERSION} (major: ${GCC_MAJOR_VERSION})")

    # Detect target triple for C++ includes
    execute_process(
        COMMAND ${GCC_EXECUTABLE} -dumpmachine
        OUTPUT_VARIABLE GCC_TARGET_TRIPLE
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message("-- GCC target triple: ${GCC_TARGET_TRIPLE}")

    # For target_compile_options (list format)
    set(CXX_STDLIB_PATHS_LIST
        "-cxx-isystem" "/usr/include/c++/${GCC_MAJOR_VERSION}"
        "-cxx-isystem" "/usr/include/c++/${GCC_MAJOR_VERSION}/${GCC_TARGET_TRIPLE}"
    )

    # For COMPILE_FLAGS (string format with spaces)
    set(CXX_STDLIB_PATHS_STRING "-cxx-isystem /usr/include/c++/${GCC_MAJOR_VERSION} -cxx-isystem /usr/include/c++/${GCC_MAJOR_VERSION}/${GCC_TARGET_TRIPLE}")
else()
    message(WARNING "GCC not found, C++ stdlib paths may not be set correctly")
    set(CXX_STDLIB_PATHS_LIST "")
    set(CXX_STDLIB_PATHS_STRING "")
endif()

set_source_files_properties(${GGML_SOURCES_CUDA} PROPERTIES LANGUAGE CXX)

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


# Set base compile options for DLCU
set(DLCU_COMPILE_OPTIONS
    -I${SDK_DIR}/include
    -L${SDK_DIR}/lib
    -use_fast_math
    -mllvm
    -dlgpu-lower-xtpvn=true
    --save-temps=.
    -fPIC
    --cuda-gpu-arch=dlgput64
)

# Add common CUDA options
list(APPEND DLCU_COMPILE_OPTIONS
    -x
    cuda
)

# Add Android-specific compile options
if(ANDROID)
    # Allow macro redefinition (from NDK and SDK)
    list(APPEND DLCU_COMPILE_OPTIONS -Wno-macro-redefined)

    # Add cross-compilation options for Android when using NDK clang++ fallback
    if(NOT EXISTS "${SDK_DIR}/bin/clang++")
        # Determine Android API level from ANDROID_PLATFORM
        message(STATUS "NOT EXISTS ${SDK_DIR}/bin/clang++")
        message(STATUS "ANDROID_PLATFORM: ${ANDROID_PLATFORM}")
        message(STATUS "ANDROID_API_NUM: ${ANDROID_API_NUM}")
        string(REGEX REPLACE "android-" "" ANDROID_API_NUM "${ANDROID_PLATFORM}")

        # Add cross-compilation options for Android (fallback when SDK clang++ not available)
        list(APPEND DLCU_COMPILE_OPTIONS --target=aarch64-none-linux-android${ANDROID_API_NUM})
        list(APPEND DLCU_COMPILE_OPTIONS --sysroot=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot)
        list(APPEND DLCU_COMPILE_OPTIONS --gcc-toolchain=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64)
        list(APPEND DLCU_COMPILE_OPTIONS -resource-dir=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/14.0.7)
        list(APPEND DLCU_COMPILE_OPTIONS -Wno-unused-command-line-argument)
    else()
        message(STATUS "EXISTS ${SDK_DIR}/bin/clang++")
    endif()

    message(STATUS "Added Android cross-compilation options to DLCU_COMPILE_OPTIONS")
endif()

target_compile_options(ggml-dlcu PRIVATE ${DLCU_COMPILE_OPTIONS})

# Android cross-compilation: Dual SDK configuration (x86 for compile, aarch64 for link)
if(ANDROID)
    message(STATUS "Android cross-compilation: Dual SDK configuration")
    message(STATUS "Compile SDK (x86): ${SDK_DIR}")
    message(STATUS "Link SDK (aarch64): ${SDK_AARCH64_DIR}")

    # Check if aarch64 SDK libraries exist for linking
    if(EXISTS "${SDK_AARCH64_DIR}/lib/libcurt.so")
        message(STATUS "Found aarch64 SDK libraries for linking, enabling full DLCU support")

        # Link with aarch64 SDK libraries
        target_link_directories(ggml-dlcu PRIVATE ${SDK_AARCH64_DIR}/lib)
        target_link_libraries(ggml-dlcu PRIVATE curt cublas cudnn)

        # Add libLLVM-15.so from aarch64 SDK
        if(EXISTS "${SDK_AARCH64_DIR}/lib/libLLVM-15.so")
            target_link_libraries(ggml-dlcu PRIVATE ${SDK_AARCH64_DIR}/lib/libLLVM-15.so)
            message(STATUS "Linked with aarch64 libLLVM-15.so: ${SDK_AARCH64_DIR}/lib/libLLVM-15.so")
        else()
            message(FATAL_ERROR "libLLVM-15.so not found in aarch64 SDK: ${SDK_AARCH64_DIR}/lib/")
        endif()
    else()
        message(FATAL_ERROR "aarch64 SDK libraries do not exist for linking: ${SDK_AARCH64_DIR}/lib/libcurt.so not found")
    endif()


    # Set Android-specific linker flags with dual SDK support
    set_target_properties(ggml-dlcu PROPERTIES
        LINK_FLAGS "-L${SDK_AARCH64_DIR}/lib -L${SDK_DIR}/lib -Wl,--allow-shlib-undefined -Wl,--unresolved-symbols=ignore-all -Wl,--no-undefined-version -resource-dir=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/14.0.7"
    )

    # Remove problematic compiler flags for Android cross-compilation
    target_compile_options(ggml-dlcu PRIVATE
        -Wno-unused-command-line-argument
    )

    message(STATUS "Android DLCU dual SDK library configuration completed")
else()
    # Standard configuration for non-Android platforms
    target_link_libraries(ggml-dlcu PRIVATE curt cublas cudnn)
endif()

set_target_properties(ggml-dlcu PROPERTIES LINKER_LANGUAGE CXX)
