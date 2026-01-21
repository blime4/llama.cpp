include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# Android + DLCU multi-model compatibility check
# When using GGML_DLCU on Android API < 28, posix_spawn is not available
# so we use a stub implementation that disables multi-model features
set(LLAMA_ANDROID_NO_MULTIMODEL FALSE)
if (DEFINED ANDROID_PLATFORM)
    # ANDROID_PLATFORM can be like "android-24" or "android-25"
    string(REGEX REPLACE "android-" "" ANDROID_API_NUM "${ANDROID_PLATFORM}")
    if (ANDROID_API_NUM LESS 28)
        if (GGML_DLCU)
            set(LLAMA_ANDROID_NO_MULTIMODEL TRUE)
            message(STATUS "Android API ${ANDROID_API_NUM} with GGML_DLCU detected: multi-model server features disabled (requires API 28+)")
        endif()
    endif()
endif()

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-context.cpp
    server-context.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC common mtmd ${CMAKE_THREAD_LIBS_INIT})


# llama-server executable

set(TARGET llama-server)

if (NOT LLAMA_HTTPLIB)
    message(FATAL_ERROR "LLAMA_HTTPLIB is OFF, cannot build llama-server. Hint: to skip building server, set -DLLAMA_BUILD_SERVER=OFF")
endif()

set(TARGET_SRCS
    server.cpp
    server-http.cpp
    server-http.h
)

# Conditionally choose between full multi-model support and stub implementation
if (LLAMA_ANDROID_NO_MULTIMODEL)
    message(STATUS "Using stub implementation for server-models (Android API < 28 + GGML_DLCU)")
    list(APPEND TARGET_SRCS server-models-stub.cpp)
else()
    list(APPEND TARGET_SRCS server-models.cpp)
endif()

list(APPEND TARGET_SRCS server-models.h)
set(PUBLIC_ASSETS
    index.html.gz
    loading.html
)

foreach(asset ${PUBLIC_ASSETS})
    set(input "${CMAKE_CURRENT_SOURCE_DIR}/public/${asset}")
    set(output "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")
    list(APPEND TARGET_SRCS ${output})
    add_custom_command(
        DEPENDS "${input}"
        OUTPUT "${output}"
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${PROJECT_SOURCE_DIR}/scripts/xxd.cmake"
    )
    set_source_files_properties(${output} PROPERTIES GENERATED TRUE)
endforeach()

add_executable(${TARGET} ${TARGET_SRCS})
install(TARGETS ${TARGET} RUNTIME)

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PRIVATE server-context PUBLIC common cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

if (WIN32)
    TARGET_LINK_LIBRARIES(${TARGET} PRIVATE ws2_32)
endif()

target_compile_features(${TARGET} PRIVATE cxx_std_17)
