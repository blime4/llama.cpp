llama_add_compile_flags()

#
# libraries
#

# llama

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-cparams.cpp
            llama-grammar.cpp
            llama-graph.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-io.cpp
            llama-kv-cache.cpp
            llama-kv-cache-iswa.cpp
            llama-memory.cpp
            llama-memory-hybrid.cpp
            llama-memory-recurrent.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-model-saver.cpp
            llama-model.cpp
            llama-quant.cpp
            llama-sampling.cpp
            llama-vocab.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            )

target_include_directories(llama PRIVATE .)
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

if (LLAMA_KINETO)
target_link_libraries(llama PUBLIC ggml kineto)
else()
target_link_libraries(llama PUBLIC ggml)
endif()

# Link with ggml-dlcu for DLFA functions (needed for prepare_varlen_buffers)
# In GGML_BACKEND_DL mode, ggml-dlcu is a MODULE, but we still need to link it for symbols
# Note: ggml_backend_cuda_get_context and ggml_backend_is_cuda are also in ggml-dlcu
if (GGML_USE_DLFA)
    # For MODULE targets, link the library file directly using relative path from build dir
    # The library is in the same output directory as binaries (bin/)
    target_link_libraries(llama PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/../bin/libggml-dlcu.so")
    message(STATUS "Linking llama with ggml-dlcu: ${CMAKE_CURRENT_BINARY_DIR}/../bin/libggml-dlcu.so")
endif()

# DL
target_compile_definitions(llama PRIVATE GGML_USE_DLFA)
target_compile_definitions(llama PRIVATE GGML_USE_DLCU)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
