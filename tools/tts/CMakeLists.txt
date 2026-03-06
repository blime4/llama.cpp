set(TARGET llama-tts)
add_executable(${TARGET} tts.cpp)
target_link_libraries(${TARGET} PRIVATE llama common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET_ORPHEUS llama-orpheus-tts)
add_executable(${TARGET_ORPHEUS} orpheus-tts.cpp snac-ggml.cpp)
target_link_libraries(${TARGET_ORPHEUS} PRIVATE llama common ggml ggml-base ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_ORPHEUS} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} ${TARGET_ORPHEUS} RUNTIME)
endif()
