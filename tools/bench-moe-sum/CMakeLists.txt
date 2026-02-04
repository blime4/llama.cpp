add_executable(bench-moe-sum
    bench-moe-sum.cpp
)

target_link_libraries(bench-moe-sum PRIVATE ggml)
target_include_directories(bench-moe-sum PRIVATE ${CMAKE_SOURCE_DIR}/ggml/include)

install(TARGETS bench-moe-sum DESTINATION ${CMAKE_INSTALL_BINDIR})
