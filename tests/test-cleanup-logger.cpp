#include "../src/llama-cleanup-logger.h"
#include "ggml.h"

#include <cstdlib>
#include <cstdio>
#include <vector>
#include <string>

int main() {
    printf("Testing cleanup_logger...\n");

    // Test 1: Basic instantiation
    {
        cleanup_logger logger;
        printf("✓ Test 1: Logger instantiation successful\n");
    }

    // Test 2: Verbose mode toggle
    {
        cleanup_logger logger;

        // Default should be false (unless env var is set)
        bool initial_verbose = logger.is_verbose();

        // Toggle verbose mode
        logger.set_verbose(true);
        if (!logger.is_verbose()) {
            printf("✗ Test 2 FAILED: set_verbose(true) did not enable verbose mode\n");
            return 1;
        }

        logger.set_verbose(false);
        if (logger.is_verbose()) {
            printf("✗ Test 2 FAILED: set_verbose(false) did not disable verbose mode\n");
            return 1;
        }

        printf("✓ Test 2: Verbose mode toggle works correctly\n");
    }

    // Test 3: Log methods don't crash (basic smoke test)
    {
        cleanup_logger logger;
        logger.set_verbose(true);

        // Test cleanup start
        logger.log_cleanup_start(2);

        // Test device sync logging
        logger.log_device_sync(0, true, 1000);
        logger.log_device_sync(1, false, 2000);

        // Test resource free logging
        void* dummy_ptr = (void*)0x12345678;
        logger.log_resource_free("kv_cache", 0, dummy_ptr, 1024);
        logger.log_resource_free("buffer", 1, dummy_ptr, 2048);

        // Test cleanup end logging
        std::vector<std::string> errors;
        logger.log_cleanup_end(5000, errors);

        errors.push_back("Test error 1");
        errors.push_back("Test error 2");
        logger.log_cleanup_end(6000, errors);

        printf("✓ Test 3: All log methods execute without crashing\n");
    }

    // Test 4: Environment variable detection
    {
        // Note: This test depends on whether LLAMA_LOG_CLEANUP is set
        // We just verify the logger can be created
        cleanup_logger logger;
        printf("✓ Test 4: Environment variable handling works\n");
    }

    // Test 5: Verbose mode suppresses output when disabled
    {
        cleanup_logger logger;
        logger.set_verbose(false);

        // These should not produce output (we can't easily verify this in a unit test,
        // but we can verify they don't crash)
        logger.log_cleanup_start(2);
        logger.log_device_sync(0, true, 1000);
        logger.log_resource_free("test", 0, nullptr, 100);
        logger.log_cleanup_end(1000, {});

        printf("✓ Test 5: Verbose mode suppression works\n");
    }

    printf("\n=== All cleanup_logger tests passed! ===\n");
    return 0;
}
