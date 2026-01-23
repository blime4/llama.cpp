#!/bin/bash
#
# Device Status Checker for llama.cpp CI
# This script checks and handles device card status before running tests
#

set -e

# Create logs directory
logs_dir="/LocalRun/$(whoami)/logs/llama_cpp_test"
mkdir -p "$logs_dir"

# Log files - use current timestamp
test_log="${logs_dir}/device_check_$(date +%Y%m%d_%H%M%S).log"
summary_log="${logs_dir}/device_check_summary_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] Device status check started at: $(date)" | tee -a "$summary_log"
echo "[INFO] Platform: ${DOCKER_PLATFORM}" | tee -a "$summary_log"

# Function to check and handle device card status
check_device_status() {
    echo "[INFO] Checking device card status with lspci..." | tee -a "$summary_log"

    if ! command -v lspci >/dev/null 2>&1; then
        echo "[WARN] lspci command not found, skipping device card status check" | tee -a "$summary_log"
        return 0
    fi

    local device_status
    device_status=$(lspci -d 1e27: -v 2>/dev/null || echo "")

    if [ -z "$device_status" ]; then
        echo "[INFO] No devices with vendor ID 1e27 found, skipping device status check" | tee -a "$summary_log"
        return 0
    fi

    echo "[INFO] Found devices with vendor ID 1e27, checking status..." | tee -a "$summary_log"
    echo "[DEBUG] Device info:" >> "$test_log"
    echo "$device_status" | head -10 >> "$test_log"  # Show first 10 lines for reference

    if echo "$device_status" | grep -q "Unknown header"; then
        echo "[ERROR] Device card has dropped (Unknown header detected in lspci output)" | tee -a "$summary_log"
        echo "[ERROR] Device status output:" >> "$test_log"
        echo "$device_status" >> "$test_log"
        echo "[ERROR] ==========================================" >> "$test_log"

        # Platform-specific handling
        if [ "${DOCKER_PLATFORM}" = "loongarch64" ]; then
            echo "[ERROR] LoongArch64 platform: dlsmi -r is temporarily unavailable" | tee -a "$summary_log"
            echo "[ERROR] Please restart the system to recover the device" | tee -a "$summary_log"
            echo "[ERROR] Device check failed due to device card failure" | tee -a "$summary_log"
            exit 1
        else
            echo "[INFO] Attempting to reset device using dlsmi -r..." | tee -a "$summary_log"
            if command -v dlsmi >/dev/null 2>&1; then
                echo "[INFO] Executing device reset command..." | tee -a "$summary_log"
                if dlsmi -r >> "$test_log" 2>&1; then
                    echo "[INFO] Device reset command executed successfully" | tee -a "$summary_log"
                    echo "[INFO] Waiting for device to stabilize..." | tee -a "$summary_log"
                    sleep 5

                    # Verify device status after reset
                    echo "[INFO] Verifying device status after reset..." | tee -a "$summary_log"
                    local post_reset_status
                    post_reset_status=$(lspci -d 1e27: -v 2>/dev/null || echo "")
                    if echo "$post_reset_status" | grep -q "Unknown header"; then
                        echo "[ERROR] Device still shows Unknown header after reset" | tee -a "$summary_log"
                        echo "[ERROR] Please manually restart the system" | tee -a "$summary_log"
                        exit 1
                    else
                        echo "[INFO] Device status verification passed after reset" | tee -a "$summary_log"
                        echo "[INFO] Device check completed successfully" | tee -a "$summary_log"
                    fi
                else
                    echo "[ERROR] Device reset command failed" | tee -a "$summary_log"
                    echo "[ERROR] Please manually execute 'dlsmi -r' to reset the device" | tee -a "$summary_log"
                    echo "[ERROR] Device check failed due to device reset failure" | tee -a "$summary_log"
                    exit 1
                fi
            else
                echo "[ERROR] dlsmi command not available, cannot reset device" | tee -a "$summary_log"
                echo "[ERROR] Please manually execute 'dlsmi -r' to reset the device" | tee -a "$summary_log"
                echo "[ERROR] Device check failed - dlsmi not available" | tee -a "$summary_log"
                exit 1
            fi
        fi
    else
        echo "[INFO] Device card status check passed - no issues detected" | tee -a "$summary_log"
    fi
}

# Execute system diagnostic commands before testing
echo "[INFO] Running system diagnostics before testing..." | tee -a "$summary_log"

# Check device card status first (critical check)
check_device_status

# Check for denglin driver issues in dmesg
if command -v dmesg >/dev/null 2>&1; then
    echo "[INFO] Running dmesg..." | tee -a "$summary_log"
    sudo dmesg 2>&1 | tail -50 >> "$test_log" || echo "[WARN] dmesg command failed or returned non-zero exit code" | tee -a "$summary_log"

    # Check for denglin driver errors
    if sudo dmesg 2>/dev/null | tail -n 50 | grep -q "denglin.*err="; then
        echo "[WARN] Detected denglin driver errors in dmesg, may affect CUDA tests" | tee -a "$summary_log"
        echo "[INFO] Check detailed log for denglin driver error details: $test_log" | tee -a "$summary_log"
    fi
else
    echo "[INFO] dmesg command not found, skipping" | tee -a "$summary_log"
fi

# Check and run dlsmi with GPU information
if command -v dlsmi >/dev/null 2>&1; then
    echo "[INFO] Running dlsmi..." | tee -a "$summary_log"
    dlsmi >> "$test_log" 2>&1 || echo "[WARN] dlsmi command failed or returned non-zero exit code" | tee -a "$summary_log"

    # Get GPU status and count
    echo "[INFO] GPU status:" | tee -a "$summary_log"
    if dlsmi --list-gpus >> "$test_log" 2>&1; then
        gpu_count=$(dlsmi --list-gpus 2>/dev/null | wc -l)
        echo "[INFO] Detected $gpu_count GPU(s)" | tee -a "$summary_log"

        # Recommend GPU limiting for large GPU counts
        if [ "$gpu_count" -gt 1 ]; then
            echo "[INFO] Note: $gpu_count GPUs detected. Consider limiting to first 1 devices for testing" | tee -a "$summary_log"
            echo "[INFO] Recommendation: export CUDA_VISIBLE_DEVICES=0" | tee -a "$summary_log"
        fi
    else
        echo "[WARN] Failed to query GPU status with dlsmi --list-gpus" | tee -a "$summary_log"
    fi
else
    echo "[INFO] dlsmi command not found, skipping GPU status check" | tee -a "$summary_log"
fi

echo "[INFO] System diagnostics completed successfully" | tee -a "$summary_log"
echo "[INFO] Device status check completed at: $(date)" | tee -a "$summary_log"

# Log file information
test_log_size=$(du -h "$test_log" | cut -f1)
summary_log_size=$(du -h "$summary_log" | cut -f1)
echo "[INFO] Device check log files created:" | tee -a "$summary_log"
echo "[INFO]   Detailed log: $test_log ($test_log_size)" | tee -a "$summary_log"
echo "[INFO]   Summary log: $summary_log ($summary_log_size)" | tee -a "$summary_log"

echo "[SUCCESS] Device status check passed - ready for testing" | tee -a "$summary_log"
