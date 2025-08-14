#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <random>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "dlblas_ext.h"

struct ggml_gptq_data {
    void * qweight;
    void * qzeros;
    void * scales;
    int bits;
    int group_size;
    cudaDataType_t scales_type;
    cudaDataType_t qzeros_type;
    int num_groups;
    int M;
    int K;
    int qweight_size;
    int qzeros_size;
    int scales_size;
};

__global__ void ggml_cuda_gptq_quantize_8_bit_fp32(
    const half* __restrict__ w,    // [M, K], half type
    uint8_t* __restrict__ qweight,
    float* __restrict__ qzeros,
    float* __restrict__ scales,
    int K, int M, int group_size
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int group = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || group * group_size >= K) return;

    float max_val = -FLT_MAX, min_val = FLT_MAX;
    for (int k = 0; k < group_size && group * group_size + k < K; ++k) {
        float v = __half2float(w[row * K + (group * group_size + k)]);
        if (v > max_val) max_val = v;
        if (v < min_val) min_val = v;
    }

    float scale, zero_point;
    const float qmin = 0.0f;
    const float qmax = 255.0f;

    if (max_val == min_val) {
        if (min_val == 0.0f) {
            scale = 1.0f;
            zero_point = 0.0f;
        } else {
            scale = std::abs(min_val) / 255.0f;
            if (min_val > 0.0f) {
                zero_point = qmin;
            } else {
                zero_point = qmax;
            }
        }
    } else {
        scale = (max_val - min_val) / (qmax - qmin);
        zero_point = qmin - min_val / scale;

        if (zero_point < qmin) zero_point = qmin;
        if (zero_point > qmax) zero_point = qmax;

        if (scale < 1e-8f) scale = 1e-8f;
    }


    const int num_groups_row = (K + group_size - 1) / group_size;
    int scale_idx = row * num_groups_row + group;
    scales[scale_idx] = scale;
    qzeros[scale_idx] = zero_point;


    for (int k = 0; k < group_size && group * group_size + k < K; ++k) {
        int col = group * group_size + k;
        float v = __half2float(w[row * K + col]);

        int q;
        if (max_val == min_val) {
            if (min_val == 0.0f) {
                q = 0;
            } else {
                q = (int)roundf(v / scale + zero_point);
            }
        } else {
            q = (int)roundf(v / scale + zero_point);
        }
        if (q < 0) q = 0;
        if (q > 255) q = 255;

        qweight[row * K + col] = (uint8_t)q;
    }
}

__global__ void ggml_cuda_gptq_quantize_8_bit_fp16(
    const half* __restrict__ w,    // [M, K], half type
    uint8_t* __restrict__ qweight,
    half* __restrict__ qzeros,
    half* __restrict__ scales,
    int K, int M, int group_size
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int group = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || group * group_size >= K) return;

    float max_val = -FLT_MAX, min_val = FLT_MAX;
    for (int k = 0; k < group_size && group * group_size + k < K; ++k) {
        float v = __half2float(w[row * K + (group * group_size + k)]);
        if (v > max_val) max_val = v;
        if (v < min_val) min_val = v;
    }

    float scale, zero_point;
    const float qmin = 0.0f;
    const float qmax = 255.0f;

    if (max_val == min_val) {
        if (min_val == 0.0f) {
            scale = 1.0f;
            zero_point = 0.0f;
        } else {
            scale = std::abs(min_val) / 255.0f;
            if (min_val > 0.0f) {
                zero_point = qmin;
            } else {
                zero_point = qmax;
            }
        }
    } else {
        scale = (max_val - min_val) / (qmax - qmin);
        zero_point = qmin - min_val / scale;

        if (zero_point < qmin) zero_point = qmin;
        if (zero_point > qmax) zero_point = qmax;

        if (scale < 1e-8f) scale = 1e-8f;
    }

    const int num_groups_row = (K + group_size - 1) / group_size;
    int scale_idx = row * num_groups_row + group;
    scales[scale_idx] = __float2half(scale);
    qzeros[scale_idx] = __float2half(zero_point);

    for (int k = 0; k < group_size && group * group_size + k < K; ++k) {
        int col = group * group_size + k;
        float v = __half2float(w[row * K + col]);

        // q = round(v/scale + zero_point)
        int q;
        if (max_val == min_val) {
            if (min_val == 0.0f) {
                q = 0;
            } else {
                q = (int)roundf(v / scale + zero_point);
            }
        } else {
            q = (int)roundf(v / scale + zero_point);
        }

        if (q < 0) q = 0;
        if (q > 255) q = 255;

        qweight[row * K + col] = (uint8_t)q;
    }
}

struct TestCase {
    std::string name;
    float weight_value;
    float input_value;
    float expected_result;
};

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t stat = call; \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

// print the max error context for debugging (col-major)
static void print_max_error_context(
    const std::vector<float> & reference_values,
    const std::vector<float> & test_values,
    int M,
    int N,
    int window_radius,
    float trigger_threshold,
    const char * title
) {
    if (reference_values.size() != static_cast<size_t>(M * N) ||
        test_values.size() != static_cast<size_t>(M * N)) {
        std::cout << "[print_max_error_context] size mismatch, skip" << std::endl;
        return;
    }

    size_t max_index = 0;
    float max_abs_error = 0.0f;
    for (size_t i = 0; i < reference_values.size(); ++i) {
        float abs_err = std::abs(reference_values[i] - test_values[i]);
        if (abs_err > max_abs_error) {
            max_abs_error = abs_err;
            max_index = i;
        }
    }

    if (max_abs_error < trigger_threshold) {
        std::cout << "[print_max_error_context] max abs error " << max_abs_error
                  << " < threshold " << trigger_threshold << ", skip" << std::endl;
        return;
    }

    int row = static_cast<int>(max_index % M);
    int col = static_cast<int>(max_index / M);
    float ref_val = reference_values[max_index];
    float tst_val = test_values[max_index];
    float rel_err = max_abs_error / std::max(std::abs(ref_val), 1e-6f) * 100.0f;

    std::cout << "\n==== Max Abs Error Context: " << title << " ====\n";
    std::cout << "  M=" << M << ", N=" << N << ", window_radius=" << window_radius << std::endl;
    std::cout << "  Max abs error at linear idx=" << max_index
              << " (row=" << row << ", col=" << col << ")\n";
    std::cout << "  ref(cuBLAS)=" << ref_val << ", test(dlBLAS)=" << tst_val
              << ", abs_err=" << max_abs_error << ", rel_err=" << rel_err << "%\n";

    int r0 = std::max(0, row - window_radius);
    int r1 = std::min(M - 1, row + window_radius);
    int c0 = std::max(0, col - window_radius);
    int c1 = std::min(N - 1, col + window_radius);

    std::cout << "\n  Neighborhood (row in [" << r0 << "," << r1 << "] , col in [" << c0 << "," << c1 << "])\n";
    for (int rr = r0; rr <= r1; ++rr) {
        std::cout << "  row " << rr << ": ";
        for (int cc = c0; cc <= c1; ++cc) {
            size_t idx = static_cast<size_t>(cc) * static_cast<size_t>(M) + static_cast<size_t>(rr);
            float rv = reference_values[idx];
            float tv = test_values[idx];
            float dv = tv - rv;
            std::cout << "[c=" << cc << ", ref=" << rv << ", dlb=" << tv << ", d=" << dv << "] ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

float run_random_weight_matrix_test(int M, int K, int N, int GROUP_SIZE, int NUM_GROUPS, std::mt19937& gen) {
    std::cout << "Weight matrix: random values, input matrix: random values" << std::endl;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    std::uniform_real_distribution<float> weight_dis(-2.0f, 2.0f);
    std::uniform_real_distribution<float> input_dis(-1.0f, 1.0f);

    std::vector<float> weight_cpu_fp32(M * K);
    std::vector<float> input_cpu_fp32(N * K);
    std::vector<half> weight_cpu_fp16(M * K);
    std::vector<half> input_cpu_fp16(N * K);

    for (int i = 0; i < M * K; ++i) {
        weight_cpu_fp32[i] = weight_dis(gen);
        weight_cpu_fp16[i] = __float2half(weight_cpu_fp32[i]);
    }
    for (int i = 0; i < N * K; ++i) {
        input_cpu_fp32[i] = input_dis(gen);
        input_cpu_fp16[i] = __float2half(input_cpu_fp32[i]);
    }

    std::vector<float> cublas_reference_result_fp32(M * N, 0.0f);
    std::vector<half> cublas_reference_result_fp16(M * N, __float2half(0.0f));

    float* d_input_fp32;
    half* d_input_fp16;
    float* d_output_cublas_fp32;
    half* d_output_cublas_fp16;

    CUDA_CHECK(cudaMalloc(&d_input_fp32, N * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_cublas_fp32, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input_fp16, N * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output_cublas_fp16, M * N * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_input_fp32, input_cpu_fp32.data(), N * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input_fp16, input_cpu_fp16.data(), N * K * sizeof(half), cudaMemcpyHostToDevice));

    float* d_weight_fp32;
    half* d_weight_fp16;
    float* d_output_fp32;
    half* d_output_fp16;
    uint8_t* d_qweight_fp32;
    uint8_t* d_qweight_fp16;
    float* d_qzeros_fp32;
    half* d_qzeros_fp16;
    float* d_scales_fp32;
    half* d_scales_fp16;

    CUDA_CHECK(cudaMalloc(&d_weight_fp32, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weight_fp16, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output_fp32, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_fp16, M * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_qweight_fp32, M * K * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_qweight_fp16, M * K * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_qzeros_fp32, M * (K + GROUP_SIZE - 1) / GROUP_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_qzeros_fp16, M * (K + GROUP_SIZE - 1) / GROUP_SIZE * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_scales_fp32, M * (K + GROUP_SIZE - 1) / GROUP_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scales_fp16, M * (K + GROUP_SIZE - 1) / GROUP_SIZE * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_weight_fp32, weight_cpu_fp32.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight_fp16, weight_cpu_fp16.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input_fp32, input_cpu_fp32.data(), N * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input_fp16, input_cpu_fp16.data(), N * K * sizeof(half), cudaMemcpyHostToDevice));

    //  --------------------------------------- FP32 FP32 ------------------------------------------
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSetStream(handle, 0));
    CUBLAS_CHECK(
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                M, N, K,
                &alpha, d_weight_fp32, CUDA_R_32F, K,
                        d_input_fp32,  CUDA_R_32F, K,
                &beta,  d_output_cublas_fp32,      CUDA_R_32F, M,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    CUDA_CHECK(cudaMemcpy(cublas_reference_result_fp32.data(), d_output_cublas_fp32, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_output_cublas_fp32);
    //  --------------------------------------- FP32 FP32 ------------------------------------------

    //  --------------------------------------- FP16 FP16 ------------------------------------------
    CUBLAS_CHECK(cublasSetStream(handle, 0));
    CUBLAS_CHECK(
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                M, N, K,
                &alpha, d_weight_fp16, CUDA_R_16F, K,
                        d_input_fp16,  CUDA_R_16F, K,
                &beta,  d_output_cublas_fp16,      CUDA_R_16F, M,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    CUDA_CHECK(cudaMemcpy(cublas_reference_result_fp16.data(), d_output_cublas_fp16, M * N * sizeof(half), cudaMemcpyDeviceToHost));

    cudaFree(d_output_cublas_fp16);
    //  --------------------------------------- FP16 FP16 ------------------------------------------


    //  --------------------------------------- FP32 FP32 ------------------------------------------
    dim3 blockDim(32, 8);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (NUM_GROUPS + blockDim.y - 1) / blockDim.y);

    ggml_cuda_gptq_quantize_8_bit_fp32<<<gridDim, blockDim>>>(
        d_weight_fp16, d_qweight_fp32, d_qzeros_fp32, d_scales_fp32, K, M, GROUP_SIZE
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    ggml_gptq_data gptq_data;
    gptq_data.qweight = d_qweight_fp32;
    gptq_data.qzeros = d_qzeros_fp32;
    gptq_data.scales = d_scales_fp32;
    gptq_data.bits = 8;
    gptq_data.group_size = GROUP_SIZE;
    gptq_data.scales_type = CUDA_R_32F;
    gptq_data.qzeros_type = CUDA_R_32F;
    gptq_data.num_groups = NUM_GROUPS;
    gptq_data.M = M;
    gptq_data.K = K;

    dlblasExtQuantParametersV2_t extParameters = {};
    extParameters.a_group_size_k = gptq_data.group_size;
    extParameters.a_group_size_m = 1;
    extParameters.a_zeropoints = gptq_data.qzeros;
    extParameters.a_zeropoints_type = gptq_data.qzeros_type;
    extParameters.a_scales = gptq_data.scales;
    extParameters.a_scales_type = gptq_data.scales_type;

    int k = K, m = M, n = N;
    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_N;
    int lda = transA == CUBLAS_OP_T ? k : m;
    int ldb = K;
    int ldc = M;

    cublasStatus_t status = dlblasGemmExV2(
        handle, transA, transB, m, n, k,
        &alpha, gptq_data.qweight, CUDA_R_8U, lda,
                d_input_fp32, CUDA_R_32F, ldb,
        &beta,  d_output_fp32, CUDA_R_32F, ldc,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP, &extParameters
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "dlblasGemmExV2 failed with status: " << status << std::endl;
        cudaFree(d_weight_fp32); cudaFree(d_weight_fp16); cudaFree(d_input_fp32);
        cudaFree(d_output_fp32); cudaFree(d_qweight_fp32); cudaFree(d_qzeros_fp32); cudaFree(d_scales_fp32);
        cublasDestroy(handle);
        return false;
    }
    //  --------------------------------------- FP32 FP32 ------------------------------------------

    //  --------------------------------------- FP16 FP16 ------------------------------------------
    ggml_cuda_gptq_quantize_8_bit_fp16<<<gridDim, blockDim>>>(
        d_weight_fp16, d_qweight_fp16, d_qzeros_fp16, d_scales_fp16, K, M, GROUP_SIZE
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    ggml_gptq_data gptq_data_fp16;
    gptq_data_fp16.qweight = d_qweight_fp16;
    gptq_data_fp16.qzeros = d_qzeros_fp16;
    gptq_data_fp16.scales = d_scales_fp16;
    gptq_data_fp16.bits = 8;
    gptq_data_fp16.group_size = GROUP_SIZE;
    gptq_data_fp16.scales_type = CUDA_R_16F;
    gptq_data_fp16.qzeros_type = CUDA_R_16F;
    gptq_data_fp16.num_groups = NUM_GROUPS;
    gptq_data_fp16.M = M;
    gptq_data_fp16.K = K;

    dlblasExtQuantParametersV2_t extParameters_fp16 = {};
    extParameters_fp16.a_group_size_k = gptq_data_fp16.group_size;
    extParameters_fp16.a_group_size_m = 1;
    extParameters_fp16.a_zeropoints = gptq_data_fp16.qzeros;
    extParameters_fp16.a_zeropoints_type = gptq_data_fp16.qzeros_type;
    extParameters_fp16.a_scales = gptq_data_fp16.scales;
    extParameters_fp16.a_scales_type = gptq_data_fp16.scales_type;
    const half alpha_fp16 = 1.0f;
    const half beta_fp16 = 0.0f;

    cublasStatus_t status_fp16 = dlblasGemmExV2(
        handle, transA, transB, m, n, k,
        &alpha_fp16, gptq_data_fp16.qweight, CUDA_R_8U, lda,
                d_input_fp16, CUDA_R_16F, ldb,
        &beta_fp16,  d_output_fp16, CUDA_R_16F, ldc,
        CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP, &extParameters_fp16
    );

    if (status_fp16 != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "dlblasGemmExV2 failed with status: " << status_fp16 << std::endl;
        cudaFree(d_weight_fp16); cudaFree(d_weight_fp16); cudaFree(d_input_fp16);
        cudaFree(d_output_fp16); cudaFree(d_qweight_fp32); cudaFree(d_qzeros_fp16); cudaFree(d_scales_fp16);
        cublasDestroy(handle);
        return false;
    }
    //  --------------------------------------- FP16 FP16 ------------------------------------------

    //  --------------------------------------- FP32 FP32 ------------------------------------------
    std::vector<float> dlblas_result_fp32(M * N);
    CUDA_CHECK(cudaMemcpy(dlblas_result_fp32.data(), d_output_fp32, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<uint8_t> qweight_cpu_fp32(M * K);
    std::vector<float> qzeros_cpu_fp32(M * NUM_GROUPS);
    std::vector<float> scales_cpu_fp32(M * NUM_GROUPS);
    CUDA_CHECK(cudaMemcpy(qweight_cpu_fp32.data(), d_qweight_fp32, M * K * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(qzeros_cpu_fp32.data(), d_qzeros_fp32, M * NUM_GROUPS * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(scales_cpu_fp32.data(), d_scales_fp32, M * NUM_GROUPS * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Quantization parameters check:" << std::endl;
    std::cout << "  First 5 scales: ";
    for (int i = 0; i < 5 && i < M * NUM_GROUPS; ++i) {
        std::cout << scales_cpu_fp32[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "  First 5 qzeros: ";
    for (int i = 0; i < 5 && i < M * NUM_GROUPS; ++i) {
        std::cout << qzeros_cpu_fp32[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "  First 5 qweights: ";
    for (int i = 0; i < 5 && i < M * K; ++i) {
        std::cout << (int)qweight_cpu_fp32[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "  Dequantization verification (first 5 weights):" << std::endl;
    float total_dequant_error = 0.0f;
    float max_dequant_error = 0.0f;
    int valid_dequant_count = 0;

    int group_size_m = 1;
    int group_size_k = GROUP_SIZE;
    int group_num_m = ceil(float(M) / group_size_m);
    int group_num_k = ceil(float(K) / group_size_k);

    for (int i = 0; i < 5 && i < M * K; ++i) {
        // T
        int scale_k = i % K;
        int scale_m = i / K;
        int scale_idx = (scale_m / group_size_m) * group_num_k + (scale_k / group_size_k);

        float dequant_val = (float(qweight_cpu_fp32[i]) - qzeros_cpu_fp32[scale_idx]) * scales_cpu_fp32[scale_idx];
        float abs_error = std::abs(weight_cpu_fp32[i] - dequant_val);
        float relative_error = abs_error / std::max(std::abs(weight_cpu_fp32[i]), 1e-6f) * 100.0f;

        std::cout << "    Weight[" << i << "]: orig=" << weight_cpu_fp32[i]
                    << ", quantized=" << (int)qweight_cpu_fp32[i]
                    << ", dequantized=" << dequant_val
                    << ", abs error=" << abs_error << std::endl;
    }

    for (int i = 0; i < M * K; ++i) {
        // T
        int scale_k = i % K;
        int scale_m = i / K;
        int scale_idx = (scale_m / group_size_m) * group_num_k + (scale_k / group_size_k);
        // printf("i: %d, scale_idx: %d\n", i, scale_idx);
        // printf("M * NUM_GROUPS: %d\n", M * NUM_GROUPS);

        float dequant_val = (float(qweight_cpu_fp32[i]) - qzeros_cpu_fp32[scale_idx]) * scales_cpu_fp32[scale_idx];
        float abs_error = std::abs(weight_cpu_fp32[i] - dequant_val);

        total_dequant_error += abs_error;
        if (abs_error > max_dequant_error) {
            max_dequant_error = abs_error;
        }
        valid_dequant_count++;
    }

    if (valid_dequant_count > 0) {
        float avg_dequant_error = total_dequant_error / valid_dequant_count;
        float avg_relative_error = (avg_dequant_error / std::max(std::abs(weight_cpu_fp32[0]), 1e-6f)) * 100.0f;
    std::cout << "  Dequantization stats (all " << valid_dequant_count << " weights):" << std::endl;
    std::cout << "    Mean absolute error: " << avg_dequant_error << std::endl;
    std::cout << "    Max absolute error: " << max_dequant_error << std::endl;
    }
    //  --------------------------------------- FP32 FP32 ------------------------------------------

    //  --------------------------------------- FP16 FP16 ------------------------------------------
    std::vector<half> dlblas_result_fp16(M * N);
    CUDA_CHECK(cudaMemcpy(dlblas_result_fp16.data(), d_output_fp16, M * N * sizeof(half), cudaMemcpyDeviceToHost));

    std::vector<uint8_t> qweight_cpu_fp16(M * K);
    std::vector<half> qzeros_cpu_fp16(M * NUM_GROUPS);
    std::vector<half> scales_cpu_fp16(M * NUM_GROUPS);
    CUDA_CHECK(cudaMemcpy(qweight_cpu_fp16.data(), d_qweight_fp16, M * K * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(qzeros_cpu_fp16.data(), d_qzeros_fp16, M * NUM_GROUPS * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(scales_cpu_fp16.data(), d_scales_fp16, M * NUM_GROUPS * sizeof(half), cudaMemcpyDeviceToHost));

    std::cout << "Quantization parameters check:" << std::endl;
    std::cout << "  First 5 scales: ";
    for (int i = 0; i < 5 && i < M * NUM_GROUPS; ++i) {
        std::cout << __half2float(scales_cpu_fp16[i]) << " ";
    }
    std::cout << std::endl;
    std::cout << "  First 5 qzeros: ";
    for (int i = 0; i < 5 && i < M * NUM_GROUPS; ++i) {
        std::cout << __half2float(qzeros_cpu_fp16[i]) << " ";
    }
    std::cout << std::endl;
    std::cout << "  First 5 qweights: ";
    for (int i = 0; i < 5 && i < M * K; ++i) {
        std::cout << (int)qweight_cpu_fp16[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "  Dequantization verification (first 5 weights):" << std::endl;
    float total_dequant_error_fp16 = 0.0f;
    float max_dequant_error_fp16 = 0.0f;
    int valid_dequant_count_fp16 = 0;

    for (int i = 0; i < 5 && i < M * K; ++i) {
        // T
        int scale_k = i % K;
        int scale_m = i / K;
        int scale_idx = (scale_m / group_size_m) * group_num_k + (scale_k / group_size_k);

        float dequant_val = (__half2float(qweight_cpu_fp16[i]) - __half2float(qzeros_cpu_fp16[scale_idx])) * __half2float(scales_cpu_fp16[scale_idx]);
        float abs_error = std::abs(__half2float(weight_cpu_fp16[i]) - dequant_val);
        float relative_error = abs_error / std::max(std::abs(__half2float(weight_cpu_fp16[i])), 1e-6f) * 100.0f;

        std::cout << "    Weight[" << i << "]: orig=" << __half2float(weight_cpu_fp16[i])
                    << ", quantized=" << (int)qweight_cpu_fp16[i]
                    << ", dequantized=" << dequant_val
                    << ", abs error=" << abs_error << std::endl;
    }

    for (int i = 0; i < M * K; ++i) {
        // T
        int scale_k = i % K;
        int scale_m = i / K;
        int scale_idx = (scale_m / group_size_m) * group_num_k + (scale_k / group_size_k);
        // printf("i: %d, scale_idx: %d\n", i, scale_idx);
        // printf("M * NUM_GROUPS: %d\n", M * NUM_GROUPS);

        float dequant_val = (__half2float(qweight_cpu_fp16[i]) - __half2float(qzeros_cpu_fp16[scale_idx])) * __half2float(scales_cpu_fp16[scale_idx]);
        float abs_error = std::abs(__half2float(weight_cpu_fp16[i]) - dequant_val);

        total_dequant_error_fp16 += abs_error;
        if (abs_error > max_dequant_error_fp16) {
            max_dequant_error_fp16 = abs_error;
        }
        valid_dequant_count_fp16++;
    }

    if (valid_dequant_count_fp16 > 0) {
        float avg_dequant_error = total_dequant_error_fp16 / valid_dequant_count_fp16;
        float avg_relative_error = (avg_dequant_error / std::max(std::abs(__half2float(weight_cpu_fp16[0])), 1e-6f)) * 100.0f;
    std::cout << "  Dequantization stats (all " << valid_dequant_count_fp16 << " weights):" << std::endl;
    std::cout << "    Mean absolute error: " << avg_dequant_error << std::endl;
    std::cout << "    Max absolute error: " << max_dequant_error_fp16 << std::endl;
    }
    //  --------------------------------------- FP16 FP16 ------------------------------------------

    std::cout << "Results comparison:" << std::endl;

    std::cout << "  First 8 cublasGemmEx [FP32] results: ";
    for (int i = 0; i < 8 && i < M * N; ++i) {
        std::cout << cublas_reference_result_fp32[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "  First 8 cublasGemmEx [FP16] results: ";
    for (int i = 0; i < 8 && i < M * N; ++i) {
        std::cout << __half2float(cublas_reference_result_fp16[i]) << " ";
    }
    std::cout << std::endl;


    std::cout << "  First 8 dlblasGemmExV2 [FP32] results: ";
    for (int i = 0; i < 8 && i < M * N; ++i) {
        std::cout << dlblas_result_fp32[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "  First 8 dlblasGemmExV2 [FP16] results: ";
    for (int i = 0; i < 8 && i < M * N; ++i) {
        std::cout << __half2float(dlblas_result_fp16[i]) << " ";
    }
    std::cout << std::endl;

    float cublas_vs_dlblas_diff_fp32 = std::abs(cublas_reference_result_fp32[0] - dlblas_result_fp32[0]);
    std::cout << "  cublasGemmEx [FP32][0] vs dlblasGemmExV2[FP32][0] diff: " << cublas_vs_dlblas_diff_fp32 << std::endl;

    float cublas_vs_dlblas_diff_fp16 = std::abs(__half2float(cublas_reference_result_fp16[0]) - __half2float(dlblas_result_fp32[0]));
    std::cout << "  cublasGemmEx [FP16][0] vs dlblasGemmExV2[FP32][0] diff: " << cublas_vs_dlblas_diff_fp16 << std::endl;

    float cublas_vs_dlblas_diff_fp32_fp16 = std::abs(cublas_reference_result_fp32[0] - __half2float(dlblas_result_fp16[0]));
    std::cout << "  cublasGemmEx [FP32][0] vs dlblasGemmExV2[FP16][0] diff: " << cublas_vs_dlblas_diff_fp32_fp16 << std::endl;

    float cublas_vs_dlblas_diff_fp16_fp16 = std::abs(__half2float(cublas_reference_result_fp16[0]) - __half2float(dlblas_result_fp16[0]));
    std::cout << "  cublasGemmEx [FP16][0] vs dlblasGemmExV2[FP16][0] diff: " << cublas_vs_dlblas_diff_fp16_fp16 << std::endl;


    bool is_correct = (cublas_vs_dlblas_diff_fp32 < 1e-2f) ||
                     (cublas_vs_dlblas_diff_fp32 / std::max(std::abs(cublas_reference_result_fp32[0]), 1e-6f) < 0.1f);

    float relative_error = cublas_vs_dlblas_diff_fp32 / std::max(std::abs(cublas_reference_result_fp32[0]), 1e-6f) * 100.0f;
    std::cout << "  Random weight matrix correctness: " << (is_correct ? "OK" : "FAIL") << std::endl;

    // if the max absolute error is large, print the context (using threshold and window radius)
    {
        float max_abs_err_scan = 0.0f;
        for (int i = 0; i < M * N; ++i) {
            float e = std::abs(cublas_reference_result_fp32[i] - dlblas_result_fp32[i]);
            if (e > max_abs_err_scan) max_abs_err_scan = e;
        }
        if (max_abs_err_scan > 10.0f) {
            print_max_error_context(cublas_reference_result_fp32, dlblas_result_fp32, M, N, 2, 10.0f, "RandomTest FP32");
        }
    }

    cudaFree(d_weight_fp32); cudaFree(d_weight_fp16); cudaFree(d_input_fp32);
    cudaFree(d_output_fp32); cudaFree(d_qweight_fp32); cudaFree(d_qzeros_fp32); cudaFree(d_scales_fp32);
    cudaFree(d_output_fp16); cudaFree(d_qweight_fp16); cudaFree(d_qzeros_fp16); cudaFree(d_scales_fp16);
    cublasDestroy(handle);

    return relative_error;
}

// Add: Function to test specific shape combinations for reproducing mismatch issues
float run_specific_shape_test(int M, int K, int N, int GROUP_SIZE, const std::string& test_name, std::mt19937& gen) {
    std::cout << "\n=== Testing specific shape: " << test_name << " ===" << std::endl;
    std::cout << "Shape: M=" << M << ", K=" << K << ", N=" << N << ", GROUP_SIZE=" << GROUP_SIZE << std::endl;

    const int NUM_GROUPS = (K + GROUP_SIZE - 1) / GROUP_SIZE;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    std::uniform_real_distribution<float> weight_dis(-2.0f, 2.0f);
    std::uniform_real_distribution<float> input_dis(-1.0f, 1.0f);

    std::vector<float> weight_cpu_fp32(M * K);
    std::vector<float> input_cpu_fp32(N * K);
    std::vector<half> weight_cpu_fp16(M * K);
    std::vector<half> input_cpu_fp16(N * K);

    for (int i = 0; i < M * K; ++i) {
        weight_cpu_fp32[i] = weight_dis(gen);
        weight_cpu_fp16[i] = __float2half(weight_cpu_fp32[i]);
    }
    for (int i = 0; i < N * K; ++i) {
        input_cpu_fp32[i] = input_dis(gen);
        input_cpu_fp16[i] = __float2half(input_cpu_fp32[i]);
    }

    std::vector<float> cublas_reference_result_fp32(M * N, 0.0f);

    float* d_input_fp32;
    float* d_output_cublas_fp32;

    CUDA_CHECK(cudaMalloc(&d_input_fp32, N * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_cublas_fp32, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input_fp32, input_cpu_fp32.data(), N * K * sizeof(float), cudaMemcpyHostToDevice));

    half* d_weight_fp16;
    float* d_output_fp32;
    uint8_t* d_qweight_fp32;
    float* d_qzeros_fp32;
    float* d_scales_fp32;

    CUDA_CHECK(cudaMalloc(&d_weight_fp16, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output_fp32, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_qweight_fp32, M * K * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_qzeros_fp32, M * NUM_GROUPS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scales_fp32, M * NUM_GROUPS * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_weight_fp16, weight_cpu_fp16.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));

    float* d_weight_fp32;
    CUDA_CHECK(cudaMalloc(&d_weight_fp32, M * K * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_weight_fp32, weight_cpu_fp32.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSetStream(handle, 0));
    CUBLAS_CHECK(
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                M, N, K,
                &alpha, d_weight_fp32, CUDA_R_32F, K,
                        d_input_fp32,  CUDA_R_32F, K,
                &beta,  d_output_cublas_fp32,      CUDA_R_32F, M,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    CUDA_CHECK(cudaMemcpy(cublas_reference_result_fp32.data(), d_output_cublas_fp32, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    dim3 blockDim(32, 8);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (NUM_GROUPS + blockDim.y - 1) / blockDim.y);

    ggml_cuda_gptq_quantize_8_bit_fp32<<<gridDim, blockDim>>>(
        d_weight_fp16, d_qweight_fp32, d_qzeros_fp32, d_scales_fp32, K, M, GROUP_SIZE
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    ggml_gptq_data gptq_data;
    gptq_data.qweight = d_qweight_fp32;
    gptq_data.qzeros = d_qzeros_fp32;
    gptq_data.scales = d_scales_fp32;
    gptq_data.bits = 8;
    gptq_data.group_size = GROUP_SIZE;
    gptq_data.scales_type = CUDA_R_32F;
    gptq_data.qzeros_type = CUDA_R_32F;
    gptq_data.num_groups = NUM_GROUPS;
    gptq_data.M = M;
    gptq_data.K = K;

    dlblasExtQuantParametersV2_t extParameters = {};
    extParameters.a_group_size_k = gptq_data.group_size;
    extParameters.a_group_size_m = 1;
    extParameters.a_zeropoints = gptq_data.qzeros;
    extParameters.a_zeropoints_type = gptq_data.qzeros_type;
    extParameters.a_scales = gptq_data.scales;
    extParameters.a_scales_type = gptq_data.scales_type;

    int k = K, m = M, n = N;
    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_N;
    int lda = transA == CUBLAS_OP_T ? k : m;
    int ldb = K;
    int ldc = M;

    cublasStatus_t status = dlblasGemmExV2(
        handle, transA, transB, m, n, k,
        &alpha, gptq_data.qweight, CUDA_R_8U, lda,
                d_input_fp32, CUDA_R_32F, ldb,
        &beta,  d_output_fp32, CUDA_R_32F, ldc,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP, &extParameters
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "dlblasGemmExV2 failed with status: " << status << std::endl;
        cudaFree(d_weight_fp32); cudaFree(d_weight_fp16); cudaFree(d_input_fp32);
        cudaFree(d_output_fp32); cudaFree(d_qweight_fp32); cudaFree(d_qzeros_fp32); cudaFree(d_scales_fp32);
        cublasDestroy(handle);
        return -1.0f;
    }

    std::vector<float> dlblas_result_fp32(M * N);
    CUDA_CHECK(cudaMemcpy(dlblas_result_fp32.data(), d_output_fp32, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    float total_abs_error = 0.0f;
    float max_abs_error = 0.0f;
    float total_relative_error = 0.0f;

    for (int i = 0; i < M * N; ++i) {
        float abs_error = std::abs(cublas_reference_result_fp32[i] - dlblas_result_fp32[i]);
        float relative_error = abs_error / std::max(std::abs(cublas_reference_result_fp32[i]), 1e-6f);

        total_abs_error += abs_error;
        total_relative_error += relative_error;
        if (abs_error > max_abs_error) {
            max_abs_error = abs_error;
        }
    }

    float avg_abs_error = total_abs_error / (M * N);
    float avg_relative_error = (total_relative_error / (M * N)) * 100.0f;

    std::cout << "Results for " << test_name << ":" << std::endl;
    std::cout << "  First 8 cuBLAS results: ";
    for (int i = 0; i < 8 && i < M * N; ++i) {
        std::cout << cublas_reference_result_fp32[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "  First 8 dlblas results: ";
    for (int i = 0; i < 8 && i < M * N; ++i) {
        std::cout << dlblas_result_fp32[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "  Average absolute error: " << avg_abs_error << std::endl;
    std::cout << "  Maximum absolute error: " << max_abs_error << std::endl;
    std::cout << "  Average relative error: " << avg_relative_error << "%" << std::endl;

    bool is_correct = (avg_relative_error < 1.0f);
    std::cout << "  Test result: " << (is_correct ? "PASS" : "FAIL") << std::endl;

    if (!is_correct) {
        std::cout << "  WARNING: This shape combination may trigger the FIXME mismatch issue!" << std::endl;
    }

    if (max_abs_error > 10.0f) {
        print_max_error_context(cublas_reference_result_fp32, dlblas_result_fp32, M, N, 2, 10.0f, test_name.c_str());
    }

    cudaFree(d_weight_fp32); cudaFree(d_weight_fp16); cudaFree(d_input_fp32);
    cudaFree(d_output_fp32); cudaFree(d_output_cublas_fp32);
    cudaFree(d_qweight_fp32); cudaFree(d_qzeros_fp32); cudaFree(d_scales_fp32);
    cublasDestroy(handle);

    return avg_relative_error;
}

int main() {
    std::cout << "=== CUDA GPTQ + dlblas verification test ===" << std::endl;
    std::cout << "This test compares two implementations:" << std::endl;
    std::cout << "1. cublasGemmEx (standard cuBLAS)" << std::endl;
    std::cout << "2. dlblasGemmExV2 (dlblas with quantization)" << std::endl;

    // test parameters (using smaller sizes for debugging)

    // const int M = 2;       // output dimension
    // const int K = 128;      // input dimension
    // const int N = 4;       // batch size

    // test parameters (using real sizes for debugging)
    const int M = 256;       // output dimension
    const int K = 1536;      // input dimension
    const int N = 4;       // batch size
    const int GROUP_SIZE = 64;   // GPTQ group size
    const int NUM_GROUPS = (K + GROUP_SIZE - 1) / GROUP_SIZE;

    std::cout << "Test parameters:" << std::endl;
    std::cout << "  M: " << M << std::endl;
    std::cout << "  K: " << K << std::endl;
    std::cout << "  N: " << N << std::endl;
    std::cout << "  GROUP_SIZE: " << GROUP_SIZE << std::endl;
    std::cout << "  NUM_GROUPS: " << NUM_GROUPS << std::endl;

    // generate random test cases
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> weight_dis(-2.0f, 2.0f);
    std::uniform_real_distribution<float> input_dis(-1.0f, 1.0f);

    std::cout << "\n=== Special test: random weight matrices (10 runs) ===" << std::endl;
    int random_matrix_passed = 0;
    // int random_matrix_total = 100;
    int random_matrix_total = 1;
    std::vector<float> relative_errors;

    for (int test_idx = 0; test_idx < random_matrix_total; ++test_idx) {
        std::cout << "\n--- Random matrix test #" << (test_idx + 1) << " ---" << std::endl;
        float relative_error = run_random_weight_matrix_test(M, K, N, GROUP_SIZE, NUM_GROUPS, gen);
        relative_errors.push_back(relative_error);
        if (relative_error < 1.0f) {  // relative error < 1%
            random_matrix_passed++;
        }
        if (relative_error > 100.0f) {
            std::cout << "Warning: relative error too high (" << relative_error << "%), skipping remaining tests" << std::endl;
            break;  // exit loop immediately
        }
    }

    std::cout << "\n=== Random weight matrix test statistics ===" << std::endl;
    std::cout << "Total test runs: " << relative_errors.size() << std::endl;

    // Compute statistics
    float avg_error = 0.0f;
    float max_error = 0.0f;
    float min_error = FLT_MAX;
    for (float error : relative_errors) {
        avg_error += error;
        if (error > max_error) max_error = error;
        if (error < min_error) min_error = error;
    }
    avg_error /= relative_errors.size();
    std::cout << "Average relative error: " << avg_error << "%" << std::endl;
    std::cout << "Maximum relative error: " << max_error << "%" << std::endl;
    std::cout << "Minimum relative error: " << min_error << "%" << std::endl;
    std::cout << "Passed tests: " << random_matrix_passed << "/" << relative_errors.size() << std::endl;

    // test specific shape combinations - these are actual model shapes extracted from logs
    std::cout << "\n=== Testing specific shape combinations that may trigger mismatch ===" << std::endl;

    std::random_device rd2;
    std::mt19937 gen2(rd2());

    // define test shape combinations, based on user provided log data
    struct ShapeTestCase {
        int M, K, N, group_size;
        std::string name;
    };

    std::vector<ShapeTestCase> test_cases = {
        // Small shapes - may trigger use_mul_mat_vec
        {128, 32, 1, 16, "Small_128x32_vec"},
        {128, 64, 1, 32, "Small_128x64_vec"},
        {32, 128, 1, 64, "Small_32x128_vec"},
        {64, 128, 1, 64, "Small_64x128_vec"},

        // Medium shapes
        {256, 1536, 1, 64, "Medium_256x1536_vec"},
        {1536, 1536, 1, 64, "Medium_1536x1536_vec"},
        {1536, 8960, 1, 128, "Medium_1536x8960_vec"},
        {8960, 1536, 1, 128, "Medium_8960x1536_vec"},

        // Large shape - embedding layer
        {151936, 1536, 1, 64, "Large_151936x1536_vec"},

        // Test with batch size > 1 (should not trigger vec path)
        {256, 1536, 4, 64, "Medium_256x1536_batch4"},
        {1536, 1536, 4, 64, "Medium_1536x1536_batch4"},
    };

    std::vector<float> shape_test_errors;
    int shape_tests_passed = 0;

    for (const auto& test_case : test_cases) {
        std::cout << "\n--- Testing shape: " << test_case.name << " ---" << std::endl;
        std::cout << "Expected to trigger: ";

        // predict if this shape will trigger use_mul_mat_vec path
        bool likely_vec_path = (test_case.N == 1) &&
                              ((test_case.M <= 256 && test_case.K <= 2048) ||
                               (test_case.K <= 256 && test_case.M <= 2048));

        if (likely_vec_path) {
            std::cout << "use_mul_mat_vec path (potential mismatch with dlblas)" << std::endl;
        } else {
            std::cout << "normal cuBLAS path" << std::endl;
        }

        float error = run_specific_shape_test(test_case.M, test_case.K, test_case.N,
                                            test_case.group_size, test_case.name, gen2);

        if (error >= 0.0f) {
            shape_test_errors.push_back(error);
            if (error < 1.0f) {
                shape_tests_passed++;
            }

            if (error > 5.0f && likely_vec_path) {
                std::cout << "*** CONFIRMED: High error (" << error
                         << "%) on shape that should trigger vec path! ***" << std::endl;
            }
        }
    }

    std::cout << "\n=== Shape-specific test summary ===" << std::endl;
    std::cout << "Total shape tests: " << shape_test_errors.size() << std::endl;
    std::cout << "Passed shape tests: " << shape_tests_passed << "/" << shape_test_errors.size() << std::endl;

    if (!shape_test_errors.empty()) {
        float avg_shape_error = 0.0f;
        float max_shape_error = 0.0f;
        for (float error : shape_test_errors) {
            avg_shape_error += error;
            if (error > max_shape_error) max_shape_error = error;
        }
        avg_shape_error /= shape_test_errors.size();

        std::cout << "Average shape test error: " << avg_shape_error << "%" << std::endl;
        std::cout << "Maximum shape test error: " << max_shape_error << "%" << std::endl;

        // identify problematic shapes
        std::cout << "\nProblematic shapes (error > 5%):" << std::endl;
        for (size_t i = 0; i < test_cases.size() && i < shape_test_errors.size(); ++i) {
            if (shape_test_errors[i] > 5.0f) {
                std::cout << "  " << test_cases[i].name << ": " << shape_test_errors[i] << "%" << std::endl;
            }
        }
    }

    std::cout << "\n=== Final Summary ===" << std::endl;
    std::cout << "This test helps identify the shapes that trigger the FIXME mismatch issue." << std::endl;
    std::cout << "High errors (>5%) on small matrix shapes with N=1 likely indicate" << std::endl;
    std::cout << "the use_mul_mat_vec vs dlblas mismatch mentioned in the FIXME comment." << std::endl;
}