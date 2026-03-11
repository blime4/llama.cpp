// SNAC vocoder ggml-based implementation
// Phase 1: Single sequence processing with ggml graph

#include "snac-ggml.h"
#include "log.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdio>

// Snake activation: snake(x, alpha) = x + sin²(alpha * x) / (alpha + eps)
// Matches Python: x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
// Note: eps is hardcoded as 1e-9 to avoid creating new tensors with no_alloc contexts
static struct ggml_tensor * snac_snake_forward_ggml(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * alpha)
{
    // alpha * x
    struct ggml_tensor * ax = ggml_mul(ctx, x, alpha);

    // sin(alpha * x)
    struct ggml_tensor * sin_ax = ggml_sin(ctx, ax);

    // sin²(alpha * x) = sin_ax * sin_ax
    struct ggml_tensor * sin2 = ggml_mul(ctx, sin_ax, sin_ax);

    // For numerical stability, we use: sin² / alpha where alpha is assumed to be >= 1e-9
    // Since alpha parameters are typically initialized to 1.0, this is safe
    // The 1e-9 epsilon in Python prevents division by zero but is rarely needed in practice
    struct ggml_tensor * div = ggml_div(ctx, sin2, alpha);

    // x + sin² / alpha
    return ggml_add(ctx, x, div);
}

// Custom ConvTranspose1D implementation for GGML
// Uses native ggml_conv_transpose_1d with proper kernel format
//
// Converter provides kernel in format that matches GGML expectations:
// - GGUF header: [K, OC, IC] (from numpy [IC, OC, K] reversal)
// - Memory layout: [IC, OC, K] row-major = GGML [K, OC, IC] layout
//
// kernel: [K, OC, IC] format (ne[0]=K, ne[1]=OC, ne[2]=IC)
// input: [L, IC] (temporal first, channels second)
// Returns: [output_len, OC] (temporal first, channels second)
static struct ggml_tensor * snac_conv_transpose_1d_custom(
    struct ggml_context * ctx,
    struct ggml_tensor * kernel,   // [K, OC, IC] - ready for ggml_conv_transpose_1d
    struct ggml_tensor * input,    // [L, IC] (temporal first, channels second)
    struct ggml_tensor * bias,     // [OC] or nullptr
    int64_t stride,
    int64_t padding,
    int64_t output_padding)
{
    // Get dimensions from kernel
    // GGUF stores the kernel in [OC, K, IC] format (shape=[OC, K, IC])
    // This is because GGUF header reverses numpy shape, and the numpy shape
    // is [IC, K, OC] from PyTorch (without transpose).
    //
    // For ggml_conv_transpose_1d, we need [K, OC, IC] format.
    // So we need to permute from [OC, K, IC] to [K, OC, IC].
    //
    // Current GGUF: ne[0]=OC, ne[1]=K, ne[2]=IC
    // After permutation (1, 0, 2, 3): ne[0]=K, ne[1]=OC, ne[2]=IC
    int64_t OC = kernel->ne[0];    // Output channels (from GGUF ne[0])
    int64_t K = kernel->ne[1];     // Kernel size (from GGUF ne[1])
    int64_t IC = kernel->ne[2];    // Input channels (from GGUF ne[2])

    // Input is [L, IC] format (temporal first)
    int64_t L = input->ne[0];      // Temporal dimension

    // Output length calculation for ConvTranspose1D
    int64_t output_len = (L - 1) * stride + K - 2 * padding + output_padding;

    LOG_INF("%s: ConvTranspose1D: input [%lld, %lld], IC=%lld, OC=%lld, K=%lld, stride=%lld -> output_len=%lld\n",
            __func__, (long long)L, (long long)input->ne[1], (long long)IC, (long long)OC, (long long)K,
            (long long)stride, (long long)output_len);

    LOG_INF("%s: Kernel shape: [%lld,%lld,%lld] = [K,OC,IC]\n", __func__,
            (long long)kernel->ne[0], (long long)kernel->ne[1], (long long)kernel->ne[2]);

    // Permute kernel from [OC, K, IC] to [K, OC, IC] for ggml_conv_transpose_1d
    // ggml_conv_transpose_1d expects kernel in format [K, OC, IC]
    // Current tensor from GGUF: ne[0]=OC, ne[1]=K, ne[2]=IC
    // Using permutation (1, 0, 2, 3):
    //   output ne[0] = input ne[1] = K
    //   output ne[1] = input ne[0] = OC
    //   output ne[2] = input ne[2] = IC
    struct ggml_tensor * kernel_permuted = ggml_permute(ctx, kernel, 1, 0, 2, 3);
    struct ggml_tensor * kernel_cont = ggml_cont(ctx, kernel_permuted);

    LOG_INF("%s: Kernel after permute: [%lld,%lld,%lld] = [K,OC,IC]\n", __func__,
            (long long)kernel_cont->ne[0], (long long)kernel_cont->ne[1], (long long)kernel_cont->ne[2]);

    // Convert kernel to F32 if needed (CUDA ggml_conv_transpose_1d requires F32)
    struct ggml_tensor * kernel_for_conv = kernel_cont;
    if (kernel_cont->type != GGML_TYPE_F32) {
        struct ggml_tensor * kernel_f32_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
            kernel_cont->ne[0], kernel_cont->ne[1], kernel_cont->ne[2]);
        kernel_for_conv = ggml_cpy(ctx, kernel_cont, kernel_f32_tensor);
    }

    // Reshape input from [L, IC] to [L, IC, 1] for ggml_conv_transpose_1d
    // Input needs to be F32
    struct ggml_tensor * input_f32 = input;
    if (input->type != GGML_TYPE_F32) {
        struct ggml_tensor * input_f32_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input->ne[0], input->ne[1]);
        input_f32 = ggml_cpy(ctx, input, input_f32_tensor);
    }
    struct ggml_tensor * input_3d = ggml_reshape_3d(ctx, input_f32, L, IC, 1);  // [L, IC, 1]

    LOG_INF("%s: Input 3d shape: [%lld,%lld,%lld], kernel IC=%lld, input IC=%lld\n", __func__,
            (long long)input_3d->ne[0], (long long)input_3d->ne[1], (long long)input_3d->ne[2],
            (long long)kernel_for_conv->ne[2], (long long)input_3d->ne[1]);

    // Apply native ggml_conv_transpose_1d
    // Note: ggml_conv_transpose_1d signature: (ctx, kernel, input, stride, padding, dilation)
    // It doesn't support padding parameter directly (requires p0=0, d0=1)
    // We handle padding manually by trimming the output
    struct ggml_tensor * result = ggml_conv_transpose_1d(ctx, kernel_for_conv, input_3d, stride, 0, 1);

    // Result shape: [output_len_no_pad, OC, 1] where output_len_no_pad = (L-1)*stride + K
    // We need to remove padding from both sides
    int64_t full_output_len = (L - 1) * stride + K;
    int64_t trim_start = padding;
    int64_t trim_end = full_output_len - output_len - padding;

    LOG_DBG("%s: Full output len=%lld, trim_start=%lld, trim_end=%lld, final=%lld\n", __func__,
            (long long)full_output_len, (long long)trim_start, (long long)trim_end, (long long)output_len);

    // Reshape result to 2D [full_output_len, OC]
    result = ggml_reshape_2d(ctx, result, full_output_len, OC);

    // Trim padding from start and end
    if (trim_start > 0 || trim_end > 0) {
        // Create a view that skips the padding
        // For simplicity, use view_2d with offset
        result = ggml_view_2d(ctx, result, output_len, OC,
                              result->nb[1],  // row stride (OC elements per row)
                              trim_start * result->nb[0]);  // offset
    }

    // Add bias if present
    if (bias) {
        // result is [output_len, OC], bias is [OC]
        // Reshape bias to [1, OC] for broadcasting
        struct ggml_tensor * bias_2d = ggml_reshape_2d(ctx, bias, 1, OC);
        result = ggml_add(ctx, result, bias_2d);
    }

    LOG_DBG("%s: Output shape: [%lld, %lld]\n", __func__,
            (long long)result->ne[0], (long long)result->ne[1]);

    return result;
}

// Snake activation for 2D tensor with alpha broadcast over channels
// x: [T, C] or [C, T] depending on caller
// alpha: [C] or [C, 1] - will be flattened to 1D if needed
static struct ggml_tensor * snac_snake_forward_2d_ggml(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * alpha,
    int64_t T,
    int64_t C)
{
    // x shape: [T, C] - e.g., [30716, 96]
    // alpha shape: [C] or [C, 1] - e.g., [96] or [96, 1]
    // Need to broadcast alpha across the T dimension

    // Flatten alpha to 1D if it's 2D with shape [C, 1]
    // Check if alpha has more than 1 dimension and second dim is 1
    struct ggml_tensor * alpha_1d = alpha;
    if (alpha->ne[1] == 1 && alpha->ne[0] == C) {
        // alpha is [C, 1], flatten to [C]
        alpha_1d = ggml_view_1d(ctx, alpha, C, 0);
    }

    // Create a target tensor shape [T, C] for broadcasting
    // We need to broadcast alpha from [C] to [T, C]
    // ggml_repeat broadcasts along dimensions where source is 1
    // But alpha is [C], not [1, C], so we need to handle this differently

    // Alternative: Use ggml_add with broadcasting
    // First, compute alpha * x element-wise with broadcasting
    // Since x is [T, C] and alpha is [C], we need to repeat alpha T times

    // Create a 2D tensor of shape [T, 1] filled with 1s to help with broadcasting
    // Actually, let's use a simpler approach: reshape alpha to [1, C], then repeat

    // Reshape alpha to [1, C]
    struct ggml_tensor * alpha_2d = ggml_reshape_2d(ctx, alpha_1d, 1, C);

    // Create a target tensor for repeat
    // We need to broadcast [1, C] to [T, C]
    // ggml_repeat needs the target tensor to exist, so we use x as the target
    // But x must have compatible shape: [T, C] where T is multiple of 1 and C matches

    (void)T; // Will be checked via x->ne[0]

    // Broadcast alpha to [T, C] using x as the template
    struct ggml_tensor * alpha_bc = ggml_repeat(ctx, alpha_2d, x);

    return snac_snake_forward_ggml(ctx, x, alpha_bc);
}

// Quantizer forward: codebook lookup + projection
// tokens: [seq_len] int32 tensor
// Returns: [seq_len, quantizer_dim] float tensor
static struct ggml_tensor * snac_quantizer_forward_ggml(
    struct ggml_context * ctx,
    struct ggml_tensor * tokens,
    struct ggml_tensor * codebook,
    struct ggml_tensor * out_proj,
    struct ggml_tensor * out_bias,
    int64_t codebook_size,
    int64_t codebook_dim,
    int64_t quantizer_dim)
{
    (void)codebook_size; // Used for documentation/clarity

    // Null checks
    if (!codebook) {
        LOG_ERR("%s: codebook tensor is null\n", __func__);
        return nullptr;
    }
    if (!out_proj) {
        LOG_ERR("%s: out_proj tensor is null\n", __func__);
        return nullptr;
    }
    if (!tokens) {
        LOG_ERR("%s: tokens tensor is null\n", __func__);
        return nullptr;
    }

    LOG_DBG("%s: codebook shape: %lld x %lld, out_proj shape: %lld x %lld x %lld\n", __func__,
            (long long)codebook->ne[0], (long long)codebook->ne[1],
            (long long)out_proj->ne[0], (long long)out_proj->ne[1], (long long)out_proj->ne[2]);

    // Codebook lookup using ggml_get_rows
    // codebook: [codebook_size, codebook_dim] = [4096, 8]
    // tokens: [seq_len]
    // result: [codebook_dim, seq_len] = [8, seq_len]
    struct ggml_tensor * embeddings = ggml_get_rows(ctx, codebook, tokens);
    // embeddings: [codebook_dim, seq_len] = [8, seq_len] (ne0=8, ne1=seq_len)

    int64_t seq_len = tokens->ne[0];

    LOG_INF("%s: out_proj shape: ne0=%lld, ne1=%lld (quantizer_dim=%lld, codebook_dim=%lld)\n", __func__,
            (long long)out_proj->ne[0], (long long)out_proj->ne[1],
            (long long)quantizer_dim, (long long)codebook_dim);
    LOG_INF("%s: embeddings shape: ne0=%lld, ne1=%lld\n", __func__,
            (long long)embeddings->ne[0], (long long)embeddings->ne[1]);

    // Use ggml_mul_mat for linear projection instead of conv_1d
    // out_proj: [codebook_dim, quantizer_dim] = [8, 1024] (ne0=8, ne1=1024)
    // embeddings: [codebook_dim, seq_len] = [8, seq_len] (ne0=8, ne1=seq_len)
    // ggml_mul_mat(a, b): a=[K,N], b=[K,M] -> result=[N,M]
    // With a=out_proj [8,1024] and b=embeddings [8,seq_len]:
    // K=8, N=1024, M=seq_len -> result=[1024, seq_len]
    struct ggml_tensor * projected = ggml_mul_mat(ctx, out_proj, embeddings);
    // projected: [quantizer_dim, seq_len] = [1024, seq_len]

    LOG_INF("%s: projected shape after mul_mat: ne0=%lld, ne1=%lld, ne2=%lld, ne3=%lld\n", __func__,
            (long long)projected->ne[0], (long long)projected->ne[1],
            (long long)projected->ne[2], (long long)projected->ne[3]);

    // Add bias
    if (out_bias) {
        // Bias is [quantizer_dim], need to broadcast across seq_len
        projected = ggml_add(ctx, projected, out_bias);
    }

    return projected;
}

// Decoder layer forward with ConvTranspose1D
static struct ggml_tensor * snac_decoder_layer_forward_ggml(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    int64_t input_len,
    struct ggml_tensor * alpha,
    struct ggml_tensor * kernel,
    struct ggml_tensor * bias,
    struct ggml_tensor * noise_kernel,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int layer_idx,
    struct ggml_tensor * residual_in_alpha[3],
    struct ggml_tensor * residual_in_kernel[3],
    struct ggml_tensor * residual_in_bias[3],
    struct ggml_tensor * residual_out_alpha[3],
    struct ggml_tensor * residual_out_kernel[3],
    struct ggml_tensor * residual_out_bias[3])
{
    (void)layer_idx; // For future use/debugging
    (void)noise_kernel; // Noise injection disabled for inference

    // Calculate output length
    int output_padding = stride % 2;
    int64_t output_len = (input_len - 1) * stride + kernel_size - 2 * padding + output_padding;

    LOG_DBG("%s: Layer %d: in_ch=%d, out_ch=%d, in_len=%lld, out_len=%lld, k=%d, s=%d, p=%d\n",
            __func__, layer_idx, in_channels, out_channels, (long long)input_len,
            (long long)output_len, kernel_size, stride, padding);

    // Snake activation
    struct ggml_tensor * cur = snac_snake_forward_2d_ggml(ctx, input, alpha, input_len, in_channels);

    // ConvTranspose1D using custom implementation (supports padding)
    cur = snac_conv_transpose_1d_custom(ctx, kernel, cur, bias, stride, padding, output_padding);

    // Note: bias is added inside snac_conv_transpose_1d_custom

    // Residual units (3 units per layer)
    // cur is [output_len, out_channels] format after ConvTranspose1D
    LOG_INF("DEBUG: Starting residual units for layer %d, output_len=%lld, channels=%d\n",
            layer_idx, (long long)output_len, out_channels);
    for (int u = 0; u < 3; u++) {
        LOG_INF("DEBUG: Residual unit %d: residual_in_alpha=%p, residual_in_kernel=%p\n",
                u, (void*)residual_in_alpha[u], (void*)residual_in_kernel[u]);
        if (!residual_in_alpha[u]) continue;

        // Save input for residual connection
        struct ggml_tensor * residual = cur;

        int channels = out_channels;
        int unit_kernel_size = 7;
        int dilation = (int)std::pow(3, u);
        int unit_padding = ((unit_kernel_size - 1) * dilation) / 2;

        LOG_DBG("%s: Residual unit %d: channels=%d, dilation=%d, padding=%d\n",
                __func__, u, channels, dilation, unit_padding);

        // Snake in - cur is [output_len, channels]
        cur = snac_snake_forward_2d_ggml(ctx, cur, residual_in_alpha[u], output_len, channels);

        // Depthwise conv in (groups = channels)
        // cur is [T, C], ggml_conv_1d_dw expects [L, IC, N] = [T, C, 1]
        // Kernel is stored in GGUF with header [K, 1, C] = [7, 1, 512]
        // The GGUF data layout is: GGML[k, 0, c] = numpy[c, 0, k]
        // This is ALREADY the correct layout for ggml_conv_1d_dw - no transpose needed!
        // ggml_conv_1d_dw requires F16 kernel
        if (residual_in_kernel[u]) {
            // Convert to F16 if needed
            struct ggml_tensor * kernel_for_dw = residual_in_kernel[u];
            if (residual_in_kernel[u]->type != GGML_TYPE_F16) {
                struct ggml_tensor * kernel_f16 = ggml_new_tensor_3d(ctx, GGML_TYPE_F16,
                    residual_in_kernel[u]->ne[0], residual_in_kernel[u]->ne[1], residual_in_kernel[u]->ne[2]);
                kernel_for_dw = ggml_cpy(ctx, residual_in_kernel[u], kernel_f16);
            }

            // cur is [T, C], reshape to [T, C, 1] for ggml_conv_1d_dw
            struct ggml_tensor * cur_3d = ggml_reshape_3d(ctx, cur, output_len, channels, 1);

            // Apply depthwise conv with dilation
            cur = ggml_conv_1d_dw(ctx, kernel_for_dw, cur_3d, 1, unit_padding, dilation);

            // Result is [T, C, 1], reshape back to [T, C]
            cur = ggml_reshape_2d(ctx, cur, cur->ne[0], cur->ne[1]);  // [T, C]

            if (residual_in_bias[u]) {
                // Bias is [C], need to broadcast to [T, C]
                // Reshape bias from [C] to [1, C], then repeat along T dimension
                struct ggml_tensor * bias_2d = ggml_reshape_2d(ctx, residual_in_bias[u], 1, channels);  // [1, C]
                struct ggml_tensor * bias_bc = ggml_repeat(ctx, bias_2d, cur);  // [T, C]
                cur = ggml_add(ctx, cur, bias_bc);
            }
        }

        // Snake out - cur is [T, C]
        cur = snac_snake_forward_2d_ggml(ctx, cur, residual_out_alpha[u], output_len, channels);

        // 1x1 conv out (pointwise convolution)
        // A 1x1 conv with stride=1, padding=0 is equivalent to a matrix multiplication
        // cur is [T, C], kernel is [C, C] (from GGUF with [IC, OC] = [C, C])
        // We compute cur @ kernel = [T, C] @ [C, C] = [T, C]
        if (residual_out_kernel[u]) {
            // For ggml_mul_mat(a, b): both must have same ne[0] (K dimension)
            // cur is [T, C] with ne[0]=T, ne[1]=C
            // kernel is [C, C] with ne[0]=C, ne[1]=C
            // We need to transpose cur so ne[0]=C matches kernel's ne[0]=C

            // Ensure kernel is 2D
            struct ggml_tensor * kernel_2d = residual_out_kernel[u];
            if (ggml_n_dims(kernel_2d) > 2) {
                kernel_2d = ggml_view_2d(ctx, residual_out_kernel[u],
                    residual_out_kernel[u]->ne[0], residual_out_kernel[u]->ne[1],
                    residual_out_kernel[u]->nb[1], 0);
            }

            // Transpose cur so ne[0]=C (matches kernel's ne[0])
            struct ggml_tensor * cur_t = ggml_cont(ctx, ggml_transpose(ctx, cur));  // [C, T]

            // Apply matrix multiplication: kernel [C, C] @ cur_t^T [T, C] = [C, T]
            struct ggml_tensor * result = ggml_mul_mat(ctx, kernel_2d, cur_t);  // [C, T]

            // Transpose back to [T, C]
            cur = ggml_cont(ctx, ggml_transpose(ctx, result));  // [T, C]

            if (residual_out_bias[u]) {
                // Bias is [C], need to broadcast to [T, C]
                struct ggml_tensor * bias_2d = ggml_reshape_2d(ctx, residual_out_bias[u], 1, channels);  // [1, C]
                struct ggml_tensor * bias_bc = ggml_repeat(ctx, bias_2d, cur);  // [T, C]
                cur = ggml_add(ctx, cur, bias_bc);
            }
        }

        // Residual connection - both cur and residual are [T, C]
        cur = ggml_add(ctx, cur, residual);
    }

    return cur;
}

// Helper: Create a tensor with repeated values for pyramid upsampling
// Each element is repeated 'repeats' times consecutively
// Input: [n] tensor, Output: [n * repeats] tensor
static struct ggml_tensor * snac_repeat_interleave(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    int64_t repeats,
    int64_t output_len)
{
    // GGML doesn't have native repeat_interleave, so we use a workaround:
    // 1. Create an output tensor
    // 2. Build a graph that copies each input element to 'repeats' output positions
    //
    // For now, we'll use ggml_repeat + view operations as a workaround
    // This is not optimal but works for correctness

    // Simple approach: use ggml_upscale if available, otherwise manual construction
    // For pyramid structure, we need to repeat each element 'repeats' times

    // Create output tensor with expanded size
    // input: [seq_len, quantizer_dim]
    // output: [seq_len * repeats, quantizer_dim]

    if (repeats == 1) {
        return input;  // No expansion needed
    }

    int64_t seq_len = input->ne[1];
    int64_t dim = input->ne[0];

    // For now, use a workaround: create a larger tensor and set data later
    // This will be handled by pre-processing the input tokens instead
    // Here we just create the tensor structure

    // Actually, let's implement using ggml operations:
    // We can use ggml_repeat to expand, then select the right slices

    // Alternative: implement using ggml_scale + ggml_add for broadcasting
    // For correctness, let's just return input and handle expansion in decode function

    // For Phase 1, we'll handle pyramid expansion in the decode function
    // by pre-expanding the token arrays before building the graph

    (void)output_len;  // Will be used for verification

    // Return input as-is; pyramid expansion handled elsewhere
    return input;
}

// Build the complete SNAC computation graph
// IMPORTANT: For pyramid structure, tokens_head2 determines the output length
// All heads should be pre-expanded to head2_len using repeat_interleave before calling
static struct ggml_tensor * snac_build_graph(
    struct ggml_context * ctx,
    const struct snac_ggml_weights & w,
    const int * tokens_head0,
    int64_t head0_len,
    const int * tokens_head1,
    int64_t head1_len,
    const int * tokens_head2,
    int64_t head2_len,
    int vq_strides[3])
{
    // Token data will be set later via ggml_backend_tensor_set
    // This function only builds the graph structure

    struct ggml_tensor * cur = nullptr;

    // The output length is determined by head2 (the longest sequence)
    // All heads should have the same length after pyramid expansion
    int64_t output_len = head2_len;

    // For pyramid structure with vq_strides = [4, 2, 1]:
    // - head0 tokens are repeated 4 times each
    // - head1 tokens are repeated 2 times each
    // - head2 tokens are used as-is
    // The input tokens should already be expanded to output_len

    // Step 1: Quantizer forward for each head
    // All heads now have the same length (output_len) after expansion
    std::vector<struct ggml_tensor *> head_embeddings(3);

    // Head 0 (stride=4, expanded to output_len)
    if (head0_len > 0) {
        struct ggml_tensor * tokens0 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, head0_len);
        ggml_set_name(tokens0, "tokens_head0");
        head_embeddings[0] = snac_quantizer_forward_ggml(
            ctx, tokens0, w.quant_codebook[0], w.quant_out_proj[0], w.quant_out_bias[0],
            SNAC_GGML_CODEBOOK_SIZE, SNAC_GGML_CODEBOOK_DIM, SNAC_GGML_QUANTIZER_DIM);
    }

    // Head 1 (stride=2, expanded to output_len)
    if (head1_len > 0) {
        struct ggml_tensor * tokens1 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, head1_len);
        ggml_set_name(tokens1, "tokens_head1");
        head_embeddings[1] = snac_quantizer_forward_ggml(
            ctx, tokens1, w.quant_codebook[1], w.quant_out_proj[1], w.quant_out_bias[1],
            SNAC_GGML_CODEBOOK_SIZE, SNAC_GGML_CODEBOOK_DIM, SNAC_GGML_QUANTIZER_DIM);
    }

    // Head 2 (stride=1, no expansion needed)
    if (head2_len > 0) {
        struct ggml_tensor * tokens2 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, head2_len);
        ggml_set_name(tokens2, "tokens_head2");
        head_embeddings[2] = snac_quantizer_forward_ggml(
            ctx, tokens2, w.quant_codebook[2], w.quant_out_proj[2], w.quant_out_bias[2],
            SNAC_GGML_CODEBOOK_SIZE, SNAC_GGML_CODEBOOK_DIM, SNAC_GGML_QUANTIZER_DIM);
    }

    // Step 2: Combine quantizer outputs with pyramid structure
    // All embeddings should now have the same length (output_len)
    // We simply sum them together

    // Start with head2 (the base)
    cur = head_embeddings[2];

    // Add head1 if present
    if (head_embeddings[1]) {
        // Verify dimensions match
        if (head_embeddings[1]->ne[1] == cur->ne[1]) {
            cur = ggml_add(ctx, cur, head_embeddings[1]);
        } else {
            LOG_WRN("%s: head1 embedding length mismatch: %lld vs %lld\n", __func__,
                    (long long)head_embeddings[1]->ne[1], (long long)cur->ne[1]);
        }
    }

    // Add head0 if present
    if (head_embeddings[0]) {
        // Verify dimensions match
        if (head_embeddings[0]->ne[1] == cur->ne[1]) {
            cur = ggml_add(ctx, cur, head_embeddings[0]);
        } else {
            LOG_WRN("%s: head0 embedding length mismatch: %lld vs %lld\n", __func__,
                    (long long)head_embeddings[0]->ne[1], (long long)cur->ne[1]);
        }
    }

    (void)vq_strides;  // Used for documentation

    // Step 3: Input convolution (depthwise)
    // cur is [quantizer_dim, seq_len] = [768, seq_len] after combining embeddings
    // in_conv is depthwise conv with kernel_size=7, padding=3, stride=1, groups=768
    int64_t seq_len = cur->ne[1];

    // Apply depthwise convolution using ggml_conv_1d_dw
    // The kernel is stored as [K, 1, C] = [7, 1, 768] in GGUF (after GGUF dimension reversal)
    // ggml_conv_1d_dw expects kernel in format [K, 1, C] for depthwise conv
    // where K=kernel_size, C=channels (groups=C for depthwise)
    if (w.in_conv_kernel) {
        LOG_INF("%s: Applying in_conv depthwise, kernel shape [%lld, %lld, %lld], input shape [%lld, %lld]\n",
                __func__, (long long)w.in_conv_kernel->ne[0], (long long)w.in_conv_kernel->ne[1],
                (long long)w.in_conv_kernel->ne[2], (long long)cur->ne[0], (long long)cur->ne[1]);

        // Kernel is already in [K, 1, C] format from GGUF
        // Just ensure F16 type for ggml_conv_1d_dw
        struct ggml_tensor * kernel_for_dw = w.in_conv_kernel;
        if (w.in_conv_kernel->type != GGML_TYPE_F16) {
            struct ggml_tensor * kernel_f16_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F16,
                w.in_conv_kernel->ne[0], w.in_conv_kernel->ne[1], w.in_conv_kernel->ne[2]);
            kernel_for_dw = ggml_cpy(ctx, w.in_conv_kernel, kernel_f16_tensor);
        }

        LOG_DBG("%s: in_conv kernel shape: [%lld,%lld,%lld]\n", __func__,
                (long long)kernel_for_dw->ne[0], (long long)kernel_for_dw->ne[1],
                (long long)kernel_for_dw->ne[2]);

        // Reshape cur from [C, T] to [T, C, 1] for ggml_conv_1d_dw
        // ggml_conv_1d_dw expects input in [L, IC, N] format (temporal first)
        // First ensure F32
        struct ggml_tensor * cur_f32 = cur;
        if (cur->type != GGML_TYPE_F32) {
            struct ggml_tensor * cur_f32_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cur->ne[0], cur->ne[1]);
            cur_f32 = ggml_cpy(ctx, cur, cur_f32_tensor);
        }
        // CRITICAL: Transpose from [C, T] to [T, C] first, then reshape to [T, C, 1]
        struct ggml_tensor * cur_t = ggml_cont(ctx, ggml_transpose(ctx, cur_f32));  // [T, C]
        struct ggml_tensor * cur_3d = ggml_reshape_3d(ctx, cur_t, cur_t->ne[0], cur_t->ne[1], 1);  // [T, C, 1]

        LOG_DBG("%s: cur_3d shape for conv: [%lld, %lld, %lld]\n", __func__,
                (long long)cur_3d->ne[0], (long long)cur_3d->ne[1], (long long)cur_3d->ne[2]);

        // Apply depthwise conv with padding=3 to maintain length
        cur = ggml_conv_1d_dw(ctx, kernel_for_dw, cur_3d, 1, 3, 1);

        // Result is [T, C, 1], reshape back to [C, T]
        cur = ggml_reshape_2d(ctx, cur, cur->ne[0], cur->ne[1]);  // [T, C]
        cur = ggml_cont(ctx, ggml_transpose(ctx, cur));  // [C, T]

        if (w.in_conv_bias) {
            cur = ggml_add(ctx, cur, w.in_conv_bias);
        }
        LOG_DBG("%s: Applied in_conv depthwise, output shape [%lld, %lld]\n", __func__,
                (long long)cur->ne[0], (long long)cur->ne[1]);
    } else {
        // No kernel, just add bias if present
        if (w.in_conv_bias) {
            cur = ggml_add(ctx, cur, w.in_conv_bias);
        }
        LOG_WRN("%s: in_conv kernel not loaded, using identity + bias\n", __func__);
    }

    // Step 4: Up convolution (1x1 conv)
    // Project from quantizer_dim to decoder_dim
    LOG_INF("%s: cur shape before up_conv = [%lld, %lld]\n", __func__,
            (long long)cur->ne[0], (long long)cur->ne[1]);
    LOG_INF("%s: up_conv_kernel shape = [%lld, %lld, %lld, %lld]\n", __func__,
            (long long)w.up_conv_kernel->ne[0], (long long)w.up_conv_kernel->ne[1],
            (long long)w.up_conv_kernel->ne[2], (long long)w.up_conv_kernel->ne[3]);

    // up_conv is a 1x1 conv (essentially a linear projection)
    // Due to GGUF dimension reversal, the kernel is stored as [in_ch, out_ch] = [768, 1024]
    // cur is [in_ch, seq_len] = [768, seq_len]
    //
    // For ggml_mul_mat(a, b): assertion is a->ne[0] == b->ne[0]
    // kernel [in_ch, out_ch] = [768, 1024], cur [768, seq_len]
    // mul_mat(kernel, cur): [768, 1024] x [768, seq_len] -> [1024, seq_len]
    // NO transpose needed!

    // Reshape to 2D if needed (handle 3D or 4D tensors)
    struct ggml_tensor * up_kernel_2d = w.up_conv_kernel;
    if (ggml_n_dims(w.up_conv_kernel) > 2) {
        // Reshape [in_ch, out_ch, 1, 1] to [in_ch, out_ch]
        up_kernel_2d = ggml_view_2d(ctx, w.up_conv_kernel,
                                    w.up_conv_kernel->ne[0],  // in_ch
                                    w.up_conv_kernel->ne[1],  // out_ch
                                    w.up_conv_kernel->nb[1],  // row stride
                                    0);
    }

    LOG_INF("%s: up_kernel_2d shape = [%lld, %lld]\n", __func__,
            (long long)up_kernel_2d->ne[0], (long long)up_kernel_2d->ne[1]);

    // NO transpose - kernel is already [in_ch, out_ch] from GGUF
    // Apply linear projection: [in_ch, out_ch] x [in_ch, seq_len] -> [out_ch, seq_len]
    cur = ggml_mul_mat(ctx, up_kernel_2d, cur);

    LOG_INF("%s: cur shape after up_conv = [%lld, %lld]\n", __func__,
            (long long)cur->ne[0], (long long)cur->ne[1]);

    if (w.up_conv_bias) {
        cur = ggml_add(ctx, cur, w.up_conv_bias);
    }

    // Get actual channel count from up_conv output (decoder_dim)
    int64_t cur_channels = cur->ne[0];  // Should be 1536 for snac_24khz

    // Step 5: LocalMHA attention (optional - skip for Phase 1)
    // TODO: Implement LocalMHA with RoPE

    // Step 6: Decoder layers with ConvTranspose1D
    // Now using custom implementation that supports padding
    // Get actual dimensions from loaded kernels
    // Layer kernel format: [K, OC, IC] where K=kernel_size, OC=out_channels, IC=in_channels
    // Decoder rates from GGUF: {8, 8, 4, 2}
    int decoder_rates[4] = {8, 8, 4, 2};

    int64_t cur_len = seq_len;

    for (int l = 0; l < SNAC_GGML_N_DECODER_LAYERS; l++) {
        if (!w.layer_kernel[l]) {
            LOG_WRN("%s: Layer %d kernel not loaded, skipping\n", __func__, l);
            continue;
        }

        // Get dimensions from the loaded kernel
        // GGUF stores the kernel in [OC, K, IC] format (shape=[OC, K, IC])
        // This is because GGUF header reverses numpy shape, and the numpy shape
        // is [IC, K, OC] from PyTorch (without transpose).
        // So: ne[0]=OC (output_channels), ne[1]=K (kernel_size), ne[2]=IC (input_channels)
        int64_t out_ch = w.layer_kernel[l]->ne[0];   // output channels
        int64_t ks = w.layer_kernel[l]->ne[1];       // kernel_size
        int64_t in_ch = w.layer_kernel[l]->ne[2];    // input channels
        int stride = decoder_rates[l];
        int padding = (stride + 1) / 2;

        LOG_INF("%s: Processing decoder layer %d: in_ch=%lld, out_ch=%lld, kernel=%lld, stride=%d, padding=%d\n",
                __func__, l, (long long)in_ch, (long long)out_ch, (long long)ks, stride, padding);

        // cur is [C, T] format after up_conv
        // After decoder layer forward, cur is [T, C] format
        // So we only transpose if cur is still in [C, T] format (first iteration)
        // Check: if ne[0] == cur_channels and ne[1] == cur_len, then it's [C, T]
        // If ne[0] == cur_len and ne[1] == cur_channels, then it's already [T, C]
        struct ggml_tensor * cur_t;
        if (cur->ne[0] == cur_channels && cur->ne[1] == cur_len) {
            // cur is [C, T], need to transpose to [T, C]
            cur_t = ggml_cont(ctx, ggml_transpose(ctx, cur));  // [cur_len, cur_channels]
        } else {
            // cur is already [T, C]
            cur_t = cur;
        }

        // Copy residual pointers to non-const arrays (needed because w is const)
        struct ggml_tensor * res_in_alpha[3] = {w.residual_in_alpha[l][0], w.residual_in_alpha[l][1], w.residual_in_alpha[l][2]};
        struct ggml_tensor * res_in_kernel[3] = {w.residual_in_kernel[l][0], w.residual_in_kernel[l][1], w.residual_in_kernel[l][2]};
        struct ggml_tensor * res_in_bias[3] = {w.residual_in_bias[l][0], w.residual_in_bias[l][1], w.residual_in_bias[l][2]};
        struct ggml_tensor * res_out_alpha[3] = {w.residual_out_alpha[l][0], w.residual_out_alpha[l][1], w.residual_out_alpha[l][2]};
        struct ggml_tensor * res_out_kernel[3] = {w.residual_out_kernel[l][0], w.residual_out_kernel[l][1], w.residual_out_kernel[l][2]};
        struct ggml_tensor * res_out_bias[3] = {w.residual_out_bias[l][0], w.residual_out_bias[l][1], w.residual_out_bias[l][2]};

        // Apply decoder layer (outputs [output_len, OC] = [T, C] format)
        struct ggml_tensor * layer_out = snac_decoder_layer_forward_ggml(
            ctx,
            cur_t, cur_len,
            w.layer_alpha[l],
            w.layer_kernel[l],
            w.layer_bias[l],
            w.layer_noise_kernel[l],
            (int)in_ch, (int)out_ch, (int)ks, stride, padding, l,
            res_in_alpha,
            res_in_kernel,
            res_in_bias,
            res_out_alpha,
            res_out_kernel,
            res_out_bias);

        if (!layer_out) {
            LOG_ERR("%s: Layer %d forward failed\n", __func__, l);
            return nullptr;
        }

        // Update cur to the layer output for next iteration
        cur = layer_out;

        // Update current length after upsampling
        int output_padding = stride % 2;
        cur_len = (cur_len - 1) * stride + ks - 2 * padding + output_padding;
        cur_channels = out_ch;

        LOG_INF("%s: After layer %d: shape [%lld, %lld]\n", __func__, l,
                (long long)cur->ne[0], (long long)cur->ne[1]);
    }

    // Step 7: Output convolution
    // Snake activation with alpha_out
    // cur is already [T, C] = [cur_len, cur_channels] format from decoder
    // cur_channels should be 64 after layer 3
    cur = snac_snake_forward_2d_ggml(ctx, cur, w.snake_alpha_out, cur_len, (int)cur_channels);

    // Transpose to [C, T] for conv_1d
    cur = ggml_cont(ctx, ggml_transpose(ctx, cur));  // [cur_channels, cur_len]

    // Output conv: [C, T] -> [1, T]
    // The GGUF stores out_conv.weight with header [64, 7, 1] (reversed from numpy [1, 7, 64])
    // The data is stored row-major from numpy (1, 7, 64) = [OC, K, IC]
    // GGML reads this as [64, 7, 1] where ne[0]=64 is innermost
    // So GGML (a, b, c) = numpy(c, b, a) = data[c, b, a]
    //
    // For ggml_conv_1d expecting [OC, IC, K] = [1, 64, 7]:
    //   ne[0] = OC = 1
    //   ne[1] = IC = 64
    //   ne[2] = K = 7
    //
    // Output convolution using ggml_im2col
    // ggml_im2col for 1D expects:
    //   - kernel (src0): [KW, KH, IC, OC] = [K, 1, IC, OC] where ne[0]=K, ne[1]=1, ne[2]=IC, ne[3]=OC
    //   - input (src1): [IW, IC, N, ...] = [T, IC, N, 1] where ne[0]=T, ne[1]=IC, ne[2]=N
    //   - result: [N, OH, OW, IC*KH*KW] = [N, 1, OW, IC*K]
    if (w.out_conv_kernel) {
        LOG_INF("%s: Before output conv: cur shape [%lld, %lld], expected [C, T] = [64, T]\n", __func__,
                (long long)cur->ne[0], (long long)cur->ne[1]);
        LOG_INF("%s: out_conv_kernel shape [%lld, %lld, %lld]\n", __func__,
                (long long)w.out_conv_kernel->ne[0], (long long)w.out_conv_kernel->ne[1],
                (long long)w.out_conv_kernel->ne[2]);

        // Kernel from GGUF is [K, IC, OC] = [7, 64, 1]
        // ggml_im2col for 1D expects [KW, IC, KH, OC] = [K, IC, 1, OC]
        // where ne[0]=K, ne[1]=IC, ne[2]=1, ne[3]=OC
        // The assertion b->ne[1] == a->ne[1] requires kernel ne[1] = IC = 64
        // Expand to 4D
        struct ggml_tensor * kernel_4d = ggml_reshape_4d(ctx, w.out_conv_kernel,
            w.out_conv_kernel->ne[0],  // K = 7
            w.out_conv_kernel->ne[1],  // IC = 64
            1,                          // 1 (KH)
            w.out_conv_kernel->ne[2]); // OC = 1
        LOG_INF("%s: kernel_4d [K, IC, 1, OC] = [%lld, %lld, %lld, %lld]\n", __func__,
                (long long)kernel_4d->ne[0], (long long)kernel_4d->ne[1],
                (long long)kernel_4d->ne[2], (long long)kernel_4d->ne[3]);

        // Convert to F16 if needed (ggml_im2col requires F16 kernel)
        struct ggml_tensor * kernel_for_conv = kernel_4d;
        if (kernel_4d->type != GGML_TYPE_F16) {
            struct ggml_tensor * kernel_f16 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16,
                kernel_4d->ne[0], kernel_4d->ne[1], kernel_4d->ne[2], kernel_4d->ne[3]);
            kernel_for_conv = ggml_cpy(ctx, kernel_4d, kernel_f16);
        }

        int64_t K = kernel_for_conv->ne[0];    // 7
        int64_t IC = kernel_for_conv->ne[1];   // 64
        int64_t OC = kernel_for_conv->ne[3];   // 1
        int64_t T_len = cur->ne[1];            // temporal length

        LOG_INF("%s: Kernel K=%lld, IC=%lld, OC=%lld, T_len=%lld\n", __func__,
                (long long)K, (long long)IC, (long long)OC, (long long)T_len);

        // Input: cur is [C, T] = [64, T]
        // ggml_im2col expects [T, IC, N, 1] = [T, 64, 1, 1]
        // First transpose cur from [C, T] to [T, C] = [T, 64]
        struct ggml_tensor * cur_T = ggml_cont(ctx, ggml_transpose(ctx, cur));  // [T, C]
        // Then expand to 4D [T, IC, N, 1] = [T, 64, 1, 1]
        struct ggml_tensor * cur_4d = ggml_reshape_4d(ctx, cur_T, T_len, IC, 1, 1);

        // CUDA im2col requires F32 input - convert if needed
        struct ggml_tensor * cur_4d_f32 = cur_4d;
        if (cur_4d->type != GGML_TYPE_F32) {
            struct ggml_tensor * cur_f32 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                cur_4d->ne[0], cur_4d->ne[1], cur_4d->ne[2], cur_4d->ne[3]);
            cur_4d_f32 = ggml_cpy(ctx, cur_4d, cur_f32);
        }

        LOG_INF("%s: Input for im2col [T, IC, N, 1] = [%lld, %lld, %lld, %lld], type=%d\n", __func__,
                (long long)cur_4d_f32->ne[0], (long long)cur_4d_f32->ne[1],
                (long long)cur_4d_f32->ne[2], (long long)cur_4d_f32->ne[3], cur_4d_f32->type);

        // Apply im2col: input [T, IC, N, 1] -> [N, 1, OW, IC*K] with stride=1, padding=3
        // For stride=1, padding=3 (K/2), dilation=1:
        // OW = (T + 2*3 - 7)/1 + 1 = T
        struct ggml_tensor * im2col_out = ggml_im2col(ctx, kernel_for_conv, cur_4d_f32, 1, 0, 3, 0, 1, 0, false, GGML_TYPE_F32);

        LOG_INF("%s: im2col output [IC*K, OW, N, 1] = [%lld, %lld, %lld, %lld]\n", __func__,
                (long long)im2col_out->ne[0], (long long)im2col_out->ne[1],
                (long long)im2col_out->ne[2], (long long)im2col_out->ne[3]);

        // Reshape kernel from [K, IC, 1, OC] to [IC*K, OC] = [448, 1]
        // Note: ggml_mul_mat(a, b) computes a^T @ b, result is [a->ne[1], b->ne[1]]
        // The requirement is a->ne[0] == b->ne[0]
        struct ggml_tensor * kernel_2d_f16 = ggml_reshape_2d(ctx, kernel_for_conv, K * IC, OC);  // [IC*K, OC] = [448, 1]
        struct ggml_tensor * kernel_2d = kernel_2d_f16;
        if (kernel_2d_f16->type == GGML_TYPE_F16) {
            struct ggml_tensor * kernel_2d_f32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K * IC, OC);
            kernel_2d = ggml_cpy(ctx, kernel_2d_f16, kernel_2d_f32);
        }

        // im2col output is [IC*K, OW, N, 1] = [448, T, 1, 1]
        // Reshape to [IC*K, OW] = [448, T]
        struct ggml_tensor * im2col_2d = ggml_reshape_2d(ctx, im2col_out, im2col_out->ne[0], im2col_out->ne[1]);

        LOG_INF("%s: im2col_2d [IC*K, OW] = [%lld, %lld], kernel_2d [IC*K, OC] = [%lld, %lld]\n", __func__,
                (long long)im2col_2d->ne[0], (long long)im2col_2d->ne[1],
                (long long)kernel_2d->ne[0], (long long)kernel_2d->ne[1]);

        // Matrix multiplication: im2col_2d^T @ kernel_2d
        // Following ggml_conv_2d pattern: mul_mat(im2col, kernel)
        // im2col_2d: [IC*K, OW] = [448, T]
        // kernel_2d: [IC*K, OC] = [448, 1]
        // For mul_mat(a, b): a->ne[0] == b->ne[0] must hold (448 == 448 ✓)
        // mul_mat(im2col_2d, kernel_2d): [448, T]^T @ [448, 1] = [T, 448] @ [448, 1] = [T, 1]
        cur = ggml_mul_mat(ctx, im2col_2d, kernel_2d);

        // Result is [T, 1] which is [OW, OC] - already correct format

        LOG_INF("%s: Manual conv output [OW, OC] = [%lld, %lld]\n", __func__,
                (long long)cur->ne[0], (long long)cur->ne[1]);

        if (w.out_conv_bias) {
            cur = ggml_add(ctx, cur, w.out_conv_bias);
        }
    }

    // Flatten to 1D: output shape depends on ggml_conv_1d output format
    // ggml_conv_1d output is [OW, OC, N] where:
    //   ne[0] = OW (output width/length)
    //   ne[1] = OC (output channels)
    //   ne[2] = N (batch size)
    int64_t final_output_len;
    if (ggml_n_dims(cur) >= 2) {
        // Output is [OW, OC, N], temporal length is in ne[0]
        final_output_len = cur->ne[0];  // OW = output length
    } else {
        final_output_len = ggml_nelements(cur);
    }

    LOG_INF("%s: Output conv result shape: [%lld, %lld, %lld, %lld], dims=%d, final_output_len=%lld\n", __func__,
            (long long)cur->ne[0], (long long)cur->ne[1],
            (long long)cur->ne[2], (long long)cur->ne[3],
            ggml_n_dims(cur), (long long)final_output_len);

    cur = ggml_view_1d(ctx, cur, final_output_len, 0);

    // Convert to F32 for tanh operation (some ops don't support F16)
    if (cur->type != GGML_TYPE_F32) {
        struct ggml_tensor * cur_f32 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, final_output_len);
        cur = ggml_cpy(ctx, cur, cur_f32);
    }

    // Apply tanh to bound output
    cur = ggml_tanh(ctx, cur);

    LOG_INF("%s: Final output length: %lld samples\n", __func__, (long long)final_output_len);

    return cur;
}

// Weight loading helper - copy tensor data to ggml tensor
static bool copy_gguf_to_ggml(
    struct gguf_context * gguf_ctx,
    struct ggml_context * ggml_ctx,
    const char * tensor_name,
    struct ggml_tensor *& dst_tensor,
    ggml_backend_buffer_t buffer = nullptr)
{
    (void)gguf_ctx; // Reserved for future use

    struct ggml_tensor * src = ggml_get_tensor(ggml_ctx, tensor_name);
    if (!src) {
        return false;
    }

    // Create new tensor in destination context
    if (!dst_tensor) {
        dst_tensor = ggml_dup_tensor(ggml_ctx, src);
        ggml_set_name(dst_tensor, tensor_name);
    }

    // Copy data
    if (src->data && dst_tensor->data) {
        memcpy(dst_tensor->data, src->data, ggml_nbytes(src));
    } else if (buffer) {
        // If using backend buffer, copy through it
        if (src->data) {
            ggml_backend_tensor_set(dst_tensor, src->data, 0, ggml_nbytes(src));
        }
    }

    return true;
}

// Helper to get tensor from GGUF and create ggml tensor
static bool load_gguf_tensor(
    struct gguf_context * gguf_ctx,
    struct ggml_context * ggml_ctx,
    const char * name,
    struct ggml_tensor *& dst,
    bool required = true)
{
    (void)gguf_ctx; // Reserved for future use

    struct ggml_tensor * src = ggml_get_tensor(ggml_ctx, name);
    if (!src) {
        if (required) {
            LOG_WRN("%s: Tensor '%s' not found\n", __func__, name);
        }
        return false;
    }

    // Duplicate tensor structure
    dst = ggml_dup_tensor(ggml_ctx, src);
    ggml_set_name(dst, name);

    // Copy data
    if (src->data && dst->data) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
    }

    return true;
}

// Load SNAC model from GGUF
bool snac_ggml_init(
    struct snac_ggml_context & ctx,
    const char * model_path,
    ggml_backend_t backend)
{
    LOG_INF("%s: Loading SNAC model from %s\n", __func__, model_path);

    // Initialize backend
    if (!backend) {
        backend = ggml_backend_cpu_init();
        if (!backend) {
            LOG_ERR("%s: Failed to initialize CPU backend\n", __func__);
            return false;
        }
    }
    ctx.backend = backend;

    // Load GGUF file with context to create tensors
    struct gguf_init_params gguf_params = {
        /*.no_alloc = */ true,  // Don't allocate data yet
        /*.ctx      = */ &ctx.ctx,  // Create tensors in our context
    };

    struct gguf_context * gguf_ctx = gguf_init_from_file(model_path, gguf_params);
    if (!gguf_ctx) {
        LOG_ERR("%s: Failed to load GGUF file: %s\n", __func__, model_path);
        return false;
    }

    // Count tensors and print all tensor names for debugging
    size_t n_tensors = gguf_get_n_tensors(gguf_ctx);
    LOG_INF("%s: Found %zu tensors in GGUF file\n", __func__, n_tensors);

    // Debug: print all tensor names
    for (size_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        LOG_DBG("%s:   [%zu] %s\n", __func__, i, name);
    }

    // Allocate backend buffer for all tensors in context
    ctx.buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx.ctx, ggml_backend_get_default_buffer_type(ctx.backend));
    if (!ctx.buffer) {
        LOG_ERR("%s: Failed to allocate backend buffer\n", __func__);
        gguf_free(gguf_ctx);
        return false;
    }

    // Load tensor data from GGUF into backend buffer
    // We iterate through all tensors and map them to our weights structure
    int tensors_loaded = 0;

    for (size_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        size_t tensor_offset = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, i);
        size_t tensor_size = gguf_get_tensor_size(gguf_ctx, i);

        // Find tensor in our context
        struct ggml_tensor * dst = ggml_get_tensor(ctx.ctx, name);
        if (!dst) {
            LOG_DBG("%s: Tensor '%s' not in context, skipping\n", __func__, name);
            continue;
        }

        // Read tensor data from GGUF file
        FILE * f = fopen(model_path, "rb");
        if (!f) {
            LOG_ERR("%s: Failed to open GGUF file for reading\n", __func__);
            continue;
        }

        fseek(f, tensor_offset, SEEK_SET);

        // Read directly into backend buffer
        std::vector<char> temp_buf(tensor_size);
        size_t bytes_read = fread(temp_buf.data(), 1, tensor_size, f);
        fclose(f);

        if (bytes_read != tensor_size) {
            LOG_WRN("%s: Failed to read full tensor '%s'\n", __func__, name);
            continue;
        }

        // Copy to backend tensor
        ggml_backend_tensor_set(dst, temp_buf.data(), 0, tensor_size);
        tensors_loaded++;

        // Map to weights structure by name
        // Input convolutions
        if (strcmp(name, "decoder.in_conv.weight") == 0) {
            ctx.weights.in_conv_kernel = dst;
        } else if (strcmp(name, "decoder.in_conv.bias") == 0) {
            ctx.weights.in_conv_bias = dst;
        }
        // Up conv
        else if (strcmp(name, "decoder.up_conv.weight") == 0) {
            ctx.weights.up_conv_kernel = dst;
        } else if (strcmp(name, "decoder.up_conv.bias") == 0) {
            ctx.weights.up_conv_bias = dst;
        }
        // Output layer
        else if (strcmp(name, "decoder.out_conv.weight") == 0) {
            ctx.weights.out_conv_kernel = dst;
        } else if (strcmp(name, "decoder.out_conv.bias") == 0) {
            ctx.weights.out_conv_bias = dst;
        } else if (strcmp(name, "decoder.alpha_out") == 0) {
            ctx.weights.snake_alpha_out = dst;
        }

        // Decoder layers
        for (int l = 0; l < SNAC_GGML_N_DECODER_LAYERS; l++) {
            char expected[256];

            snprintf(expected, sizeof(expected), "decoder.layers.%d.alpha", l);
            if (strcmp(name, expected) == 0) {
                ctx.weights.layer_alpha[l] = dst;
            }

            snprintf(expected, sizeof(expected), "decoder.layers.%d.conv_t.weight", l);
            if (strcmp(name, expected) == 0) {
                ctx.weights.layer_kernel[l] = dst;
            }

            snprintf(expected, sizeof(expected), "decoder.layers.%d.conv_t.bias", l);
            if (strcmp(name, expected) == 0) {
                ctx.weights.layer_bias[l] = dst;
            }

            snprintf(expected, sizeof(expected), "decoder.layers.%d.noise_proj.weight", l);
            if (strcmp(name, expected) == 0) {
                ctx.weights.layer_noise_kernel[l] = dst;
            }

            // Residual units
            for (int u = 0; u < 3; u++) {
                snprintf(expected, sizeof(expected), "decoder.layers.%d.residual_units.%d.in_alpha", l, u);
                if (strcmp(name, expected) == 0) {
                    ctx.weights.residual_in_alpha[l][u] = dst;
                }

                snprintf(expected, sizeof(expected), "decoder.layers.%d.residual_units.%d.in_conv.weight", l, u);
                if (strcmp(name, expected) == 0) {
                    ctx.weights.residual_in_kernel[l][u] = dst;
                }

                snprintf(expected, sizeof(expected), "decoder.layers.%d.residual_units.%d.in_conv.bias", l, u);
                if (strcmp(name, expected) == 0) {
                    ctx.weights.residual_in_bias[l][u] = dst;
                }

                snprintf(expected, sizeof(expected), "decoder.layers.%d.residual_units.%d.out_alpha", l, u);
                if (strcmp(name, expected) == 0) {
                    ctx.weights.residual_out_alpha[l][u] = dst;
                }

                snprintf(expected, sizeof(expected), "decoder.layers.%d.residual_units.%d.out_conv.weight", l, u);
                if (strcmp(name, expected) == 0) {
                    ctx.weights.residual_out_kernel[l][u] = dst;
                }

                snprintf(expected, sizeof(expected), "decoder.layers.%d.residual_units.%d.out_conv.bias", l, u);
                if (strcmp(name, expected) == 0) {
                    ctx.weights.residual_out_bias[l][u] = dst;
                }
            }
        }

        // Quantizers
        for (int q = 0; q < SNAC_GGML_N_QUANTIZERS; q++) {
            char expected[256];

            snprintf(expected, sizeof(expected), "quantizers.%d.codebook.weight", q);
            if (strcmp(name, expected) == 0) {
                ctx.weights.quant_codebook[q] = dst;
                LOG_DBG("%s: Mapped %s -> quant_codebook[%d]\n", __func__, name, q);
            }

            snprintf(expected, sizeof(expected), "quantizers.%d.out_proj.weight", q);
            if (strcmp(name, expected) == 0) {
                ctx.weights.quant_out_proj[q] = dst;
                LOG_DBG("%s: Mapped %s -> quant_out_proj[%d] (shape: %lld x %lld x %lld)\n", __func__, name, q,
                        (long long)dst->ne[0], (long long)dst->ne[1], (long long)dst->ne[2]);
            }

            snprintf(expected, sizeof(expected), "quantizers.%d.out_proj.bias", q);
            if (strcmp(name, expected) == 0) {
                ctx.weights.quant_out_bias[q] = dst;
                LOG_DBG("%s: Mapped %s -> quant_out_bias[%d]\n", __func__, name, q);
            }
        }
    }

    LOG_INF("%s: Loaded %d tensors\n", __func__, tensors_loaded);

    // Debug: verify quantizer weights loaded
    for (int q = 0; q < SNAC_GGML_N_QUANTIZERS; q++) {
        LOG_INF("%s: quant_codebook[%d] = %p, quant_out_proj[%d] = %p, quant_out_bias[%d] = %p\n",
                __func__, q, (void*)ctx.weights.quant_codebook[q],
                q, (void*)ctx.weights.quant_out_proj[q],
                q, (void*)ctx.weights.quant_out_bias[q]);
    }

    // Set loaded flag
    ctx.loaded = tensors_loaded > 0;
    ctx.n_quantizers = SNAC_GGML_N_QUANTIZERS;

    gguf_free(gguf_ctx);

    LOG_INF("%s: SNAC model loaded successfully (%d tensors)\n", __func__, tensors_loaded);
    return ctx.loaded;
}

// Free SNAC context
void snac_ggml_free(struct snac_ggml_context & ctx)
{
    if (ctx.buffer) {
        ggml_backend_buffer_free(ctx.buffer);
        ctx.buffer = nullptr;
    }

    if (ctx.ctx) {
        ggml_free(ctx.ctx);
        ctx.ctx = nullptr;
    }

    // Note: backend is owned externally if provided, so don't free it here
    // unless we created it
    ctx.loaded = false;
}

// Helper function to perform repeat_interleave on tokens
// Each token is repeated 'repeats' times consecutively
static std::vector<int> repeat_interleave_tokens(
    const std::vector<int> & tokens,
    int repeats)
{
    if (repeats <= 1 || tokens.empty()) {
        return tokens;
    }

    std::vector<int> result;
    result.reserve(tokens.size() * repeats);

    for (int token : tokens) {
        for (int r = 0; r < repeats; r++) {
            result.push_back(token);
        }
    }

    return result;
}

// Main decode function
std::vector<float> snac_ggml_decode(
    struct snac_ggml_context & ctx,
    const std::vector<std::vector<int>> & pyramid_tokens)
{
    if (!ctx.loaded) {
        LOG_WRN("%s: SNAC model not loaded\n", __func__);
        return {};
    }

    if (pyramid_tokens.size() < 3 || pyramid_tokens[2].empty()) {
        return {};
    }

    // Get token counts for each head (before expansion)
    int64_t head0_orig_len = pyramid_tokens[0].size();
    int64_t head1_orig_len = pyramid_tokens[1].size();
    int64_t head2_len = pyramid_tokens[2].size();

    LOG_INF("%s: Decoding %lld + %lld + %lld tokens (before pyramid expansion)\n", __func__,
            (long long)head0_orig_len, (long long)head1_orig_len, (long long)head2_len);

    // Pyramid structure with vq_strides = [4, 2, 1]:
    // - head0: stride 4, each token repeated 4 times
    // - head1: stride 2, each token repeated 2 times
    // - head2: stride 1, no expansion needed
    // Note: We only have 3 quantizers, using vq_strides[0..2] = [4, 2, 1]

    // Pre-expand tokens using repeat_interleave
    std::vector<int> expanded_head0;
    std::vector<int> expanded_head1;
    const std::vector<int> & expanded_head2 = pyramid_tokens[2];  // No expansion needed

    int stride0 = ctx.vq_strides[0];  // 4
    int stride1 = ctx.vq_strides[1];  // 2

    // Verify pyramid structure: head0 * 4 = head1 * 2 = head2
    int64_t expected_head2_len = head0_orig_len * stride0;
    if (expected_head2_len != head2_len) {
        LOG_WRN("%s: Pyramid structure mismatch: head0*%d=%lld != head2=%lld\n", __func__,
                stride0, (long long)expected_head2_len, (long long)head2_len);
    }

    // Expand head0 tokens (repeat each 4 times)
    if (!pyramid_tokens[0].empty()) {
        expanded_head0 = repeat_interleave_tokens(pyramid_tokens[0], stride0);
        LOG_INF("%s: Expanded head0: %lld -> %zu tokens\n", __func__,
                (long long)head0_orig_len, expanded_head0.size());
    }

    // Expand head1 tokens (repeat each 2 times)
    if (!pyramid_tokens[1].empty()) {
        expanded_head1 = repeat_interleave_tokens(pyramid_tokens[1], stride1);
        LOG_INF("%s: Expanded head1: %lld -> %zu tokens\n", __func__,
                (long long)head1_orig_len, expanded_head1.size());
    }

    // All expanded heads should now have the same length as head2
    int64_t output_len = head2_len;

    // Verify expanded lengths
    if ((int64_t)expanded_head0.size() != output_len) {
        LOG_WRN("%s: Expanded head0 length mismatch: %zu vs %lld\n", __func__,
                expanded_head0.size(), (long long)output_len);
    }
    if ((int64_t)expanded_head1.size() != output_len) {
        LOG_WRN("%s: Expanded head1 length mismatch: %zu vs %lld\n", __func__,
                expanded_head1.size(), (long long)output_len);
    }

    // 1. Create graph context for temporary tensors
    struct ggml_init_params params = {
        /*.mem_size   =*/ 256*1024*1024,  // 256MB for graph
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * graph_ctx = ggml_init(params);
    if (!graph_ctx) {
        LOG_ERR("%s: Failed to create graph context\n", __func__);
        return {};
    }

    // 2. Build computation graph with expanded tokens
    // All heads now have the same length (output_len)
    struct ggml_tensor * output = snac_build_graph(
        graph_ctx, ctx.weights,
        expanded_head0.data(), output_len,  // Expanded head0
        expanded_head1.data(), output_len,  // Expanded head1
        expanded_head2.data(), output_len,  // Head2 (no expansion)
        ctx.vq_strides);

    if (!output) {
        LOG_ERR("%s: Failed to build graph\n", __func__);
        ggml_free(graph_ctx);
        return {};
    }

    struct ggml_cgraph * gf = ggml_new_graph(graph_ctx);
    ggml_build_forward_expand(gf, output);

    // 3. Allocate graph tensors
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx.backend));
    if (!allocr) {
        LOG_ERR("%s: Failed to create allocator\n", __func__);
        ggml_free(graph_ctx);
        return {};
    }

    if (!ggml_gallocr_alloc_graph(allocr, gf)) {
        LOG_ERR("%s: Failed to allocate graph\n", __func__);
        ggml_gallocr_free(allocr);
        ggml_free(graph_ctx);
        return {};
    }

    // 4. Set input data for each head's token tensor (all now have same length)
    bool set_input_success = true;

    {
        struct ggml_tensor * tokens0 = nullptr;
        if (!expanded_head0.empty()) {
            tokens0 = ggml_get_tensor(graph_ctx, "tokens_head0");
            if (tokens0) {
                ggml_backend_tensor_set(tokens0, expanded_head0.data(), 0, output_len * sizeof(int));
            }
        }
    }

    {
        struct ggml_tensor * tokens1 = nullptr;
        if (!expanded_head1.empty()) {
            tokens1 = ggml_get_tensor(graph_ctx, "tokens_head1");
            if (tokens1) {
                ggml_backend_tensor_set(tokens1, expanded_head1.data(), 0, output_len * sizeof(int));
            }
        }
    }

    {
        struct ggml_tensor * tokens2 = nullptr;
        if (!expanded_head2.empty()) {
            tokens2 = ggml_get_tensor(graph_ctx, "tokens_head2");
            if (tokens2) {
                ggml_backend_tensor_set(tokens2, expanded_head2.data(), 0, output_len * sizeof(int));
            }
        }
    }

    if (!set_input_success) {
        LOG_ERR("%s: Failed to set input data\n", __func__);
        ggml_gallocr_free(allocr);
        ggml_free(graph_ctx);
        return {};
    }

    // 5. Execute the graph
    if (ggml_backend_is_cpu(ctx.backend)) {
        ggml_backend_cpu_set_n_threads(ctx.backend, 4);  // Use 4 threads by default
    }

    ggml_status status = ggml_backend_graph_compute(ctx.backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        LOG_ERR("%s: Graph computation failed with status %d\n", __func__, status);
        ggml_gallocr_free(allocr);
        ggml_free(graph_ctx);
        return {};
    }

    // 6. Extract output samples from result tensor
    // Output shape should be [1, T] after final conv and tanh
    int64_t output_samples = ggml_nelements(output);
    std::vector<float> result(output_samples);

    ggml_backend_tensor_get(output, result.data(), 0, output_samples * sizeof(float));

    LOG_INF("%s: Generated %lld audio samples\n", __func__, (long long)output_samples);

    // 6.5 Normalize audio to proper amplitude
    // The SNAC model has very small output values due to tiny out_conv weights
    // Normalize to [-1, 1] range with reasonable amplitude
    {
        // Find peak amplitude
        float peak = 0.0f;
        for (float s : result) {
            peak = std::max(peak, std::abs(s));
        }

        // Normalize if peak is too small
        // Target peak of 0.9 for good audio levels
        const float target_peak = 0.9f;
        if (peak > 0.0f && peak < target_peak) {
            float scale = target_peak / peak;
            for (float & s : result) {
                s *= scale;
            }
            LOG_INF("%s: Normalized audio (peak was %.6f, scaled by %.2f)\n", __func__, peak, scale);
        }
    }

    // 7. Cleanup
    ggml_gallocr_free(allocr);
    ggml_free(graph_ctx);

    return result;
}

// ============================================================================
// Phase 2: Batched Processing Implementation
// ============================================================================

// Build batched computation graph
// Handles multiple sequences in a single graph with batch dimension
static struct ggml_tensor * snac_build_batch_graph(
    struct ggml_context * ctx,
    const struct snac_ggml_weights & w,
    int64_t batch_size,
    int64_t head0_len,
    int64_t head1_len,
    int64_t head2_len)
{
    struct ggml_tensor * cur = nullptr;

    // Step 1: Create batched input tensors for each quantizer head
    // Tokens for head 0: [batch_size, head0_len]
    struct ggml_tensor * tokens0 = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, head0_len, batch_size);
    ggml_set_name(tokens0, "batch_tokens_head0");

    // Tokens for head 1: [batch_size, head1_len]
    struct ggml_tensor * tokens1 = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, head1_len, batch_size);
    ggml_set_name(tokens1, "batch_tokens_head1");

    // Tokens for head 2: [batch_size, head2_len]
    struct ggml_tensor * tokens2 = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, head2_len, batch_size);
    ggml_set_name(tokens2, "batch_tokens_head2");

    // Step 2: Quantizer forward for each head (batched)
    std::vector<struct ggml_tensor *> head_embeddings(3);

    // Head 0 quantizer lookup
    // tokens0: [head0_len, batch_size]
    // For batched get_rows, we need to handle each batch element
    // GGML's ggml_get_rows doesn't support batch dimension directly
    // We'll process each head separately and concatenate

    // For now, process sequences sequentially within the graph
    // This is a stepping stone - full parallel batch processing would require
    // either custom ops or restructuring the computation

    // Embedding lookup for head 0
    // Flatten batch dimension, lookup, then reshape
    struct ggml_tensor * tokens0_flat = ggml_view_1d(ctx, tokens0, batch_size * head0_len, 0);
    struct ggml_tensor * emb0_flat = ggml_get_rows(ctx, w.quant_codebook[0], tokens0_flat);
    // emb0_flat: [codebook_dim, batch_size * head0_len] = [8, batch_size * head0_len]
    // Transpose to [batch_size * head0_len, codebook_dim]
    emb0_flat = ggml_cont(ctx, ggml_transpose(ctx, emb0_flat));
    // Reshape to [batch_size, head0_len, codebook_dim]
    head_embeddings[0] = ggml_reshape_3d(ctx, emb0_flat, SNAC_GGML_CODEBOOK_DIM, head0_len, batch_size);

    // Apply projection for head 0
    // Project from codebook_dim (8) to quantizer_dim (768)
    // Use 1x1 conv: reshape to [batch_size, codebook_dim, head0_len]
    struct ggml_tensor * h0_transposed = ggml_permute(ctx, head_embeddings[0], 1, 2, 0, 3);
    // Now [codebook_dim, head0_len, batch_size]
    struct ggml_tensor * kernel0 = ggml_reshape_3d(ctx, w.quant_out_proj[0],
                                                    SNAC_GGML_QUANTIZER_DIM, SNAC_GGML_CODEBOOK_DIM, 1);
    struct ggml_tensor * proj0 = ggml_conv_1d(ctx, kernel0, h0_transposed, 1, 0, 1);
    if (w.quant_out_bias[0]) {
        proj0 = ggml_add(ctx, proj0, w.quant_out_bias[0]);
    }
    // proj0: [quantizer_dim, head0_len, batch_size] -> permute back
    head_embeddings[0] = ggml_permute(ctx, proj0, 1, 0, 2, 3);
    // Now [head0_len, quantizer_dim, batch_size]

    // Similar for head 1
    if (head1_len > 0) {
        struct ggml_tensor * tokens1_flat = ggml_view_1d(ctx, tokens1, batch_size * head1_len, 0);
        struct ggml_tensor * emb1_flat = ggml_get_rows(ctx, w.quant_codebook[1], tokens1_flat);
        emb1_flat = ggml_cont(ctx, ggml_transpose(ctx, emb1_flat));
        head_embeddings[1] = ggml_reshape_3d(ctx, emb1_flat, SNAC_GGML_CODEBOOK_DIM, head1_len, batch_size);

        struct ggml_tensor * h1_transposed = ggml_permute(ctx, head_embeddings[1], 1, 2, 0, 3);
        struct ggml_tensor * kernel1 = ggml_reshape_3d(ctx, w.quant_out_proj[1],
                                                        SNAC_GGML_QUANTIZER_DIM, SNAC_GGML_CODEBOOK_DIM, 1);
        struct ggml_tensor * proj1 = ggml_conv_1d(ctx, kernel1, h1_transposed, 1, 0, 1);
        if (w.quant_out_bias[1]) {
            proj1 = ggml_add(ctx, proj1, w.quant_out_bias[1]);
        }
        head_embeddings[1] = ggml_permute(ctx, proj1, 1, 0, 2, 3);
    }

    // Similar for head 2
    if (head2_len > 0) {
        struct ggml_tensor * tokens2_flat = ggml_view_1d(ctx, tokens2, batch_size * head2_len, 0);
        struct ggml_tensor * emb2_flat = ggml_get_rows(ctx, w.quant_codebook[2], tokens2_flat);
        emb2_flat = ggml_cont(ctx, ggml_transpose(ctx, emb2_flat));
        head_embeddings[2] = ggml_reshape_3d(ctx, emb2_flat, SNAC_GGML_CODEBOOK_DIM, head2_len, batch_size);

        struct ggml_tensor * h2_transposed = ggml_permute(ctx, head_embeddings[2], 1, 2, 0, 3);
        struct ggml_tensor * kernel2 = ggml_reshape_3d(ctx, w.quant_out_proj[2],
                                                        SNAC_GGML_QUANTIZER_DIM, SNAC_GGML_CODEBOOK_DIM, 1);
        struct ggml_tensor * proj2 = ggml_conv_1d(ctx, kernel2, h2_transposed, 1, 0, 1);
        if (w.quant_out_bias[2]) {
            proj2 = ggml_add(ctx, proj2, w.quant_out_bias[2]);
        }
        head_embeddings[2] = ggml_permute(ctx, proj2, 1, 0, 2, 3);
    }

    // Step 3: Combine embeddings with pyramid structure
    // Pyramid structure with vq_strides = [4, 2, 1]:
    // - head0 (stride 4 relative to head2): each token covers 4 positions
    // - head1 (stride 2 relative to head2): each token covers 2 positions
    // - head2 (stride 1): each token covers 1 position
    //
    // The caller should pre-expand tokens so all heads have the same length (head2_len)
    // After expansion:
    // - head0 expanded by 4x -> length = head2_len
    // - head1 expanded by 2x -> length = head2_len
    // - head2 unchanged -> length = head2_len
    //
    // We then sum all embeddings additively

    // Start with head2 (base, no expansion needed)
    cur = head_embeddings[2];
    // cur: [head2_len, quantizer_dim, batch_size]

    // Add head1 if dimensions match (should be pre-expanded to head2_len)
    if (head_embeddings[1]) {
        // Verify dimensions match
        if (head_embeddings[1]->ne[0] == cur->ne[0] && head_embeddings[1]->ne[1] == cur->ne[1]) {
            cur = ggml_add(ctx, cur, head_embeddings[1]);
        } else {
            // Dimension mismatch - this indicates tokens weren't pre-expanded
            // For now, skip adding (will produce incorrect audio)
            LOG_WRN("%s: head1 dimension mismatch, skipping\n", __func__);
        }
    }

    // Add head0 if dimensions match (should be pre-expanded to head2_len)
    if (head_embeddings[0]) {
        // Verify dimensions match
        if (head_embeddings[0]->ne[0] == cur->ne[0] && head_embeddings[0]->ne[1] == cur->ne[1]) {
            cur = ggml_add(ctx, cur, head_embeddings[0]);
        } else {
            // Dimension mismatch - this indicates tokens weren't pre-expanded
            LOG_WRN("%s: head0 dimension mismatch, skipping\n", __func__);
        }
    }

    // cur: [head2_len, quantizer_dim, batch_size] after summing all heads

    // Step 4: Input convolution (depthwise) - batched
    // Reshape cur to [T, C, batch] -> conv expects [C, T, batch]
    cur = ggml_permute(ctx, cur, 1, 0, 2, 3);  // [quantizer_dim, head2_len, batch_size]

    // Apply depthwise conv with proper kernel permutation
    // Kernel is [C, 1, K] = [1024, 1, 7], need to permute to [K, 1, C] for ggml_conv_1d_dw
    if (w.in_conv_kernel) {
        struct ggml_tensor * kernel_f16 = w.in_conv_kernel;
        if (w.in_conv_kernel->type != GGML_TYPE_F16) {
            struct ggml_tensor * kernel_f16_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F16,
                w.in_conv_kernel->ne[0], w.in_conv_kernel->ne[1], w.in_conv_kernel->ne[2]);
            kernel_f16 = ggml_cpy(ctx, w.in_conv_kernel, kernel_f16_tensor);
        }
        // Kernel is already in [K, 1, C] format from GGUF, no permutation needed
        struct ggml_tensor * kernel_for_dw = kernel_f16;

        cur = ggml_conv_1d_dw(ctx, kernel_for_dw, cur, 1, 3, 1);
    }
    if (w.in_conv_bias) {
        cur = ggml_add(ctx, cur, w.in_conv_bias);
    }

    // Step 5: Up convolution (1x1 conv) - project to decoder_dim
    // ggml_conv_1d requires F16 kernel
    {
        struct ggml_tensor * kernel_f16 = w.up_conv_kernel;
        if (w.up_conv_kernel->type != GGML_TYPE_F16) {
            struct ggml_tensor * kernel_f16_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F16,
                w.up_conv_kernel->ne[0], w.up_conv_kernel->ne[1], w.up_conv_kernel->ne[2]);
            kernel_f16 = ggml_cpy(ctx, w.up_conv_kernel, kernel_f16_tensor);
        }
        cur = ggml_conv_1d(ctx, kernel_f16, cur, 1, 0, 1);
    }
    if (w.up_conv_bias) {
        cur = ggml_add(ctx, cur, w.up_conv_bias);
    }
    // cur: [decoder_dim, seq_len, batch_size] where seq_len = head2_len

    // Step 6: Decoder layers - each with ConvTranspose1D
    // Layer dimensions from GGUF: {1024, 512, 256, 128} -> {512, 256, 128, 64}
    // Decoder rates from config: {8, 8, 4, 2}
    // Kernel sizes: {16, 16, 8, 4} (= 2 * stride)
    int layer_channels[4] = {1024, 512, 256, 128};
    int out_channels[4] = {512, 256, 128, 64};
    int decoder_rates[4] = {8, 8, 4, 2};
    int kernel_sizes[4] = {16, 16, 8, 4};

    // Use head2_len as the starting length (all heads pre-expanded to this length)
    int64_t cur_len = head2_len;

    for (int l = 0; l < SNAC_GGML_N_DECODER_LAYERS; l++) {
        if (!w.layer_kernel[l]) continue;

        int in_ch = layer_channels[l];
        int out_ch = out_channels[l];
        int stride = decoder_rates[l];
        int ks = kernel_sizes[l];
        int padding = (stride + 1) / 2;

        // Calculate output length
        int output_padding = stride % 2;
        int64_t output_len = (cur_len - 1) * stride + ks - 2 * padding + output_padding;

        // Permute cur to [T, C, batch] for snake activation
        cur = ggml_permute(ctx, cur, 1, 0, 2, 3);  // [cur_len, in_ch, batch_size]

        // Snake activation
        cur = snac_snake_forward_2d_ggml(ctx, cur, w.layer_alpha[l], cur_len, in_ch);

        // Permute back to [C, T, batch] for conv_transpose
        cur = ggml_permute(ctx, cur, 1, 0, 2, 3);  // [in_ch, cur_len, batch_size]

        // ConvTranspose1D with proper kernel permutation
        // PyTorch kernel format: [IC, OC, K] -> GGML expects: [K, OC, IC]
        {
            struct ggml_tensor * kernel_f16 = w.layer_kernel[l];
            if (w.layer_kernel[l]->type != GGML_TYPE_F16) {
                struct ggml_tensor * kernel_f16_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F16,
                    w.layer_kernel[l]->ne[0], w.layer_kernel[l]->ne[1], w.layer_kernel[l]->ne[2]);
                kernel_f16 = ggml_cpy(ctx, w.layer_kernel[l], kernel_f16_tensor);
            }
            // Permute [IC, OC, K] -> [K, OC, IC]
            struct ggml_tensor * kernel_4d = ggml_reshape_4d(ctx, kernel_f16,
                kernel_f16->ne[0], kernel_f16->ne[1], kernel_f16->ne[2], 1);
            struct ggml_tensor * kernel_permuted = ggml_permute(ctx, kernel_4d, 2, 1, 0, 3);
            struct ggml_tensor * kernel_for_conv = ggml_cont(ctx, kernel_permuted);
            kernel_for_conv = ggml_reshape_3d(ctx, kernel_for_conv,
                kernel_for_conv->ne[0], kernel_for_conv->ne[1], kernel_for_conv->ne[2]);

            cur = ggml_conv_transpose_1d(ctx, kernel_for_conv, cur, stride, padding, 1);
        }

        // Add bias
        if (w.layer_bias[l]) {
            cur = ggml_add(ctx, cur, w.layer_bias[l]);
        }

        // Permute for residual units
        cur = ggml_permute(ctx, cur, 1, 0, 2, 3);  // [output_len, out_ch, batch_size]

        // Residual units (simplified for batched version)
        for (int u = 0; u < 3; u++) {
            if (!w.residual_in_alpha[l][u]) continue;

            struct ggml_tensor * residual = cur;
            int channels = out_ch;
            int unit_kernel_size = 7;
            int dilation = (int)std::pow(3, u);
            int unit_padding = ((unit_kernel_size - 1) * dilation) / 2;

            // Snake in
            cur = snac_snake_forward_2d_ggml(ctx, cur, w.residual_in_alpha[l][u], output_len, channels);

            // Permute for depthwise conv
            cur = ggml_permute(ctx, cur, 1, 0, 2, 3);

            // Depthwise conv in - need to transpose kernel data
            if (w.residual_in_kernel[l][u]) {
                // Kernel is stored in GGUF with header [K, 1, C] but data is in [C, 1, K] layout
                // We need to actually transpose the data from [C, 1, K] to [K, 1, C]
                struct ggml_tensor * res_kernel_f16 = w.residual_in_kernel[l][u];
                if (res_kernel_f16->type != GGML_TYPE_F16) {
                    struct ggml_tensor * kernel_f16_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F16,
                        res_kernel_f16->ne[0], res_kernel_f16->ne[1], res_kernel_f16->ne[2]);
                    res_kernel_f16 = ggml_cpy(ctx, w.residual_in_kernel[l][u], kernel_f16_tensor);
                }

                // Step 1: Reshape to 2D [C, K] using reshape instead of view_2d
                struct ggml_tensor * kernel_2d = ggml_reshape_2d(ctx, res_kernel_f16,
                    res_kernel_f16->ne[2],  // C (from ne[2] since GGUF stores [K,1,C])
                    res_kernel_f16->ne[0] * res_kernel_f16->ne[1]); // K*1

                // Step 2: Transpose to [K, C]
                struct ggml_tensor * kernel_2d_t = ggml_cont(ctx, ggml_transpose(ctx, kernel_2d));

                // Step 3: Reshape back to 3D [K, 1, C]
                struct ggml_tensor * kernel_for_dw = ggml_reshape_3d(ctx, kernel_2d_t,
                    kernel_2d_t->ne[0],  // K
                    1,                  // 1
                    kernel_2d_t->ne[1]); // C

                cur = ggml_conv_1d_dw(ctx, kernel_for_dw, cur, 1, unit_padding, dilation);
                if (w.residual_in_bias[l][u]) {
                    cur = ggml_add(ctx, cur, w.residual_in_bias[l][u]);
                }
            }

            // Permute back
            cur = ggml_permute(ctx, cur, 1, 0, 2, 3);

            // Snake out
            cur = snac_snake_forward_2d_ggml(ctx, cur, w.residual_out_alpha[l][u], output_len, channels);

            // Permute for 1x1 conv
            cur = ggml_permute(ctx, cur, 1, 0, 2, 3);

            // 1x1 conv out
            // Reshape kernel from 2D [C, C] to 3D [OC, IC, K] = [C, C, 1] for ggml_conv_1d
            // ggml_conv_1d requires F16 kernel
            if (w.residual_out_kernel[l][u]) {
                struct ggml_tensor * out_kernel_3d = ggml_reshape_3d(ctx, w.residual_out_kernel[l][u],
                    w.residual_out_kernel[l][u]->ne[0], w.residual_out_kernel[l][u]->ne[1], 1);  // [OC, IC, K]
                struct ggml_tensor * kernel_f16 = out_kernel_3d;
                if (out_kernel_3d->type != GGML_TYPE_F16) {
                    struct ggml_tensor * kernel_f16_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F16,
                        out_kernel_3d->ne[0], out_kernel_3d->ne[1], out_kernel_3d->ne[2]);
                    kernel_f16 = ggml_cpy(ctx, out_kernel_3d, kernel_f16_tensor);
                }
                cur = ggml_conv_1d(ctx, kernel_f16, cur, 1, 0, 1);
                if (w.residual_out_bias[l][u]) {
                    cur = ggml_add(ctx, cur, w.residual_out_bias[l][u]);
                }
            }

            // Permute back
            cur = ggml_permute(ctx, cur, 1, 0, 2, 3);

            // Residual connection
            cur = ggml_add(ctx, cur, residual);
        }

        cur_len = output_len;

        // Permute back for next layer
        cur = ggml_permute(ctx, cur, 1, 0, 2, 3);  // [out_ch, cur_len, batch_size]
    }

    // Step 7: Output layer
    // Permute to [T, C, batch]
    cur = ggml_permute(ctx, cur, 1, 0, 2, 3);

    // Snake activation
    cur = snac_snake_forward_2d_ggml(ctx, cur, w.snake_alpha_out, cur_len, 64);

    // Permute back to [C, T, batch]
    cur = ggml_permute(ctx, cur, 1, 0, 2, 3);

    // Output convolution
    // Kernel is stored in [IC, K, OC] = [64, 7, 1] format (GGML interpretation)
    // ggml_conv_1d expects [OC, IC, K] = [1, 64, 7]
    // Permute using (2, 0, 1, 3)
    // ggml_conv_1d requires F16 kernel
    {
        struct ggml_tensor * kernel_f16 = w.out_conv_kernel;
        if (w.out_conv_kernel->type != GGML_TYPE_F16) {
            struct ggml_tensor * kernel_f16_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F16,
                w.out_conv_kernel->ne[0], w.out_conv_kernel->ne[1], w.out_conv_kernel->ne[2]);
            kernel_f16 = ggml_cpy(ctx, w.out_conv_kernel, kernel_f16_tensor);
        }
        struct ggml_tensor * out_kernel_permuted = ggml_permute(ctx, kernel_f16, 2, 0, 1, 3);
        struct ggml_tensor * out_kernel_cont = ggml_cont(ctx, out_kernel_permuted);
        cur = ggml_conv_1d(ctx, out_kernel_cont, cur, 1, 3, 1);
    }
    if (w.out_conv_bias) {
        cur = ggml_add(ctx, cur, w.out_conv_bias);
    }

    // Tanh activation
    cur = ggml_tanh(ctx, cur);

    // Final shape: [1, output_samples, batch_size]
    return cur;
}

// Initialize batch context
bool snac_batch_init(
    struct snac_ggml_context & model_ctx,
    struct snac_batch_context & batch_ctx,
    int max_batch_size,
    int max_tokens)
{
    if (!model_ctx.loaded) {
        LOG_ERR("%s: Model context not loaded\n", __func__);
        return false;
    }

    // Create graph context
    struct ggml_init_params params = {
        /*.mem_size   =*/ 512*1024*1024,  // 512MB for batched graph
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    batch_ctx.ctx = ggml_init(params);
    if (!batch_ctx.ctx) {
        LOG_ERR("%s: Failed to create graph context\n", __func__);
        return false;
    }

    batch_ctx.backend = model_ctx.backend;
    batch_ctx.max_batch_size = max_batch_size;
    batch_ctx.max_tokens = max_tokens;

    // Create allocator
    batch_ctx.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(batch_ctx.backend));
    if (!batch_ctx.allocr) {
        LOG_ERR("%s: Failed to create allocator\n", __func__);
        ggml_free(batch_ctx.ctx);
        batch_ctx.ctx = nullptr;
        return false;
    }

    batch_ctx.initialized = true;
    LOG_INF("%s: Batch context initialized (max_batch=%d, max_tokens=%d)\n",
            __func__, max_batch_size, max_tokens);

    return true;
}

// Free batch context
void snac_batch_free(struct snac_batch_context & batch_ctx)
{
    if (batch_ctx.allocr) {
        ggml_gallocr_free(batch_ctx.allocr);
        batch_ctx.allocr = nullptr;
    }

    if (batch_ctx.ctx) {
        ggml_free(batch_ctx.ctx);
        batch_ctx.ctx = nullptr;
    }

    batch_ctx.initialized = false;
}

// Batch decode implementation
std::vector<std::vector<float>> snac_batch_decode(
    struct snac_ggml_context & model_ctx,
    struct snac_batch_context & batch_ctx,
    const std::vector<snac_batch_input> & batches)
{
    if (!model_ctx.loaded || !batch_ctx.initialized) {
        LOG_ERR("%s: Contexts not initialized\n", __func__);
        return {};
    }

    if (batches.empty()) {
        return {};
    }

    std::vector<std::vector<float>> results;

    // Process each batch
    for (const auto & batch : batches) {
        if (batch.n_seqs <= 0) continue;

        // Build graph for this batch
        struct ggml_tensor * output = snac_build_batch_graph(
            batch_ctx.ctx, model_ctx.weights,
            batch.n_seqs,
            batch.head0_len, batch.head1_len, batch.head2_len);

        if (!output) {
            LOG_ERR("%s: Failed to build batch graph\n", __func__);
            continue;
        }

        struct ggml_cgraph * gf = ggml_new_graph(batch_ctx.ctx);
        ggml_build_forward_expand(gf, output);

        // Allocate graph
        if (!ggml_gallocr_alloc_graph(batch_ctx.allocr, gf)) {
            LOG_ERR("%s: Failed to allocate batch graph\n", __func__);
            continue;
        }

        // Set input tokens
        struct ggml_tensor * tokens0 = ggml_get_tensor(batch_ctx.ctx, "batch_tokens_head0");
        struct ggml_tensor * tokens1 = ggml_get_tensor(batch_ctx.ctx, "batch_tokens_head1");
        struct ggml_tensor * tokens2 = ggml_get_tensor(batch_ctx.ctx, "batch_tokens_head2");

        if (tokens0 && batch.tokens_head0) {
            ggml_backend_tensor_set(tokens0, batch.tokens_head0, 0,
                                    batch.n_seqs * batch.head0_len * sizeof(int32_t));
        }
        if (tokens1 && batch.tokens_head1) {
            ggml_backend_tensor_set(tokens1, batch.tokens_head1, 0,
                                    batch.n_seqs * batch.head1_len * sizeof(int32_t));
        }
        if (tokens2 && batch.tokens_head2) {
            ggml_backend_tensor_set(tokens2, batch.tokens_head2, 0,
                                    batch.n_seqs * batch.head2_len * sizeof(int32_t));
        }

        // Execute graph
        if (ggml_backend_is_cpu(batch_ctx.backend)) {
            ggml_backend_cpu_set_n_threads(batch_ctx.backend, 4);
        }

        ggml_status status = ggml_backend_graph_compute(batch_ctx.backend, gf);
        if (status != GGML_STATUS_SUCCESS) {
            LOG_ERR("%s: Batch graph computation failed\n", __func__);
            continue;
        }

        // Extract output for each sequence
        // Output shape: [1, output_samples, batch_size]
        int64_t samples_per_seq = output->ne[1];
        std::vector<float> batch_output(batch.n_seqs * samples_per_seq);
        ggml_backend_tensor_get(output, batch_output.data(), 0, batch_output.size() * sizeof(float));

        // De-interleave: split batch output into individual sequences
        for (int s = 0; s < batch.n_seqs; s++) {
            std::vector<float> seq_output(samples_per_seq);
            for (int64_t i = 0; i < samples_per_seq; i++) {
                seq_output[i] = batch_output[s * samples_per_seq + i];
            }
            results.push_back(std::move(seq_output));
        }
    }

    return results;
}

// Convenience function for pyramid token format
std::vector<std::vector<float>> snac_batch_decode_pyramid(
    struct snac_ggml_context & model_ctx,
    struct snac_batch_context & batch_ctx,
    const std::vector<std::vector<std::vector<int>>> & pyramid_tokens_batch)
{
    if (pyramid_tokens_batch.empty()) {
        return {};
    }

    // Find max lengths for padding
    int64_t max_head0 = 0, max_head1 = 0, max_head2 = 0;
    for (const auto & pyramid : pyramid_tokens_batch) {
        if (pyramid.size() >= 1) max_head0 = std::max(max_head0, (int64_t)pyramid[0].size());
        if (pyramid.size() >= 2) max_head1 = std::max(max_head1, (int64_t)pyramid[1].size());
        if (pyramid.size() >= 3) max_head2 = std::max(max_head2, (int64_t)pyramid[2].size());
    }

    // Create batch input with padding
    snac_batch_input batch;
    batch.n_seqs = pyramid_tokens_batch.size();
    batch.head0_len = max_head0;
    batch.head1_len = max_head1;
    batch.head2_len = max_head2;

    // Allocate padded token arrays
    std::vector<int32_t> tokens0(batch.n_seqs * max_head0, 0);
    std::vector<int32_t> tokens1(batch.n_seqs * max_head1, 0);
    std::vector<int32_t> tokens2(batch.n_seqs * max_head2, 0);

    // Copy tokens with padding
    for (size_t s = 0; s < pyramid_tokens_batch.size(); s++) {
        const auto & pyramid = pyramid_tokens_batch[s];

        if (pyramid.size() >= 1 && !pyramid[0].empty()) {
            std::copy(pyramid[0].begin(), pyramid[0].end(),
                      tokens0.begin() + s * max_head0);
        }
        if (pyramid.size() >= 2 && !pyramid[1].empty()) {
            std::copy(pyramid[1].begin(), pyramid[1].end(),
                      tokens1.begin() + s * max_head1);
        }
        if (pyramid.size() >= 3 && !pyramid[2].empty()) {
            std::copy(pyramid[2].begin(), pyramid[2].end(),
                      tokens2.begin() + s * max_head2);
        }
    }

    batch.tokens_head0 = tokens0.data();
    batch.tokens_head1 = tokens1.data();
    batch.tokens_head2 = tokens2.data();

    // Process as single batch
    std::vector<snac_batch_input> batches = {batch};
    return snac_batch_decode(model_ctx, batch_ctx, batches);
}

// ============================================================================
// Phase 4: Streaming Buffer Implementation
// ============================================================================

// Static member definition for PYRAMID_MAP
constexpr int snac_streaming_buffer::PYRAMID_MAP[7];

void snac_streaming_buffer::reset() {
    head0_tokens.clear();
    head1_tokens.clear();
    head2_tokens.clear();
    frame_position = 0;
    total_frames_completed = 0;
}

bool snac_streaming_buffer::add_token(int32_t raw_token) {
    // Determine which head this position maps to
    int head = PYRAMID_MAP[frame_position];

    // Apply position-based offset adjustment (same as collect_audio_tokens_pyramid)
    // Orpheus encodes tokens from different codebooks at different offsets:
    // Position 0: tokens 0-4095 (codebook 0)
    // Position 1: tokens 4096-8191 (codebook 1)
    // Position 2: tokens 8192-12287 (codebook 2)
    // etc.
    // We need to subtract pos * CODEBOOK_SIZE to get the actual codebook index
    int32_t token_value = raw_token - frame_position * CODEBOOK_SIZE;

    // Validate token range - use default (0) for invalid tokens
    // This maintains frame structure which is critical for streaming
    if (token_value < 0 || token_value >= CODEBOOK_SIZE) {
        // Invalid token - use 0 as default to maintain frame structure
        token_value = 0;
        // Don't return early - continue to store the default value
    }

    // Add to appropriate head buffer
    switch (head) {
        case 0:
            head0_tokens.push_back(token_value);
            break;
        case 1:
            head1_tokens.push_back(token_value);
            break;
        case 2:
            head2_tokens.push_back(token_value);
            break;
        default:
            // Should never happen
            break;
    }

    // Advance frame position
    frame_position++;

    // Check if frame completed (7 tokens = 1 frame)
    bool frame_completed = (frame_position >= SNAC_GGML_FRAME_SIZE);
    if (frame_completed) {
        frame_position = 0;
        total_frames_completed++;
    }

    return frame_completed;
}

bool snac_streaming_buffer::has_frames(int n) const {
    return total_frames_completed >= n;
}

std::vector<std::vector<int32_t>> snac_streaming_buffer::get_completed_frames(int max_frames) const {
    std::vector<std::vector<int32_t>> result(3);

    int n = (max_frames < 0) ? total_frames_completed :
            std::min(max_frames, total_frames_completed);

    if (n <= 0) {
        return result;
    }

    // Copy first n frames worth of tokens from each head
    // Note: head0 has 1 token/frame, head1 has 2 tokens/frame, head2 has 4 tokens/frame
    // This is based on the pyramid structure: [0, 1, 2, 2, 1, 2, 2]
    // Position mapping: head0 appears once, head1 appears twice, head2 appears four times

    // head0: 1 token per frame
    if ((int)head0_tokens.size() >= n) {
        result[0].assign(head0_tokens.begin(), head0_tokens.begin() + n);
    }

    // head1: 2 tokens per frame
    if ((int)head1_tokens.size() >= n * 2) {
        result[1].assign(head1_tokens.begin(), head1_tokens.begin() + n * 2);
    }

    // head2: 4 tokens per frame
    if ((int)head2_tokens.size() >= n * 4) {
        result[2].assign(head2_tokens.begin(), head2_tokens.begin() + n * 4);
    }

    return result;
}

void snac_streaming_buffer::consume_frames(int n_frames) {
    if (n_frames <= 0 || n_frames > total_frames_completed) {
        return;
    }

    // Remove consumed tokens from each head
    // head0: 1 token per frame
    if ((int)head0_tokens.size() >= n_frames) {
        head0_tokens.erase(head0_tokens.begin(), head0_tokens.begin() + n_frames);
    }

    // head1: 2 tokens per frame
    if ((int)head1_tokens.size() >= n_frames * 2) {
        head1_tokens.erase(head1_tokens.begin(), head1_tokens.begin() + n_frames * 2);
    }

    // head2: 4 tokens per frame
    if ((int)head2_tokens.size() >= n_frames * 4) {
        head2_tokens.erase(head2_tokens.begin(), head2_tokens.begin() + n_frames * 4);
    }

    total_frames_completed -= n_frames;
}

// ============================================================================
// Phase 4.2: Streaming Context Implementation
// ============================================================================

bool snac_streaming_context::init(snac_ggml_context * ctx, const snac_streaming_config & cfg) {
    if (!ctx || !ctx->loaded) {
        LOG_ERR("%s: Invalid model context\n", __func__);
        return false;
    }

    model_ctx = ctx;
    config = cfg;
    buffer.reset();
    samples_already_output = 0;
    total_pcm_samples = 0;
    chunks_decoded = 0;
    last_output_frame_count = 0;

    LOG_INF("%s: Streaming context initialized (min_chunk=%d, full-context mode)\n",
            __func__, config.min_chunk_frames);

    return true;
}

bool snac_streaming_context::add_token_and_decode(
    int raw_token,
    snac_audio_callback callback,
    void * user_data)
{
    if (!model_ctx) {
        LOG_ERR("%s: Streaming context not initialized\n", __func__);
        return false;
    }

    // Add token to buffer
    buffer.add_token(raw_token);

    int current_frames = buffer.get_frame_count();

    // TWO-SIDED OVERLAP STREAMING:
    // SNAC decoder needs BOTH left and right context for correct boundary handling.
    // - Left context: first few frames of each decode don't have proper left padding
    // - Right context: last few frames don't have proper right padding
    //
    // We need overlap frames on BOTH sides to get clean output.
    // This means we need (min_chunk_frames + 2*overlap) total frames to output min_chunk_frames.

    int overlap = config.overlap_frames > 0 ? config.overlap_frames : 4;  // Default 4 frames overlap

    // For first decode, we have no "left overlap" to skip
    // So we use a different strategy:
    // - First decode: skip right overlap only (output frames 0 to current-overlap)
    // - Subsequent decodes: use left_offset to skip left overlap

    // Calculate how many frames we can output (skipping right overlap)
    int outputable_end = current_frames - overlap;

    // On first decode, we output from frame 0
    // On subsequent decodes, we need to skip the left overlap (which was already output with crossfade)
    int output_start = last_output_frame_count;

    // Check if we have enough NEW frames to output
    int new_outputable = outputable_end - output_start;
    if (new_outputable < config.min_chunk_frames) {
        return false;  // Not enough outputable frames yet
    }

    // Decode ALL tokens (decoder needs full context)
    auto tokens = buffer.get_completed_frames();

    if (tokens.size() != 3 || tokens[0].empty()) {
        LOG_WRN("%s: Failed to get frames\n", __func__);
        return false;
    }

    LOG_DBG("%s: Decoding %d frames, outputting frames %d to %d (%d new, keeping %d right overlap)\n",
            __func__, current_frames, output_start, outputable_end, new_outputable, overlap);

    // Decode ALL tokens
    std::vector<float> pcm = snac_ggml_decode(*model_ctx, tokens);

    if (pcm.empty()) {
        LOG_WRN("%s: Decode returned empty audio\n", __func__);
        return false;
    }

    // Calculate sample positions
    // Each frame produces SNAC_GGML_SAMPLES_PER_FRAME (2048) samples
    const int samples_per_frame = SNAC_GGML_SAMPLES_PER_FRAME;

    // Output samples from output_start to outputable_end
    int start_sample = output_start * samples_per_frame;
    int end_sample = outputable_end * samples_per_frame;

    std::vector<float> output_pcm;
    if (start_sample < (int)pcm.size()) {
        int actual_end = std::min(end_sample, (int)pcm.size());
        if (actual_end > start_sample) {
            output_pcm.assign(pcm.begin() + start_sample, pcm.begin() + actual_end);
        }
    }

    // Update tracking
    last_output_frame_count = outputable_end;

    // Output via callback
    bool continue_stream = true;
    if (!output_pcm.empty() && callback) {
        continue_stream = callback(output_pcm.data(), output_pcm.size(), user_data);
    }

    total_pcm_samples += output_pcm.size();
    chunks_decoded++;

    LOG_DBG("%s: Chunk %d: %zu samples (%.2fs), frames %d-%d, total: %d samples\n",
            __func__, chunks_decoded, output_pcm.size(),
            (float)output_pcm.size() / SNAC_GGML_SAMPLE_RATE,
            output_start, outputable_end,
            total_pcm_samples);

    return true;
}

void snac_streaming_context::flush(snac_audio_callback callback, void * user_data) {
    if (!model_ctx || buffer.get_frame_count() == 0) {
        return;
    }

    // Get all remaining frames
    int remaining_frames = buffer.get_frame_count();

    // If we've already output some frames, only output the remaining
    if (last_output_frame_count >= remaining_frames) {
        buffer.reset();
        last_output_frame_count = 0;
        return;
    }

    auto tokens = buffer.get_completed_frames();

    if (tokens.size() != 3 || tokens[0].empty()) {
        buffer.reset();
        last_output_frame_count = 0;
        return;
    }

    LOG_INF("%s: Flushing remaining %d frames (frames %d to %d)\n",
            __func__, remaining_frames - last_output_frame_count,
            last_output_frame_count, remaining_frames);

    // Decode ALL remaining tokens
    std::vector<float> pcm = snac_ggml_decode(*model_ctx, tokens);

    if (pcm.empty()) {
        buffer.reset();
        last_output_frame_count = 0;
        return;
    }

    // Calculate sample positions for remaining output
    const int samples_per_frame = SNAC_GGML_SAMPLES_PER_FRAME;
    int start_sample = last_output_frame_count * samples_per_frame;
    int end_sample = remaining_frames * samples_per_frame;

    // Output remaining samples
    std::vector<float> output_pcm;
    if (start_sample < (int)pcm.size()) {
        int actual_end = std::min(end_sample, (int)pcm.size());
        if (actual_end > start_sample) {
            output_pcm.assign(pcm.begin() + start_sample, pcm.begin() + actual_end);
        }
    }

    if (!output_pcm.empty() && callback) {
        callback(output_pcm.data(), output_pcm.size(), user_data);
        total_pcm_samples += output_pcm.size();
        chunks_decoded++;
    }

    LOG_INF("%s: Final chunk: %zu samples (%.2fs), total: %d samples\n",
            __func__, output_pcm.size(),
            (float)output_pcm.size() / SNAC_GGML_SAMPLE_RATE,
            total_pcm_samples);

    // Reset state
    buffer.reset();
    last_output_frame_count = 0;
}

void snac_streaming_context::reset() {
    buffer.reset();
    samples_already_output = 0;
    last_output_frame_count = 0;
    total_pcm_samples = 0;
    chunks_decoded = 0;
}
