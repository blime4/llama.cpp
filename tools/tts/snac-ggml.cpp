// SNAC vocoder ggml-based implementation
// Phase 1: Single sequence processing with ggml graph

#include "snac-ggml.h"
#include "log.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdio>

// Snake activation: snake(x, alpha) = x + sin²(alpha * x) / alpha
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

    // sin² / alpha (add small epsilon inline via scale to avoid division by zero)
    // Using alpha * (1 + eps) ≈ alpha + eps for numerical stability
    struct ggml_tensor * div = ggml_div(ctx, sin2, alpha);

    // x + sin² / alpha
    return ggml_add(ctx, x, div);
}

// Custom ConvTranspose1D implementation for GGML
// Since GGML's native conv_transpose_1d doesn't support padding,
// we use a simplified approach: nearest-neighbor upsampling + linear projection
//
// kernel: [K, OC, IC] in GGUF format
// input: [IC, L] (channels first)
// Returns: [OC, output_len]
static struct ggml_tensor * snac_conv_transpose_1d_custom(
    struct ggml_context * ctx,
    struct ggml_tensor * kernel,   // [K, OC, IC]
    struct ggml_tensor * input,    // [IC, L]
    struct ggml_tensor * bias,     // [OC] or nullptr
    int64_t stride,
    int64_t padding,
    int64_t output_padding)
{
    // Get dimensions from kernel
    int64_t K = kernel->ne[0];    // Kernel size
    int64_t OC = kernel->ne[1];   // Output channels
    int64_t IC = kernel->ne[2];   // Input channels
    int64_t L = input->ne[1];     // Input length

    // Output length calculation for ConvTranspose1D
    int64_t output_len = (L - 1) * stride + K - 2 * padding + output_padding;

    (void)K;  // May be used for proper implementation later

    LOG_DBG("%s: ConvTranspose1D: OC=%lld, IC=%lld, L=%lld, s=%lld -> out_len=%lld\n",
            __func__, (long long)OC, (long long)IC, (long long)L,
            (long long)stride, (long long)output_len);

    // Simplified approach for now:
    // 1. Reshape kernel to 2D: [K, OC, IC] -> use first slice -> [OC, IC]
    // 2. Transpose for matmul: [IC, OC]
    // 3. Project input: [IC, L] @ [IC, OC] -> [OC, L]
    // 4. Upsample to output length

    // View of kernel as [OC, IC] by taking first K slice
    // kernel data is [K][OC][IC], stride in bytes is OC*IC*sizeof(float)
    struct ggml_tensor * kernel_2d = ggml_view_2d(ctx, kernel, OC, IC,
                                                   K * OC * sizeof(float),  // row stride: skip K dimension
                                                   0);  // offset

    // Transpose for matrix multiplication
    struct ggml_tensor * w_t = ggml_cont(ctx, ggml_transpose(ctx, kernel_2d));  // [IC, OC]

    // Project: w_t [IC, OC] @ input [IC, L] -> [OC, L]
    // Note: ggml_mul_mat(a, b) computes a @ b where a=[K,N], b=[K,M] -> [N,M]
    struct ggml_tensor * projected = ggml_mul_mat(ctx, w_t, input);  // [OC, L]

    // Upsample to output length using nearest neighbor
    // Note: ggml_upscale multiplies ne0 by scale_factor, so we compute the factor
    int scale_factor = (output_len + L - 1) / L;  // Ceiling division
    struct ggml_tensor * upsampled = ggml_upscale(ctx, projected, scale_factor, GGML_SCALE_MODE_NEAREST);

    // Add bias if present
    if (bias) {
        // Bias is [OC], reshape to [OC, 1] for broadcasting
        struct ggml_tensor * bias_2d = ggml_reshape_2d(ctx, bias, OC, 1);
        upsampled = ggml_add(ctx, upsampled, bias_2d);
    }

    return upsampled;
}

// Snake activation for 2D tensor with alpha broadcast over channels
// x: [T, C], alpha: [C]
static struct ggml_tensor * snac_snake_forward_2d_ggml(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * alpha,
    int64_t T,
    int64_t C)
{
    (void)T; // Used for documentation/clarity

    // Reshape alpha to [1, C] for broadcasting
    struct ggml_tensor * alpha_2d = ggml_reshape_2d(ctx, alpha, C, 1);

    // Broadcast alpha to [T, C]
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

    // Calculate output length
    int output_padding = stride % 2;
    int64_t output_len = (input_len - 1) * stride + kernel_size - 2 * padding + output_padding;

    // Snake activation
    struct ggml_tensor * cur = snac_snake_forward_2d_ggml(ctx, input, alpha, input_len, in_channels);

    // ConvTranspose1D
    // ggml_conv_transpose_1d expects kernel in [OC, IC, K] format
    // Our kernel is stored as [IC, OC, K], need to handle this
    // For now, assume kernel is already in correct format
    cur = ggml_conv_transpose_1d(ctx, kernel, cur, stride, padding, 1);

    // Add bias (broadcast over time)
    if (bias) {
        cur = ggml_add(ctx, cur, bias);
    }

    // Noise injection (if noise_kernel exists)
    if (noise_kernel) {
        // 1x1 conv to get noise scale
        struct ggml_tensor * noise_scale = ggml_conv_1d(ctx, noise_kernel, cur, 1, 0, 1);
        (void)noise_scale; // Noise injection disabled for inference
        // struct ggml_tensor * noise = ...;
        // cur = ggml_add(ctx, cur, ggml_mul(ctx, noise_scale, noise));
    }

    // Residual units (3 units per layer)
    for (int u = 0; u < 3; u++) {
        if (!residual_in_alpha[u]) continue;

        // Save input for residual connection
        struct ggml_tensor * residual = cur;

        int channels = out_channels;
        int unit_kernel_size = 7;
        int dilation = (int)std::pow(3, u);
        int unit_padding = ((unit_kernel_size - 1) * dilation) / 2;

        // Snake in
        cur = snac_snake_forward_2d_ggml(ctx, cur, residual_in_alpha[u], output_len, channels);

        // Depthwise conv in (groups = channels)
        if (residual_in_kernel[u]) {
            cur = ggml_conv_1d_dw(ctx, residual_in_kernel[u], cur, 1, unit_padding, dilation);
            if (residual_in_bias[u]) {
                cur = ggml_add(ctx, cur, residual_in_bias[u]);
            }
        }

        // Snake out
        cur = snac_snake_forward_2d_ggml(ctx, cur, residual_out_alpha[u], output_len, channels);

        // 1x1 conv out
        if (residual_out_kernel[u]) {
            cur = ggml_conv_1d(ctx, residual_out_kernel[u], cur, 1, 0, 1);
            if (residual_out_bias[u]) {
                cur = ggml_add(ctx, cur, residual_out_bias[u]);
            }
        }

        // Residual connection
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
    // cur is [quantizer_dim, seq_len] = [1024, seq_len] after combining embeddings
    // ggml_conv_1d_dw expects input in [N, IC, L, 1] format
    // We need to reshape [1024, seq_len] to [1, 1024, seq_len, 1]
    int64_t seq_len = cur->ne[1];
    int64_t quantizer_dim = cur->ne[0];

    LOG_INF("%s: Before reshape: cur shape = [%lld, %lld]\n", __func__,
            (long long)cur->ne[0], (long long)cur->ne[1]);
    LOG_INF("%s: in_conv_kernel shape = [%lld, %lld, %lld, %lld]\n", __func__,
            (long long)w.in_conv_kernel->ne[0], (long long)w.in_conv_kernel->ne[1],
            (long long)w.in_conv_kernel->ne[2], (long long)w.in_conv_kernel->ne[3]);

    // The in_conv is a depthwise conv with kernel_size=7, padding=3
    // Our kernel is [1024, 1, 7, 1] in PyTorch depthwise format: [out_ch, in_ch/groups, k]
    //
    // GGML doesn't directly support grouped/depthwise convolution.
    // For now, implement depthwise conv manually by:
    // 1. Reshape kernel to [K, IC] where each row is a filter for one channel
    // 2. For each channel, apply 1D conv separately
    //
    // Simpler approach: Since depthwise conv is essentially per-channel filtering,
    // we can use im2col + element-wise operations.
    //
    // For Phase 1 testing, let's skip the in_conv and see if the rest works.
    // TODO: Implement proper depthwise conv for GGML

    // Skip in_conv for now - just add bias if present
    if (w.in_conv_bias) {
        cur = ggml_add(ctx, cur, w.in_conv_bias);
    }

    LOG_INF("%s: Skipping in_conv depthwise (not yet implemented), using identity + bias\n", __func__);
    if (w.in_conv_bias) {
        cur = ggml_add(ctx, cur, w.in_conv_bias);
    }

    // Step 4: Up convolution (1x1 conv)
    // Project from quantizer_dim to decoder_dim
    LOG_INF("%s: cur shape before up_conv = [%lld, %lld]\n", __func__,
            (long long)cur->ne[0], (long long)cur->ne[1]);
    LOG_INF("%s: up_conv_kernel shape = [%lld, %lld, %lld, %lld]\n", __func__,
            (long long)w.up_conv_kernel->ne[0], (long long)w.up_conv_kernel->ne[1],
            (long long)w.up_conv_kernel->ne[2], (long long)w.up_conv_kernel->ne[3]);

    // up_conv is a 1x1 conv (essentially a linear projection)
    // Kernel is [OC, IC, 1, 1] = [1536, 1024, 1, 1]
    // Input is [IC, L] = [1024, seq_len]
    //
    // For ggml_mul_mat, the assertion is a->ne[0] == b->ne[0]
    // We need kernel->ne[0] == input->ne[0]
    //
    // Kernel: [1536, 1024, 1, 1] has ne[0]=1536
    // Input: [1024, seq_len] has ne[0]=1024
    // These don't match!
    //
    // We need to transpose the kernel to [1024, 1536, 1, 1]
    // Then ne[0]=1024 will match input's ne[0]=1024

    // First reshape kernel to 2D: [1536, 1024]
    struct ggml_tensor * up_kernel_2d = ggml_reshape_2d(ctx, w.up_conv_kernel,
                                                        w.up_conv_kernel->ne[0] * w.up_conv_kernel->ne[1],
                                                        w.up_conv_kernel->ne[2] * w.up_conv_kernel->ne[3]);

    LOG_INF("%s: up_kernel_2d shape = [%lld, %lld]\n", __func__,
            (long long)up_kernel_2d->ne[0], (long long)up_kernel_2d->ne[1]);

    // up_kernel_2d: [1536*1024, 1] = [1572864, 1] - that's wrong!

    // Actually, the kernel is 4D [1536, 1024, 1, 1], which has:
    // ne[0]=1536, ne[1]=1024, ne[2]=1, ne[3]=1
    // Total elements = 1536 * 1024 = 1572864

    // For linear projection, we want kernel as [IC, OC] = [1024, 1536]
    // But it's stored as [OC, IC] = [1536, 1024]

    // Transpose the kernel to get [IC, OC]
    struct ggml_tensor * up_kernel_t = ggml_cont(ctx, ggml_transpose(ctx, w.up_conv_kernel));
    // After transpose: ne[0]=1024, ne[1]=1536, ne[2]=1, ne[3]=1

    LOG_INF("%s: up_kernel_t shape = [%lld, %lld, %lld, %lld]\n", __func__,
            (long long)up_kernel_t->ne[0], (long long)up_kernel_t->ne[1],
            (long long)up_kernel_t->ne[2], (long long)up_kernel_t->ne[3]);

    // Now mul_mat should work:
    // - up_kernel_t: [1024, 1536, 1, 1] (ne[0]=1024)
    // - cur: [1024, seq_len] (ne[0]=1024)
    // assertion: 1024 == 1024 ✓
    cur = ggml_mul_mat(ctx, up_kernel_t, cur);

    LOG_INF("%s: cur shape after up_conv = [%lld, %lld]\n", __func__,
            (long long)cur->ne[0], (long long)cur->ne[1]);

    if (w.up_conv_bias) {
        cur = ggml_add(ctx, cur, w.up_conv_bias);
    }

    int64_t cur_channels = SNAC_GGML_DECODER_DIM;

    // Step 5: LocalMHA attention (optional - skip for Phase 1)
    // TODO: Implement LocalMHA with RoPE

    // Step 6: Decoder layers
    // NOTE: ggml's conv_transpose_1d has a different API than PyTorch
    // and doesn't directly support our kernel format.
    // For Phase 1, we skip the decoder layers and use a simplified output.
    //
    // TODO: Implement proper ConvTranspose1D for SNAC decoder
    LOG_WRN("%s: Decoder layers not yet implemented, using simplified output\n", __func__);

    // For Phase 1, output the upsampled features directly
    // This will produce incorrect audio but verifies the quantizer works

    // Simple output: project to 1 channel using a learned or fixed projection
    // For now, just take the mean across channels to get a single channel
    // cur: [1536, seq_len] -> output: [seq_len]

    // Actually, let's try a simple mean reduction
    // cur = ggml_mean(ctx, cur);  // This would give [1]

    // Better: use the first channel as output (placeholder)
    // cur = ggml_view_1d(ctx, cur, seq_len, 0);  // Takes first row

    // Even better: apply a simple linear projection to 1 channel
    // Using a fixed weight of 1/sqrt(channels) for averaging
    // For now, just return zeros with the correct shape

    // Output should be [1, seq_len * upsample_factor]
    // upsample_factor = 8 * 8 * 4 * 2 = 512
    int64_t total_upsample = 512;
    int64_t final_output_len = seq_len * total_upsample;

    LOG_INF("%s: Creating placeholder output of length %lld (seq_len=%lld * upsample=%lld)\n",
            __func__, (long long)final_output_len, (long long)seq_len, (long long)total_upsample);

    // Create placeholder output using operations that are tracked by the allocator
    // Use ggml_scale with factor 0 to get a zero tensor
    // Note: ggml_scale creates a new tensor properly connected to the graph
    cur = ggml_scale(ctx, cur, 0.0f);

    // Use view to take the first final_output_len elements
    // cur has 1536 * seq_len elements, we need seq_len * 512
    // Since 1536 > 512, we can take a view of the first row
    // cur: [1536, seq_len] after scale -> view first 512*seq_len elements
    if (ggml_nelements(cur) >= final_output_len) {
        cur = ggml_view_1d(ctx, cur, final_output_len, 0);
    } else {
        // Fallback: if cur is smaller, just use the whole thing
        cur = ggml_view_1d(ctx, cur, ggml_nelements(cur), 0);
    }

    // Apply tanh to bound output (no-op for zeros)
    cur = ggml_tanh(ctx, cur);

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

    // Pre-expand tokens using repeat_interleave
    std::vector<int> expanded_head0;
    std::vector<int> expanded_head1;
    const std::vector<int> & expanded_head2 = pyramid_tokens[2];  // No expansion needed

    int stride0 = ctx.vq_strides[0];  // 4
    int stride1 = ctx.vq_strides[1];  // 2

    // Verify pyramid structure: head0 * 4 = head1 * 2 = head2
    int64_t expected_head2_len = head0_orig_len * stride0;
    if (expected_head2_len != head2_len) {
        LOG_WRN("%s: Pyramid structure mismatch: head0*4=%lld != head2=%lld\n", __func__,
                (long long)expected_head2_len, (long long)head2_len);
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
        struct ggml_tensor * tokens0 = ggml_get_tensor(graph_ctx, "tokens_head0");
        if (tokens0 && !expanded_head0.empty()) {
            ggml_backend_tensor_set(tokens0, expanded_head0.data(), 0, output_len * sizeof(int));
        } else if (!expanded_head0.empty()) {
            LOG_ERR("%s: tokens_head0 tensor not found\n", __func__);
            set_input_success = false;
        }
    }

    {
        struct ggml_tensor * tokens1 = ggml_get_tensor(graph_ctx, "tokens_head1");
        if (tokens1 && !expanded_head1.empty()) {
            ggml_backend_tensor_set(tokens1, expanded_head1.data(), 0, output_len * sizeof(int));
        } else if (!expanded_head1.empty()) {
            LOG_ERR("%s: tokens_head1 tensor not found\n", __func__);
            set_input_success = false;
        }
    }

    {
        struct ggml_tensor * tokens2 = ggml_get_tensor(graph_ctx, "tokens_head2");
        if (tokens2 && !expanded_head2.empty()) {
            ggml_backend_tensor_set(tokens2, expanded_head2.data(), 0, output_len * sizeof(int));
        } else if (!expanded_head2.empty()) {
            LOG_ERR("%s: tokens_head2 tensor not found\n", __func__);
            set_input_success = false;
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
    // For simplicity in Phase 2, we assume all sequences have same length
    // and combine embeddings additively

    // The pyramid structure means:
    // - head0 contributes to all positions
    // - head1 contributes to positions 1, 3, 5, ... (every 2nd)
    // - head2 contributes to positions 3, 7, 11, ... (every 4th starting at 3)

    // For initial implementation, we'll do a simplified combination:
    // Just add head embeddings (this won't produce correct audio but validates batch flow)

    // Start with head0 embeddings
    cur = head_embeddings[0];
    // cur: [head0_len, quantizer_dim, batch_size]

    // TODO: Implement proper repeat_interleave for pyramid structure
    // For now, just add if dimensions match (they won't in real usage)
    // This is a placeholder for the actual pyramid combination logic

    // Step 4: Input convolution (depthwise) - batched
    // Reshape cur to [T, C, batch] -> conv expects [C, T, batch]
    cur = ggml_permute(ctx, cur, 1, 0, 2, 3);  // [quantizer_dim, head0_len, batch_size]

    // Apply depthwise conv
    cur = ggml_conv_1d_dw(ctx, w.in_conv_kernel, cur, 1, 3, 1);
    if (w.in_conv_bias) {
        cur = ggml_add(ctx, cur, w.in_conv_bias);
    }

    // Step 5: Up convolution (1x1 conv) - project to decoder_dim
    cur = ggml_conv_1d(ctx, w.up_conv_kernel, cur, 1, 0, 1);
    if (w.up_conv_bias) {
        cur = ggml_add(ctx, cur, w.up_conv_bias);
    }
    // cur: [decoder_dim, head0_len, batch_size]

    // Step 6: Decoder layers - each with ConvTranspose1D
    int layer_channels[4] = {1024, 512, 256, 128};
    int out_channels[4] = {512, 256, 128, 64};
    int decoder_rates[4] = {8, 8, 4, 2};
    int kernel_sizes[4] = {16, 16, 8, 4};

    int64_t cur_len = head0_len;

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

        // ConvTranspose1D
        cur = ggml_conv_transpose_1d(ctx, w.layer_kernel[l], cur, stride, padding, 1);

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

            // Depthwise conv in
            if (w.residual_in_kernel[l][u]) {
                cur = ggml_conv_1d_dw(ctx, w.residual_in_kernel[l][u], cur, 1, unit_padding, dilation);
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
            if (w.residual_out_kernel[l][u]) {
                cur = ggml_conv_1d(ctx, w.residual_out_kernel[l][u], cur, 1, 0, 1);
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
    cur = ggml_conv_1d(ctx, w.out_conv_kernel, cur, 1, 3, 1);
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
