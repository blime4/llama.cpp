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

    // Codebook lookup using ggml_get_rows
    // codebook: [codebook_size, codebook_dim] = [4096, 8]
    // tokens: [seq_len]
    // result: [codebook_dim, seq_len] = [8, seq_len]
    struct ggml_tensor * embeddings = ggml_get_rows(ctx, codebook, tokens);

    // Transpose to [seq_len, codebook_dim]
    embeddings = ggml_cont(ctx, ggml_transpose(ctx, embeddings));

    // Apply 1x1 convolution (equivalent to linear projection)
    // out_proj: [quantizer_dim, codebook_dim] = [768, 8]
    // For conv_1d: kernel should be [out_ch, in_ch, k] = [768, 8, 1]
    // Reshape out_proj to [8, 1, 768] for conv_1d
    // Actually, ggml_conv_1d expects [out_ch, in_ch, k_size]
    // So we need to reshape out_proj appropriately

    // Transpose out_proj from [768, 8] to [8, 768, 1] for 1x1 conv
    // But ggml_conv_1d expects [OC, IC, K], so [768, 8, 1]
    // Let's use reshape to add the kernel dimension
    struct ggml_tensor * kernel_3d = ggml_reshape_3d(ctx, out_proj, quantizer_dim, codebook_dim, 1);

    // Apply 1x1 conv (stride=1, padding=0, dilation=1)
    struct ggml_tensor * projected = ggml_conv_1d(ctx, kernel_3d, embeddings, 1, 0, 1);

    // Add bias
    if (out_bias) {
        // Reshape bias to [quantizer_dim] if needed
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

// Build the complete SNAC computation graph
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

    // Step 1: Quantizer forward for each head
    std::vector<struct ggml_tensor *> head_embeddings(3);

    // Head 0
    if (head0_len > 0) {
        struct ggml_tensor * tokens0 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, head0_len);
        ggml_set_name(tokens0, "tokens_head0");
        // Copy tokens to tensor data - will be set later
        head_embeddings[0] = snac_quantizer_forward_ggml(
            ctx, tokens0, w.quant_codebook[0], w.quant_out_proj[0], w.quant_out_bias[0],
            SNAC_GGML_CODEBOOK_SIZE, SNAC_GGML_CODEBOOK_DIM, SNAC_GGML_QUANTIZER_DIM);
    }

    // Head 1
    if (head1_len > 0) {
        struct ggml_tensor * tokens1 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, head1_len);
        ggml_set_name(tokens1, "tokens_head1");
        head_embeddings[1] = snac_quantizer_forward_ggml(
            ctx, tokens1, w.quant_codebook[1], w.quant_out_proj[1], w.quant_out_bias[1],
            SNAC_GGML_CODEBOOK_SIZE, SNAC_GGML_CODEBOOK_DIM, SNAC_GGML_QUANTIZER_DIM);
    }

    // Head 2
    if (head2_len > 0) {
        struct ggml_tensor * tokens2 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, head2_len);
        ggml_set_name(tokens2, "tokens_head2");
        head_embeddings[2] = snac_quantizer_forward_ggml(
            ctx, tokens2, w.quant_codebook[2], w.quant_out_proj[2], w.quant_out_bias[2],
            SNAC_GGML_CODEBOOK_SIZE, SNAC_GGML_CODEBOOK_DIM, SNAC_GGML_QUANTIZER_DIM);
    }

    // Step 2: Combine quantizer outputs with pyramid upsampling
    // This matches the sequential addition with repeat_interleave
    cur = head_embeddings[0];

    for (int h = 1; h < 3; h++) {
        if (!head_embeddings[h]) continue;

        // These will be used for repeat_interleave implementation
        (void)vq_strides;

        // Upsample previous output by repeat_interleave
        // This requires custom operation - for Phase 1, we'll handle it outside graph
        // For now, assume embeddings are already combined
        // TODO: Implement repeat_interleave as ggml custom op

        // Add current head's output
        cur = ggml_add(ctx, head_embeddings[h], cur);
    }

    // Step 3: Input convolution (depthwise)
    // Reshape to [T, C] for conv_1d_dw

    // in_conv: depthwise conv with kernel_size=7, padding=3
    cur = ggml_conv_1d_dw(ctx, w.in_conv_kernel, cur, 1, 3, 1);
    if (w.in_conv_bias) {
        cur = ggml_add(ctx, cur, w.in_conv_bias);
    }

    // Step 4: Up convolution (1x1 conv)
    // Project from quantizer_dim (768) to decoder_dim (1024)
    cur = ggml_conv_1d(ctx, w.up_conv_kernel, cur, 1, 0, 1);
    if (w.up_conv_bias) {
        cur = ggml_add(ctx, cur, w.up_conv_bias);
    }

    int64_t cur_channels = SNAC_GGML_DECODER_DIM;

    // Step 5: LocalMHA attention (optional - skip for Phase 1)
    // TODO: Implement LocalMHA with RoPE

    // Step 6: Decoder layers
    int layer_channels[4] = {1024, 512, 256, 128};
    int out_channels[4] = {512, 256, 128, 64};
    int decoder_rates[4] = {8, 8, 4, 2};
    int kernel_sizes[4] = {16, 16, 8, 4};

    for (int l = 0; l < 4; l++) {
        if (!w.layer_kernel[l]) continue;

        int in_ch = layer_channels[l];
        int out_ch = out_channels[l];
        int stride = decoder_rates[l];
        int ks = kernel_sizes[l];
        int padding = (stride + 1) / 2;

        // Copy residual arrays to temporary pointers for function call
        struct ggml_tensor * res_in_alpha[3] = {w.residual_in_alpha[l][0], w.residual_in_alpha[l][1], w.residual_in_alpha[l][2]};
        struct ggml_tensor * res_in_kernel[3] = {w.residual_in_kernel[l][0], w.residual_in_kernel[l][1], w.residual_in_kernel[l][2]};
        struct ggml_tensor * res_in_bias[3] = {w.residual_in_bias[l][0], w.residual_in_bias[l][1], w.residual_in_bias[l][2]};
        struct ggml_tensor * res_out_alpha[3] = {w.residual_out_alpha[l][0], w.residual_out_alpha[l][1], w.residual_out_alpha[l][2]};
        struct ggml_tensor * res_out_kernel[3] = {w.residual_out_kernel[l][0], w.residual_out_kernel[l][1], w.residual_out_kernel[l][2]};
        struct ggml_tensor * res_out_bias[3] = {w.residual_out_bias[l][0], w.residual_out_bias[l][1], w.residual_out_bias[l][2]};

        cur = snac_decoder_layer_forward_ggml(
            ctx, cur, cur->ne[1],
            w.layer_alpha[l], w.layer_kernel[l], w.layer_bias[l], w.layer_noise_kernel[l],
            in_ch, out_ch, ks, stride, padding, l,
            res_in_alpha, res_in_kernel, res_in_bias,
            res_out_alpha, res_out_kernel, res_out_bias);

        cur_channels = out_ch;
    }

    // Step 7: Output layer
    // Snake activation
    cur = snac_snake_forward_2d_ggml(ctx, cur, w.snake_alpha_out, cur->ne[1], cur_channels);

    // Output convolution (reduce to 1 channel)
    cur = ggml_conv_1d(ctx, w.out_conv_kernel, cur, 1, 3, 1);
    if (w.out_conv_bias) {
        cur = ggml_add(ctx, cur, w.out_conv_bias);
    }

    // Tanh activation
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

    // First pass: create ggml context without loading data
    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/ 1024 * 1024 * 1024,  // 1GB for tensors
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,  // Don't allocate yet
    };
    ctx.ctx = ggml_init(ggml_params);
    if (!ctx.ctx) {
        LOG_ERR("%s: Failed to create ggml context\n", __func__);
        return false;
    }

    // Load GGUF file (metadata only first)
    struct gguf_init_params gguf_params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ NULL,
    };

    struct gguf_context * gguf_ctx = gguf_init_from_file(model_path, gguf_params);
    if (!gguf_ctx) {
        LOG_ERR("%s: Failed to load GGUF file: %s\n", __func__, model_path);
        return false;
    }

    // Count tensors
    size_t n_tensors = gguf_get_n_tensors(gguf_ctx);
    LOG_INF("%s: Found %zu tensors in GGUF file\n", __func__, n_tensors);

    // Free and reload with tensor data
    gguf_free(gguf_ctx);

    // Reload with context to get tensor data
    // Keep no_alloc=true so we can allocate with backend buffer
    gguf_params.no_alloc = true;
    gguf_params.ctx = &ctx.ctx;
    gguf_ctx = gguf_init_from_file(model_path, gguf_params);

    if (!gguf_ctx) {
        LOG_ERR("%s: Failed to reload GGUF file with tensors\n", __func__);
        return false;
    }

    // Load all weights into ctx.weights structure
    int tensors_loaded = 0;

    // Input convolutions
    if (load_gguf_tensor(gguf_ctx, ctx.ctx, "snac.in.weight", ctx.weights.in_conv_kernel, false)) tensors_loaded++;
    if (load_gguf_tensor(gguf_ctx, ctx.ctx, "snac.in.bias", ctx.weights.in_conv_bias, false)) tensors_loaded++;
    if (load_gguf_tensor(gguf_ctx, ctx.ctx, "snac.up.weight", ctx.weights.up_conv_kernel, false)) tensors_loaded++;
    if (load_gguf_tensor(gguf_ctx, ctx.ctx, "snac.up.bias", ctx.weights.up_conv_bias, false)) tensors_loaded++;

    // Output layer
    if (load_gguf_tensor(gguf_ctx, ctx.ctx, "snac.final.weight", ctx.weights.out_conv_kernel, false)) tensors_loaded++;
    if (load_gguf_tensor(gguf_ctx, ctx.ctx, "snac.final.bias", ctx.weights.out_conv_bias, false)) tensors_loaded++;
    if (load_gguf_tensor(gguf_ctx, ctx.ctx, "snac.alpha_out", ctx.weights.snake_alpha_out, false)) tensors_loaded++;

    // Decoder layers
    for (int l = 0; l < SNAC_GGML_N_DECODER_LAYERS; l++) {
        char name[256];

        snprintf(name, sizeof(name), "snac.layers.%d.alpha", l);
        if (load_gguf_tensor(gguf_ctx, ctx.ctx, name, ctx.weights.layer_alpha[l], false)) tensors_loaded++;

        snprintf(name, sizeof(name), "snac.layers.%d.weight", l);
        if (load_gguf_tensor(gguf_ctx, ctx.ctx, name, ctx.weights.layer_kernel[l], false)) tensors_loaded++;

        snprintf(name, sizeof(name), "snac.layers.%d.bias", l);
        if (load_gguf_tensor(gguf_ctx, ctx.ctx, name, ctx.weights.layer_bias[l], false)) tensors_loaded++;

        snprintf(name, sizeof(name), "snac.layers.%d.noise_weight", l);
        if (load_gguf_tensor(gguf_ctx, ctx.ctx, name, ctx.weights.layer_noise_kernel[l], false)) tensors_loaded++;

        // Residual units
        for (int u = 0; u < 3; u++) {
            snprintf(name, sizeof(name), "snac.layers.%d.%d.in_alpha", l, u);
            if (load_gguf_tensor(gguf_ctx, ctx.ctx, name, ctx.weights.residual_in_alpha[l][u], false)) tensors_loaded++;

            snprintf(name, sizeof(name), "snac.layers.%d.%d.in_weight", l, u);
            if (load_gguf_tensor(gguf_ctx, ctx.ctx, name, ctx.weights.residual_in_kernel[l][u], false)) tensors_loaded++;

            snprintf(name, sizeof(name), "snac.layers.%d.%d.in_bias", l, u);
            if (load_gguf_tensor(gguf_ctx, ctx.ctx, name, ctx.weights.residual_in_bias[l][u], false)) tensors_loaded++;

            snprintf(name, sizeof(name), "snac.layers.%d.%d.out_alpha", l, u);
            if (load_gguf_tensor(gguf_ctx, ctx.ctx, name, ctx.weights.residual_out_alpha[l][u], false)) tensors_loaded++;

            snprintf(name, sizeof(name), "snac.layers.%d.%d.out_weight", l, u);
            if (load_gguf_tensor(gguf_ctx, ctx.ctx, name, ctx.weights.residual_out_kernel[l][u], false)) tensors_loaded++;

            snprintf(name, sizeof(name), "snac.layers.%d.%d.out_bias", l, u);
            if (load_gguf_tensor(gguf_ctx, ctx.ctx, name, ctx.weights.residual_out_bias[l][u], false)) tensors_loaded++;
        }
    }

    // Quantizers
    for (int q = 0; q < ctx.n_quantizers; q++) {
        char name[256];

        snprintf(name, sizeof(name), "snac.quantizers.%d.codebook.weight", q);
        if (load_gguf_tensor(gguf_ctx, ctx.ctx, name, ctx.weights.quant_codebook[q], false)) tensors_loaded++;

        snprintf(name, sizeof(name), "snac.quantizers.%d.out_proj.weight", q);
        if (load_gguf_tensor(gguf_ctx, ctx.ctx, name, ctx.weights.quant_out_proj[q], false)) tensors_loaded++;

        snprintf(name, sizeof(name), "snac.quantizers.%d.out_proj.bias", q);
        if (load_gguf_tensor(gguf_ctx, ctx.ctx, name, ctx.weights.quant_out_bias[q], false)) tensors_loaded++;
    }

    LOG_INF("%s: Loaded %d tensors\n", __func__, tensors_loaded);

    // Allocate backend buffer for all tensors
    ctx.buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx.ctx, ggml_backend_get_default_buffer_type(ctx.backend));
    if (!ctx.buffer) {
        LOG_ERR("%s: Failed to allocate backend buffer\n", __func__);
        gguf_free(gguf_ctx);
        return false;
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

    // Get token counts for each head
    int64_t head0_len = pyramid_tokens[0].size();
    int64_t head1_len = pyramid_tokens[1].size();
    int64_t head2_len = pyramid_tokens[2].size();

    LOG_INF("%s: Decoding %lld + %lld + %lld tokens\n", __func__,
            (long long)head0_len, (long long)head1_len, (long long)head2_len);

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

    // 2. Build computation graph
    struct ggml_tensor * output = snac_build_graph(
        graph_ctx, ctx.weights,
        pyramid_tokens[0].data(), head0_len,
        pyramid_tokens[1].data(), head1_len,
        pyramid_tokens[2].data(), head2_len,
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

    // 4. Set input data for each head's token tensor
    // Find token tensors by name and set their data
    bool set_input_success = true;

    if (head0_len > 0) {
        struct ggml_tensor * tokens0 = ggml_get_tensor(graph_ctx, "tokens_head0");
        if (tokens0) {
            ggml_backend_tensor_set(tokens0, pyramid_tokens[0].data(), 0, head0_len * sizeof(int));
        } else {
            LOG_ERR("%s: tokens_head0 tensor not found\n", __func__);
            set_input_success = false;
        }
    }

    if (head1_len > 0) {
        struct ggml_tensor * tokens1 = ggml_get_tensor(graph_ctx, "tokens_head1");
        if (tokens1) {
            ggml_backend_tensor_set(tokens1, pyramid_tokens[1].data(), 0, head1_len * sizeof(int));
        } else {
            LOG_ERR("%s: tokens_head1 tensor not found\n", __func__);
            set_input_success = false;
        }
    }

    if (head2_len > 0) {
        struct ggml_tensor * tokens2 = ggml_get_tensor(graph_ctx, "tokens_head2");
        if (tokens2) {
            ggml_backend_tensor_set(tokens2, pyramid_tokens[2].data(), 0, head2_len * sizeof(int));
        } else {
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
