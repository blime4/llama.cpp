#ifdef GGML_USE_DLFA
#include <unistd.h>
#include "ggml.h"

#include "dl-fattn.cuh"
#include "../../../include/ggml-dlfa.h"
#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "../ggml-cuda/common.cuh"
#include "../ggml-cuda/convert.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <dldnn_ext.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <memory>
#include <vector>

// Per-runtime device buffer cache (shared by varlen/non-varlen forward paths).
struct flash_attn_device_layout_cache {
    int device = -1;
    size_t combined_cap = 0;             // capacity in ints for the combined buffer
    size_t host_combined_cap = 0;        // capacity for pinned host staging
    int * combined_ptr = nullptr;
    int * host_combined_ptr = nullptr;   // pinned host staging buffer
    int * cu_seqlens_ptr = nullptr;
    int * seqused_ptr = nullptr;
    int * block_table_ptr = nullptr;

    ~flash_attn_device_layout_cache() = default; // rely on process teardown to free
};

static inline bool ggml_dlfa_graphs_enabled() {
    const char * env = getenv("GGML_CUDA_DISABLE_GRAPHS");
    return !(env && strcmp(env, "1") == 0);
}

static inline cudnnDataType_t ggml_type_to_cudnn_type(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:  return CUDNN_DATA_HALF;
        case GGML_TYPE_F32:  return CUDNN_DATA_FLOAT;
        case GGML_TYPE_BF16: return CUDNN_DATA_BFLOAT16;
        default:
            GGML_ASSERT(false && "Unsupported data type for cuDNN");
            return CUDNN_DATA_FLOAT;
    }
}

// ============================================================================
// GGMLTensorDescriptor: Bridge between GGML and cuDNN tensor layouts
// ============================================================================

struct GGMLTensorDescriptor {
    cudnnTensorDescriptor_t desc;

    GGMLTensorDescriptor() {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    }

    ~GGMLTensorDescriptor() {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
    }

    void mha_set_from_ggml_qkv(const struct ggml_tensor* tensor, enum ggml_type target_type) const {
        // MHA : Q/K/V Tensors (zero-copy mapping for cudnnMHAForward):
        // --------------------------------------
        // GGML format: ne[0]=D, ne[1]=S, ne[2]=H, ne[3]=B (row-major, D is innermost)
        //   Physical offset (bytes): [d,s,h,b] = d*nb[0] + s*nb[1] + h*nb[2] + b*nb[3]
        //   Strides: [1, nb[1]/nb[0], nb[2]/nb[0], nb[3]/nb[0]]
        //
        // cuDNN MHA format: [B, S, H, D] (BSHD, row-major, D is innermost)
        //   Physical offset (bytes): [b,s,h,d] = b*stride[0] + s*stride[1] + h*stride[2] + d*stride[3]
        //   Mapping: b→nb[3], s→nb[1], h→nb[2], d→nb[0]
        //   Expected strides: [nb[3]/nb[0], nb[1]/nb[0], nb[2]/nb[0], 1]
        //
        cudnnDataType_t data_type = ggml_type_to_cudnn_type(target_type);

        const int64_t D = tensor->ne[0];  // head_dim
        const int64_t S = tensor->ne[1];  // seq_len
        const int64_t H = tensor->ne[2];  // num_heads
        const int64_t B = tensor->ne[3];  // batch_size

        // cuDNN MHA expects shape: [B, S, H, D] (BSHD format)
        int dims[4] = {
            static_cast<int>(B),
            static_cast<int>(S),
            static_cast<int>(H),
            static_cast<int>(D)
        };

        int strides[4] = {
            static_cast<int>(tensor->nb[3]/tensor->nb[0]),  // B stride
            static_cast<int>(tensor->nb[1]/tensor->nb[0]),  // S stride
            static_cast<int>(tensor->nb[2]/tensor->nb[0]),  // H stride
            1                                               // D stride
        };

        GGML_DL_FATTN_DEBUG_PRINT("GGMLTensorDescriptor: MHA Q/K/V GGML [D=%ld,S=%ld,H=%ld,B=%ld] -> cuDNN BSHD dims=[%d,%d,%d,%d] strides=[%d,%d,%d,%d]\n",
            D, S, H, B, dims[0], dims[1], dims[2], dims[3], strides[0], strides[1], strides[2], strides[3]);

        CUDNN_CHECK(cudnnSetTensorNdDescriptor(
            desc,
            data_type,
            4,
            dims,
            strides
        ));
    }

    void mha_set_from_ggml_output(const struct ggml_tensor* tensor, enum ggml_type target_type) const {
        // MHA : Output O Tensor (zero-copy mapping for cudnnMHAForward):
        // -----------------------------------------------------
        // GGML format: ne[0]=D, ne[1]=H, ne[2]=S, ne[3]=B (row-major, D is innermost)
        //   Physical offset (bytes): [d,h,s,b] = d*nb[0] + h*nb[1] + s*nb[2] + b*nb[3]
        //   Strides: [1, nb[1]/nb[0], nb[2]/nb[0], nb[3]/nb[0]]
        //
        // cuDNN MHA format: [B, S, H, D] (BSHD, same as Q/K/V format)
        //   Physical offset (bytes): [b,s,h,d] = b*stride[0] + s*stride[1] + h*stride[2] + d*stride[3]
        //   Mapping: b→nb[3], s→nb[2], h→nb[1], d→nb[0]
        //   Expected strides: [nb[3]/nb[0], nb[2]/nb[0], nb[1]/nb[0], 1]
        //
        cudnnDataType_t data_type = ggml_type_to_cudnn_type(target_type);

        const int64_t D = tensor->ne[0];  // head_dim
        const int64_t H = tensor->ne[1];  // num_heads
        const int64_t S = tensor->ne[2];  // seq_len
        const int64_t B = tensor->ne[3];  // batch_size

        // cuDNN MHA expects shape: [B, S, H, D] (BSHD format)
        int dims[4] = {
            static_cast<int>(B),
            static_cast<int>(S),
            static_cast<int>(H),
            static_cast<int>(D)
        };

        int strides[4] = {
            static_cast<int>(tensor->nb[3]/tensor->nb[0]),  // B stride
            static_cast<int>(tensor->nb[2]/tensor->nb[0]),  // S stride (from GGML dim 2)
            static_cast<int>(tensor->nb[1]/tensor->nb[0]),  // H stride (from GGML dim 1)
            1                                               // D stride
        };

        GGML_DL_FATTN_DEBUG_PRINT("GGMLTensorDescriptor: MHA OUTPUT GGML [D=%ld,H=%ld,S=%ld,B=%ld] -> cuDNN BSHD dims=[%d,%d,%d,%d] strides=[%d,%d,%d,%d]\n",
            D, H, S, B, dims[0], dims[1], dims[2], dims[3], strides[0], strides[1], strides[2], strides[3]);

        CUDNN_CHECK(cudnnSetTensorNdDescriptor(
            desc,
            data_type,
            4,
            dims,
            strides
        ));
    }

    // For seqlens_k: [B] -> expand to [1, 1, B] for cuDNN compatibility
    void set_from_dims_1d(int dim_0, cudnnDataType_t data_type) const {
        // cuDNN requires at least 3 dimensions, so expand [B] to [1, 1, B]
        // Validation expects: dims must be [1, 1, batch]
        int dims[3] = {1, 1, dim_0};
        int strides[3] = {dim_0, dim_0, 1};  // strides: [batch, batch, 1] to maintain correct layout

        GGML_DL_FATTN_DEBUG_PRINT("Setting 1D descriptor (expanded to 3D): dims=[1, 1, %d], strides=[%d, %d, 1]\n",
            dim_0, dim_0, dim_0);

        CUDNN_CHECK(cudnnSetTensorNdDescriptor(
            desc,
            data_type,
            3,
            dims,
            strides
        ));
    }

    // For ALiBi slopes: [batch_size, 1, num_heads]
    void set_from_dims_3d(int dim_0, int dim_1, int dim_2, enum ggml_type type) const {
        cudnnDataType_t data_type = ggml_type_to_cudnn_type(type);
        int dims[3] = {dim_0, dim_1, dim_2};
        int strides[3] = {
            dim_1 * dim_2,  // batch stride
            dim_2,          // seq_len stride (should be 1 for ALiBi)
            1               // num_heads stride
        };

        GGML_DL_FATTN_DEBUG_PRINT("Setting 3D descriptor: dims=[%d, %d, %d], strides=[%d, %d, %d]\n",
                      dims[0], dims[1], dims[2], strides[0], strides[1], strides[2]);

        CUDNN_CHECK(cudnnSetTensorNdDescriptor(
            desc,
            data_type,
            3,
            dims,
            strides
        ));
    }

    cudnnTensorDescriptor_t get() const { return desc; }
};

struct dldnn_mha_qkvo_pack {
    ggml_tensor * KQV;
    const ggml_tensor * Q;
    const ggml_tensor * K;
    const ggml_tensor * V;
    const void * q_data;
    const void * k_data;
    const void * v_data;
    GGMLTensorDescriptor q_desc;
    GGMLTensorDescriptor k_desc;
    GGMLTensorDescriptor v_desc;
    GGMLTensorDescriptor out_desc;

    explicit dldnn_mha_qkvo_pack(ggml_tensor * dst)
        : KQV(dst),
          Q(dst->src[0]),
          K(dst->src[1]),
          V(dst->src[2]),
          q_data(Q->data),
          k_data(K->data),
          v_data(V->data) {
        q_desc.mha_set_from_ggml_qkv(Q, Q->type);
        k_desc.mha_set_from_ggml_qkv(K, K->type);
        v_desc.mha_set_from_ggml_qkv(V, V->type);
        out_desc.mha_set_from_ggml_output(KQV, KQV->type);
    }

    dldnn_mha_qkvo_pack(const dldnn_mha_qkvo_pack &) = delete;
    dldnn_mha_qkvo_pack & operator=(const dldnn_mha_qkvo_pack &) = delete;
};


// MHA Forward implementation using cudnnMHAForward (non-varlen)
static void flash_attn_ext_dldnn_mha_forward(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_DL_FATTN_DEBUG_PRINT("\n========== ENTERING %s ==========\n", __FUNCTION__);

    const int id = ggml_cuda_get_device();
    ggml_cuda_pool_alloc<float> alibi_slopes_mem(ctx.pool(id));
    ggml_cuda_pool_alloc<float> softmax_lse_mem(ctx.pool(id));
    ggml_cuda_pool_alloc<uint8_t> workspace_mem(ctx.pool(id));

    // Persist layout buffers per runtime/mask so they are allocated once and reused across layers.

    const struct ggml_tensor * KQV  = dst;
    const struct ggml_tensor * Q    = dst->src[0];
    const struct ggml_tensor * K    = dst->src[1];
    const struct ggml_tensor * V    = dst->src[2];
    const struct ggml_tensor * mask = dst->src[3];
    ggml_dl::flash_attn_dlfa_runtime * runtime =
        mask ? static_cast<ggml_dl::flash_attn_dlfa_runtime *>(mask->extra) : nullptr;

    const char* env_dl_fattn_debug = getenv("GGML_DL_FATTN_DEBUG");
    ggml_flash_attn_mask_params mask_info{};
    if (runtime && runtime->has_mask_params) {
        mask_info = runtime->mask_params;
    } else {
        mask_info.present = mask != nullptr;
        mask_info.is_causal = false;
        mask_info.window_left = -1;
        mask_info.window_right = -1;
        mask_info.per_token_window = false;
        mask_info.multi_sequence = false;
        mask_info.has_alibi_bias = false;
    }
    if (!(runtime && runtime->has_mask_params)) {
        // Fallback to reading params already stored on the attn tensor (op params)
        ggml_flash_attn_ext_get_mask_params(dst, &mask_info);
    }

    const bool mask_supports_cudnn = mask_info.present && !mask_info.per_token_window && !mask_info.multi_sequence;
    if (!mask_supports_cudnn || mask_info.has_alibi_bias) {
        if (!mask_info.present) {
            GGML_DL_FATTN_DEBUG_PRINT("INFO: mask metadata not present\n");
        } else if (mask_info.per_token_window || mask_info.multi_sequence) {
            GGML_DL_FATTN_DEBUG_PRINT(
                "INFO: mask_metadata has unsupported configuration (per_token_window=%d, multi_sequence=%d)\n",
                mask_info.per_token_window, mask_info.multi_sequence);
        } else if (mask_info.has_alibi_bias) {
            GGML_DL_FATTN_DEBUG_PRINT("INFO: mask_metadata->has_alibi_bias is true\n");
        }
    }

    cudnnHandle_t cudnn_handle = ctx.cudnn_handle();
    CUDNN_CHECK(cudnnSetStream(cudnn_handle, ctx.stream()));

    dldnn_mha_qkvo_pack pack(dst);

    const int64_t seq_q = pack.Q->ne[1];
    const int64_t seq_k = pack.K->ne[1];

    // Track the real (unpadded) key length during decode per K buffer
    // IMPORTANT: Prefer using runtime->seqlen_k_real from host side if available,
    // to ensure synchronization between host and CUDA kernel across multi-round conversations.
    auto & decode_state = ggml_dl::flash_attn_ext_dldnn_decode_state().seq_k_real_by_k_ptr;
    const uintptr_t k_key = reinterpret_cast<uintptr_t>(pack.K->data);
    int64_t & seqlen_k_real = decode_state[k_key];

    // If runtime exists and has a valid seqlen_k_real, use it as the authoritative source
    // This ensures host-side accumulation (in set_flash_attn_runtime) is respected
    if (runtime && runtime->seqlen_k_real > 0) {
        // Initialize or sync decode_state with host-side value
        if (seqlen_k_real <= 0 || seqlen_k_real != runtime->seqlen_k_real) {
            seqlen_k_real = runtime->seqlen_k_real;
        }
    }

    GGMLTensorDescriptor k_desc_trunc;
    GGMLTensorDescriptor v_desc_trunc;
    const GGMLTensorDescriptor * k_desc_ptr = &pack.k_desc;
    const GGMLTensorDescriptor * v_desc_ptr = &pack.v_desc;

    if (mask != nullptr) {
        auto set_trunc_desc = [&](const ggml_tensor * tensor, GGMLTensorDescriptor & desc, int64_t trunc_seq_len) {
            const int64_t clamped_seq = std::min<int64_t>(trunc_seq_len, tensor->ne[1]);
            int dims[4] = {
                static_cast<int>(tensor->ne[3]),
                static_cast<int>(clamped_seq),
                static_cast<int>(tensor->ne[2]),
                static_cast<int>(tensor->ne[0])
            };
            int strides[4] = {
                static_cast<int>(tensor->nb[3]/tensor->nb[0]),
                static_cast<int>(tensor->nb[1]/tensor->nb[0]),
                static_cast<int>(tensor->nb[2]/tensor->nb[0]),
                1
            };
            CUDNN_CHECK(cudnnSetTensorNdDescriptor(
                desc.get(),
                ggml_type_to_cudnn_type(tensor->type),
                4,
                dims,
                strides));
        };

        const auto clamp_seq = [&](int64_t val) {
            return std::min<int64_t>(std::max<int64_t>(val, 1), seq_k);
        };

        if (seq_q != 1) {
            // Prefill: in multi-round conversations we want K length to keep
            // accumulating instead of resetting to the prompt length of the
            // current round.
            // IMPORTANT: If runtime->seqlen_k_real exists, it's already been accumulated
            // on the host side (in set_flash_attn_runtime), so use it directly.
            // Otherwise, fall back to local accumulation logic.
            if (runtime && runtime->seqlen_k_real > 0) {
                // Host side has already done the accumulation, use it
                seqlen_k_real = clamp_seq(runtime->seqlen_k_real);
            } else {
                // Fallback: local accumulation (shouldn't happen in normal flow)
                int64_t prev = seqlen_k_real;
                if (prev <= 0 || prev > seq_k) {
                    prev = 0;
                }
                int64_t candidate = prev + seq_q;
                seqlen_k_real = clamp_seq(candidate);
            }
            // printf(
            //     "for debug: prefill seqlen_k_real : %lld (runtime=%lld, seq_q=%lld, seq_k=%lld)\n",
            //     (long long) seqlen_k_real,
            //     (long long) (runtime ? runtime->seqlen_k_real : -1),
            //     (long long) seq_q,
            //     (long long) seq_k);
        } else {
            if (pack.K->ne[3] != 1) {
                GGML_ABORT(
                    "decode path currently supports batch=1 (got %lld); falling back to padded descriptors\n",
                    (long long) pack.K->ne[3]);
            }
            if (env_dl_fattn_debug && strcmp(env_dl_fattn_debug, "1") == 0) {
                const int64_t mask_sk = mask->ne[0];
                const size_t elem_size = ggml_type_size(mask->type);
                const size_t row_bytes = (size_t) mask_sk * elem_size;
                std::vector<uint8_t> row_buf(row_bytes);
                ggml_backend_tensor_get(mask, row_buf.data(), 0, row_bytes);
                float min_v = INFINITY, max_v = -INFINITY;
                for (int64_t c = 0; c < mask_sk; ++c) {
                    float val;
                    switch (mask->type) {
                        case GGML_TYPE_F32:  val = reinterpret_cast<float *>(row_buf.data())[c]; break;
                        case GGML_TYPE_F16:  val = ggml_fp16_to_fp32(reinterpret_cast<ggml_fp16_t *>(row_buf.data())[c]); break;
                        case GGML_TYPE_BF16: val = ggml_bf16_to_fp32(reinterpret_cast<ggml_bf16_t *>(row_buf.data())[c]); break;
                        default: val = -INFINITY; break;
                    }
                    min_v = std::min(min_v, val);
                    max_v = std::max(max_v, val);
                    if (c < 8) {
                        GGML_DL_FATTN_DEBUG_PRINT("mask row0 c=%lld val=%.3f\n", (long long) c, (double) val);
                    }
                }
            }
            // Decode: similar to prefill, prefer runtime->seqlen_k_real if available
            // Host side has already accumulated: prev_seqlen_k_real + seqlen_q
            if (runtime && runtime->seqlen_k_real > 0) {
                // Host side has already done the accumulation, use it
                seqlen_k_real = clamp_seq(runtime->seqlen_k_real);
            }
            GGML_DL_FATTN_DEBUG_PRINT(
                "decode seqlen_k_real updated for layer %p: new=%lld (runtime=%lld, max=%lld)\n",
                (const void *) pack.K->data,
                (long long) seqlen_k_real,
                (long long) (runtime ? runtime->seqlen_k_real : -1),
                (long long) seq_k);
        }

        set_trunc_desc(pack.K, k_desc_trunc, seqlen_k_real);
        set_trunc_desc(pack.V, v_desc_trunc, seqlen_k_real);
        k_desc_ptr = &k_desc_trunc;
        v_desc_ptr = &v_desc_trunc;

        // Sync back to decode_state so it persists for next call
        // (This ensures decode_state stays in sync with runtime->seqlen_k_real)
        decode_state[k_key] = seqlen_k_real;
    }

    float scale;
    float max_bias;
    float logit_softcap;
    memcpy(&scale,         ((const int32_t *) dst->op_params) + 0, sizeof(scale));
    memcpy(&max_bias,      ((const int32_t *) dst->op_params) + 1, sizeof(max_bias));
    memcpy(&logit_softcap, ((const int32_t *) dst->op_params) + 2, sizeof(logit_softcap));
    GGML_UNUSED(max_bias);
    GGML_UNUSED(logit_softcap);

    auto debug_desc = [&](const char *name, const GGMLTensorDescriptor *d) {
        if (!env_dl_fattn_debug || strcmp(env_dl_fattn_debug, "1") != 0) return;
        int dims[4], strides[4], nb_dims = 0;
        cudnnDataType_t dt;
        CUDNN_CHECK(cudnnGetTensorNdDescriptor(d->get(), 4, &dt, &nb_dims, dims, strides));
        GGML_DL_FATTN_DEBUG_PRINT("desc %s: dims=[%d,%d,%d,%d] strides=[%d,%d,%d,%d]\n",
            name, dims[0], dims[1], dims[2], dims[3], strides[0], strides[1], strides[2], strides[3]);
    };

    debug_desc("Q", &pack.q_desc);
    debug_desc("K_use", k_desc_ptr);
    debug_desc("V_use", v_desc_ptr);
    debug_desc("OUT", &pack.out_desc);

    GGML_DL_FATTN_DEBUG_PRINT("calling cudnnGetMHAForwardWorkspaceSize...\n");
    GGML_DL_FATTN_DEBUG_PRINT("INFO: params scale=%.6f max_bias=%.6f logit_softcap=%.6f\n", scale, max_bias, logit_softcap);
    GGML_DL_FATTN_DEBUG_PRINT("INFO: is_causal=%d, window_left=%d, window_right=%d\n", mask_info.is_causal, mask_info.window_left, mask_info.window_right);

    const int64_t batch = pack.Q->ne[3];
    const int64_t n_heads = pack.Q->ne[2];
    GGMLTensorDescriptor softmax_lse_desc;
    {
        int dims[3] = {
            (int) batch,
            (int) n_heads,
            (int) seq_q,
        };
        int strides[3] = {
            (int) (n_heads * seq_q),
            (int) seq_q,
            1,
        };
        CUDNN_CHECK(cudnnSetTensorNdDescriptor(
            softmax_lse_desc.get(),
            CUDNN_DATA_FLOAT,
            3,
            dims,
            strides));
    }

    float * softmax_lse = nullptr;
    softmax_lse = softmax_lse_mem.alloc(batch * n_heads * seq_q);

    size_t workspace_size = 0;
    CUDNN_CHECK(cudnnGetMHAForwardWorkspaceSize(
        cudnn_handle,
        pack.q_desc.get(), k_desc_ptr->get(), v_desc_ptr->get(),
        nullptr,
        pack.out_desc.get(), softmax_lse_desc.get(), nullptr,
        0.0f, scale,
        mask_info.is_causal,
        mask_info.window_left,
        mask_info.window_right,
        false,
        &workspace_size));

    void * workspace = nullptr;
    if (workspace_size > 0) {
        workspace = workspace_mem.alloc(workspace_size);
    }

    unsigned long long philox_seed = 0;
    unsigned long long philox_offset = 0;

    CUDNN_CHECK(cudnnMHAForward(
        cudnn_handle,
        pack.q_desc.get(), pack.q_data,
        k_desc_ptr->get(), pack.k_data,
        v_desc_ptr->get(), pack.v_data,
        nullptr, nullptr,
        pack.out_desc.get(), pack.KQV->data,
        softmax_lse_desc.get(), softmax_lse,
        nullptr, nullptr,
        0.0f,
        scale,
        mask_info.is_causal,
        mask_info.window_left,
        mask_info.window_right,
        false,
        &philox_seed, &philox_offset,
        workspace, workspace_size));

    CUDA_CHECK(cudaGetLastError());
}


static void flash_attn_ext_dldnn_mha_varlen_forward(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_DL_FATTN_DEBUG_PRINT("\n========== ENTERING %s ==========\n", __FUNCTION__);

    const int id = ggml_cuda_get_device();

    ggml_cuda_pool_alloc<float> softmax_lse_mem(ctx.pool(id));
    ggml_cuda_pool_alloc<uint8_t> workspace_mem(ctx.pool(id));
    const ggml_tensor * mask = dst->src[3];
    ggml_dl::flash_attn_dlfa_runtime * runtime =
        mask ? static_cast<ggml_dl::flash_attn_dlfa_runtime *>(mask->extra) : nullptr;
    ggml_flash_attn_mask_params mask_info{};
    if (runtime && runtime->has_mask_params) {
        mask_info = runtime->mask_params;
    } else {
        mask_info.present = mask != nullptr;
        mask_info.is_causal = false;
        mask_info.window_left = -1;
        mask_info.window_right = -1;
        mask_info.per_token_window = false;
        mask_info.multi_sequence = false;
        mask_info.has_alibi_bias = false;
    }

    {
        const ggml_tensor * Q    = dst->src[0];
        const ggml_tensor * K    = dst->src[1];
        const ggml_tensor * V    = dst->src[2];
        const ggml_tensor * mask  = dst->src[3];

        // parameters
        float scale;
        float max_bias;
        float logit_softcap;
        memcpy(&scale,         ((const int32_t *) dst->op_params) + 0, sizeof(scale));
        memcpy(&max_bias,      ((const int32_t *) dst->op_params) + 1, sizeof(max_bias));
        memcpy(&logit_softcap, ((const int32_t *) dst->op_params) + 2, sizeof(logit_softcap));
        GGML_UNUSED(max_bias);
        if (logit_softcap != 0.0f) {
            GGML_DL_FATTN_DEBUG_PRINT("INFO: logit_softcap not supported in varlen path yet (ignored)\n");
        }

        const int64_t D = Q->ne[0];
        const int64_t S_q = Q->ne[1];
        const int64_t H_q = Q->ne[2];
        const int64_t B = Q->ne[3];
        const int64_t S_k = K->ne[1];
        const int64_t H_k = K->ne[2];
        const int64_t H_v = V->ne[2];
        GGML_ASSERT(H_k == H_v && "GQA expects K and V to share head count");
        const int64_t H_kv = H_k;

        // total tokens and max seqlen
        const int page_block_size = 256;  // dldnn paged KV block size
        const int num_blocks_per_seq = (int)((S_k + page_block_size - 1) / page_block_size);

        const int max_seqlen_q = (int) S_q;
        const int max_seqlen_k = (int) S_k;
        const int num_blocks_per_seq_runtime = num_blocks_per_seq;

        // Device buffers should already be prepared and copied in flash_attn_ext_dldnn
        // (called before this function for async overlap). Just retrieve pointers from cache.
        int * cu_seqlens_q_dev = nullptr;
        int * seqused_k_dev    = nullptr;
        int * block_table_dev  = nullptr;

        {
            GGML_ASSERT((int) runtime->batch == (int) B);
            GGML_ASSERT((int) runtime->cu_seqlens_q.size() == (int) B + 1);
            GGML_ASSERT((int) runtime->seqused_k.size() == (int) B);

            // Device cache should already be initialized by flash_attn_ext_dldnn_prepare_varlen_buffers.
            // Multi-GPU runs may execute on a different device than the one that primed the cache,
            // so re-prepare if the device does not match.
            if (!runtime->device_cache || static_cast<flash_attn_device_layout_cache *>(runtime->device_cache)->device != id) {
                ggml_cuda_set_device(ctx.device);
                ggml_dl::flash_attn_ext_dldnn_prepare_varlen_buffers(ctx, dst);
            }
            GGML_ASSERT(runtime->device_cache != nullptr);
            flash_attn_device_layout_cache * cache = static_cast<flash_attn_device_layout_cache *>(runtime->device_cache);
            if (cache->device != id) {
                // As a last resort, rebuild the cache on this device.
                cache->device = id;
                cache->combined_cap = 0;
                cache->combined_ptr = cache->cu_seqlens_ptr = cache->seqused_ptr = cache->block_table_ptr = nullptr;
                ggml_dl::flash_attn_ext_dldnn_prepare_varlen_buffers(ctx, dst);
                cache = static_cast<flash_attn_device_layout_cache *>(runtime->device_cache);
            }

            // Retrieve device pointers (buffers already allocated and copied)
            cu_seqlens_q_dev = cache->cu_seqlens_ptr;
            seqused_k_dev    = cache->seqused_ptr;
            block_table_dev  = cache->block_table_ptr;

            GGML_ASSERT(cu_seqlens_q_dev != nullptr && "cu_seqlens_q_dev should be allocated");
            GGML_ASSERT(seqused_k_dev != nullptr && "seqused_k_dev should be allocated");
            GGML_ASSERT(block_table_dev != nullptr && "block_table_dev should be allocated");
        }

        // Descriptors per dldnn requirements (doc: docs/cudnnMHAVarlenForward-dldnn-requirements.md)
        struct ScopedDesc {
            cudnnTensorDescriptor_t desc;
            ScopedDesc() { CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc)); }
            ~ScopedDesc() { CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc)); }
        };
        ScopedDesc q_desc;
        ScopedDesc k_desc;
        ScopedDesc v_desc;
        ScopedDesc out_desc;
        ScopedDesc cu_seqlens_desc;
        ScopedDesc seqused_desc;
        ScopedDesc block_table_desc;
        ScopedDesc softmax_lse_desc;

        auto set_desc_3d_contig = [](cudnnTensorDescriptor_t d, int dim0, int dim1, int dim2, cudnnDataType_t dt) {
            int dims[3] = { dim0, dim1, dim2 };
            int strides[3] = { dim1 * dim2, dim2, 1 };
            CUDNN_CHECK(cudnnSetTensorNdDescriptor(d, dt, 3, dims, strides));
        };
        auto set_desc_4d_paged = [](cudnnTensorDescriptor_t d, int num_blocks, int page_block_size, int heads, int dim, cudnnDataType_t dt) {
            int dims[4] = { num_blocks, page_block_size, heads, dim };
            int strides[4] = {
                page_block_size * heads * dim, // block stride
                heads * dim,                   // page stride
                dim,                           // head stride
                1                              // dim stride
            };
            CUDNN_CHECK(cudnnSetTensorNdDescriptor(d, dt, 4, dims, strides));
        };

        // q/out 3D: [total_q_max, H_q, D]  (total_q_max = batch * max_seqlen_q)
        const int total_q_max = (int) (B * max_seqlen_q);
        set_desc_3d_contig(q_desc.desc, total_q_max, (int) H_q, (int) D, ggml_type_to_cudnn_type(Q->type));
        set_desc_3d_contig(out_desc.desc, total_q_max, (int) H_q, (int) D, ggml_type_to_cudnn_type(Q->type));

        const int num_blocks = (int)(B * num_blocks_per_seq_runtime);
        set_desc_4d_paged(k_desc.desc, num_blocks, page_block_size, (int) H_k, (int) D, ggml_type_to_cudnn_type(Q->type));
        set_desc_4d_paged(v_desc.desc, num_blocks, page_block_size, (int) H_k, (int) D, ggml_type_to_cudnn_type(Q->type));

        set_desc_3d_contig(cu_seqlens_desc.desc, 1, 1, (int)(B + 1), CUDNN_DATA_INT32);
        set_desc_3d_contig(seqused_desc.desc, 1, 1, (int) B, CUDNN_DATA_INT32);
        set_desc_3d_contig(block_table_desc.desc, 1, (int) B, num_blocks_per_seq_runtime, CUDNN_DATA_INT32);
        set_desc_3d_contig(softmax_lse_desc.desc, 1, (int) H_q, total_q_max, CUDNN_DATA_FLOAT);

        size_t workspace_bytes = 0;
        // printf("max_seqlen_q: %d, max_seqlen_k: %d\n", max_seqlen_q, max_seqlen_k);
        CUDNN_CHECK(cudnnGetMHAVarlenForwardWorkspaceSize(
            ctx.cudnn_handle(),
            q_desc.desc,
            k_desc.desc,
            v_desc.desc,
            cu_seqlens_desc.desc,
            /*cu_seqlens_k_desc*/ nullptr,
            seqused_desc.desc,
            block_table_desc.desc,
            /*alibi*/ nullptr,
            out_desc.desc,
            softmax_lse_desc.desc,
            /*p_desc*/ nullptr,
            max_seqlen_q,
            max_seqlen_k,
            /*dropout*/0.0f,
            scale,
            /*zero_tensors*/ false,
            mask_info.is_causal,
            mask_info.window_left,
            mask_info.window_right,
            /*return_softmax*/ false,
            &workspace_bytes));

        void * workspace = nullptr;
        if (workspace_bytes > 0) {
            workspace = workspace_mem.alloc(workspace_bytes);
        }

        const int total_q_softmax = (int)(B * max_seqlen_q);
        // softmax_lse must always be float regardless of Q/K/V dtype
        const size_t lse_bytes = (size_t) H_q * total_q_softmax * sizeof(float);
        float * softmax_lse = softmax_lse_mem.alloc(lse_bytes / sizeof(float));

        CUDNN_CHECK(cudnnMHAVarlenForward(
            ctx.cudnn_handle(),
            q_desc.desc, Q->data,
            k_desc.desc, K->data,
            v_desc.desc, V->data,
            cu_seqlens_desc.desc, cu_seqlens_q_dev,
            /*cu_seqlens_k_desc*/ nullptr, nullptr,
            seqused_desc.desc, seqused_k_dev,
            block_table_desc.desc, block_table_dev,
            /*alibi*/ nullptr, nullptr,
            out_desc.desc, dst->data,
            softmax_lse_desc.desc, softmax_lse,
            /*p_desc*/ nullptr, nullptr,
            max_seqlen_q, max_seqlen_k,
            0.0f, scale,
            false, mask_info.is_causal, mask_info.window_left, mask_info.window_right,
            false,
            nullptr, nullptr,
            workspace, workspace_bytes));

        CUDA_CHECK(cudaGetLastError());
        return;

    }
}


namespace ggml_dl {

bool flash_attn_dldnn_available(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);

    const char *env_force_no_dlfa = getenv("GGML_FORCE_NO_DLFA");
    if (env_force_no_dlfa != nullptr && strcmp(env_force_no_dlfa, "1") == 0) {
        return false;
    }

    const struct ggml_tensor * Q = dst->src[0];
    const struct ggml_tensor * K = dst->src[1];
    const struct ggml_tensor * V = dst->src[2];
    const struct ggml_tensor * mask = dst->src[3];
    ggml_dl::flash_attn_dlfa_runtime * runtime =
        mask ? static_cast<ggml_dl::flash_attn_dlfa_runtime *>(mask->extra) : nullptr;

    // Extract parameters
    const int64_t hsk = Q->ne[0];       // head size for K/Q
    // const int64_t hsv = V->ne[0];       // head size for V
    // const int64_t kv  = K->ne[1];       // sequence length
    // const int64_t nb  = Q->ne[3];       // batch size

    const int64_t n_head_q = Q->ne[2];  // Number of query heads
    const int64_t n_head_k = K->ne[2];  // Number of key heads
    const int64_t n_head_v = V->ne[2];  // Number of value heads

    const int64_t gqa_ratio = n_head_k > 0 ? n_head_q / n_head_k : 1;
    // const bool has_gqa = (gqa_ratio > 1);
    const bool has_mask = (mask != nullptr);

    // Extract max_bias and logit_softcap from op_params
    float max_bias;
    float logit_softcap;
    memcpy(&max_bias, ((const int32_t *) dst->op_params) + 1, sizeof(max_bias));
    memcpy(&logit_softcap, ((const int32_t *) dst->op_params) + 2, sizeof(logit_softcap));

    const bool has_alibi = (max_bias > 0.0f);
    // const bool has_softcap = (logit_softcap != 0.0f);
    ggml_flash_attn_mask_params mask_info{};
    bool have_mask_params = false;
    if (runtime && runtime->has_mask_params) {
        mask_info = runtime->mask_params;
        have_mask_params = true;
    } else {
        mask_info.present = mask != nullptr;
        mask_info.is_causal = false;
        mask_info.window_left = -1;
        mask_info.window_right = -1;
        mask_info.per_token_window = false;
        mask_info.multi_sequence = false;
        mask_info.has_alibi_bias = false;
    }
    if (!have_mask_params) {
        have_mask_params = ggml_flash_attn_ext_get_mask_params(dst, &mask_info);
    }

    // Check for GQA (Grouped Query Attention) support
    if (n_head_q != n_head_k || n_head_k != n_head_v) {
        // This is GQA configuration - now supported by SDK
        GGML_DL_FATTN_DEBUG_PRINT("DLDNN Flash Attention: GQA configuration detected (Q heads: %ld, K heads: %ld, V heads: %ld, ratio: %ld)\n",
                      n_head_q, n_head_k, n_head_v, gqa_ratio);
        // Verify that head counts are compatible
        if (n_head_q % n_head_k != 0 || n_head_k != n_head_v) {
            GGML_LOG_WARN("DLDNN Flash Attention: Invalid GQA configuration - Q heads must be divisible by K heads, and K/V heads must match\n");
            return false;
        }
    }

    if (!has_alibi) {
        if (!has_mask) {
            // Nothing special to encode, allowed.
        } else if (!mask_info.present) {
            GGML_DL_FATTN_DEBUG_PRINT("mask metadata missing or not present\n");
            return false;
        } else if (mask_info.per_token_window || mask_info.multi_sequence) {
            GGML_DL_FATTN_DEBUG_PRINT(
                "mask metadata unsupported (per_token_window=%d, multi_sequence=%d).\n",
                mask_info.per_token_window, mask_info.multi_sequence);
            return false;
        }
    }

    // Check basic requirements
    if (hsk > 288) { // adapt from flash-attn
        GGML_LOG_WARN("DLDNN is not available for ne[0] %ld\n", hsk);
        return false;
    }

    return true;
}

// Helper function to prepare device buffers for varlen forward (called early for async overlap)
void ggml_dl::flash_attn_ext_dldnn_prepare_varlen_buffers(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int id = ggml_cuda_get_device();
    const ggml_tensor * mask = dst->src[3];
    GGML_ASSERT(mask);
    ggml_dl::flash_attn_dlfa_runtime * runtime = static_cast<ggml_dl::flash_attn_dlfa_runtime *>(mask->extra);

    const ggml_tensor * K = dst->src[1];
    const int64_t B = dst->src[0]->ne[3];
    const int64_t S_k = K->ne[1];
    const int page_block_size = 256;
    const int max_num_seqs = 1;

    const bool runtime_has_layout =
        runtime && runtime->block_table_stride > 0 && runtime->seqlen_q > 0 && runtime->seqlen_k_real > 0 && runtime->batch > 0;
    GGML_ASSERT(runtime_has_layout);

    const int num_blocks_per_seq = runtime->block_table_stride;

    GGML_ASSERT((int) runtime->batch == (int) B);
    GGML_ASSERT((int) runtime->cu_seqlens_q.size() == (int) B + 1);
    GGML_ASSERT((int) runtime->seqused_k.size() == (int) B);

    const size_t cu_bytes = runtime->cu_seqlens_q.size() * sizeof(int32_t);
    const size_t su_bytes = runtime->seqused_k.size() * sizeof(int32_t);
    const size_t bt_elems = (size_t)((runtime->seqlen_k_real + page_block_size - 1) / page_block_size);

    const size_t bt_bytes = bt_elems * sizeof(int32_t);

    // Ensure device-side cached buffers exist (per runtime/mask) and are large enough.
    if (!runtime->device_cache) {
        runtime->device_cache = new flash_attn_device_layout_cache();
        flash_attn_device_layout_cache * cache = static_cast<flash_attn_device_layout_cache *>(runtime->device_cache);
        cache->device = id;
    }
    flash_attn_device_layout_cache * cache = static_cast<flash_attn_device_layout_cache *>(runtime->device_cache);
    if (cache->device != id) {
        cache->device = id;
        cache->combined_cap = 0;
        cache->combined_ptr = cache->cu_seqlens_ptr = cache->seqused_ptr = cache->block_table_ptr = nullptr;
    }

    const size_t total_elems = runtime->cu_seqlens_q.size() + runtime->seqused_k.size() + num_blocks_per_seq;
    if (cache->combined_cap < total_elems) {
        if (cache->combined_ptr) {
            ggml_cuda_set_device(id);
            CUDA_CHECK(cudaFree(cache->combined_ptr));
        }
        ggml_cuda_set_device(id);
        CUDA_CHECK(cudaMalloc(&cache->combined_ptr, total_elems * sizeof(int)));
        cache->combined_cap = total_elems;
    }
    if (cache->host_combined_cap < total_elems) {
        if (cache->host_combined_ptr) {
            CUDA_CHECK(cudaFreeHost(cache->host_combined_ptr));
        }
        CUDA_CHECK(cudaHostAlloc(&cache->host_combined_ptr, total_elems * sizeof(int), cudaHostAllocDefault));
        cache->host_combined_cap = total_elems;
    }

    cache->cu_seqlens_ptr = cache->combined_ptr;
    cache->seqused_ptr = cache->cu_seqlens_ptr + runtime->cu_seqlens_q.size();
    cache->block_table_ptr = cache->seqused_ptr + runtime->seqused_k.size();

    // Start async copies early for better overlap with subsequent operations
    int * host_ptr = cache->host_combined_ptr;
    memcpy(host_ptr, runtime->cu_seqlens_q.data(), cu_bytes);
    host_ptr += runtime->cu_seqlens_q.size();
    memcpy(host_ptr, runtime->seqused_k.data(), su_bytes);
    host_ptr += runtime->seqused_k.size();
    memcpy(host_ptr, runtime->block_table.data(), bt_bytes);

    CUDA_CHECK(cudaMemcpyAsync(cache->combined_ptr, cache->host_combined_ptr, total_elems * sizeof(int), cudaMemcpyHostToDevice, ctx.stream()));
}

void flash_attn_ext_dldnn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {

    ggml_cuda_set_device(ctx.device);

    cudnnHandle_t cudnn_handle = ctx.cudnn_handle();
    CUDNN_CHECK(cudnnSetStream(cudnn_handle, ctx.stream()));

    float max_bias;
    memcpy(&max_bias,      ((const int32_t *) dst->op_params) + 1, sizeof(max_bias));

    bool has_alibi = (max_bias > 0.0f);
    if (has_alibi) {
        GGML_ABORT("alibi not supported yet");
    }

    // Use varlen path when CUDA graphs are enabled, otherwise fall back to legacy cudnnMHAForward.
    if (ggml_dlfa_graphs_enabled()) {
        if (dst->src[0]->ne[1] == 1) { // cudnnMHAVarlenForward prefill is not supported yet.
            flash_attn_ext_dldnn_mha_varlen_forward(ctx, dst);
        } else {
            flash_attn_ext_dldnn_mha_forward(ctx, dst);
        }
    } else {
        flash_attn_ext_dldnn_mha_forward(ctx, dst);
    }
}

} // namespace ggml_dl

// C linkage wrapper for proc_address registration
extern "C" void ggml_dl_flash_attn_ext_dldnn_prepare_varlen_buffers(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_dl::flash_attn_ext_dldnn_prepare_varlen_buffers(ctx, dst);
}

#endif // GGML_USE_DLFA
