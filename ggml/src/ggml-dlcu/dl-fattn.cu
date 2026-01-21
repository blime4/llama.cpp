#ifdef GGML_USE_DLFA
#include <unistd.h>
#include "ggml.h"

#include "dl-fattn.cuh"
#include "../../../include/ggml-dlfa.h"
#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "../ggml-cuda/common.cuh"
#include "../ggml-cuda/convert.cuh"

// Include llama types for the callback function
#include "../../../include/llama.h"

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
#include <mutex>
#include <unordered_map>

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

// Varlen data structure that holds host-side vectors and device cache map
struct flash_attn_varlen_data {
    // Host-side vectors
    std::vector<int32_t> cu_seqlens_q;
    std::vector<int32_t> seqused_k;
    std::vector<int32_t> block_table;

    // Per-device cache map
    std::unordered_map<int, flash_attn_device_layout_cache*> device_cache_map;

    ~flash_attn_varlen_data() {
        // Clean up device caches
        for (auto & pair : device_cache_map) {
            delete pair.second;
        }
    }
};

// ============================================================================
// NEW: Host data structure for varlen parameters (shared across all layers)
// This structure stores computed host-side data that is shared by all 48 FA layers
// Key: mask pointer (all layers share the same mask tensor)
// ============================================================================
struct flash_attn_varlen_host_data {
    std::vector<int32_t> cu_seqlens_q;   // [0, seqlen_q_for_cu] - cumulative sequence lengths
    std::vector<int32_t> seqused_k;      // [seqlen_k_real] - actual used key lengths
    std::vector<int32_t> block_table;    // block mapping for paged KV cache
    int seqlen_q;                         // actual seqlen_q (1 for decode, prefill seqlen_q for prefill)
    int seqlen_q_for_cu;                  // seqlen_q for cu_seqlens_q (prefill seqlen_q, kept constant in decode)
    int seqlen_k_real;                    // accumulated seqlen_k_real
    int block_table_stride;               // number of blocks per sequence
    bool is_causal;                       // whether to use causal attention (always true for autoregressive models)
    bool is_valid;                        // whether the data is valid (set by set_flash_attn_runtime)
};

// Global varlen_data map for storing varlen metadata keyed by flash attention operation tensor pointer
// This map allows multiple flash attention operations to each have their own varlen_data
static std::mutex g_varlen_data_mutex;
static std::unordered_map<void*, flash_attn_varlen_data*> g_varlen_data_map;

// Global map: mask pointer -> host data (shared across all layers)
static std::mutex g_varlen_host_data_mutex;
static std::unordered_map<void*, flash_attn_varlen_host_data*> g_varlen_host_data_map;

// Global maps for accumulation logic (restoring baseline behavior)
// These maps store per-attention-tensor state that persists across calls
static std::mutex g_accumulation_mutex;
// prev_seqlen_k_real: stores the accumulated seqlen_k_real from previous calls
// This is used to implement: seqlen_k_real = prev_seqlen_k_real + seqlen_q
static std::unordered_map<void*, int> g_prev_seqlen_k_real_map;
// prefill_seqlen_q: stores the seqlen_q from the prefill phase
// This is used to keep cu_seqlens_q constant across decode steps
static std::unordered_map<void*, int> g_prefill_seqlen_q_map;

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


// Forward declaration
static bool infer_is_causal_from_mask_data(ggml_tensor * mask);

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

    const char* env_dl_fattn_debug = getenv("GGML_DL_FATTN_DEBUG");
    ggml_flash_attn_mask_params mask_info{};
    bool have_mask_params = ggml_flash_attn_ext_get_mask_params(dst, &mask_info);

    const bool mask_supports_cudnn = mask_info.present && !mask_info.per_token_window && !mask_info.multi_sequence;
    if (!mask_supports_cudnn || mask_info.has_alibi_bias) {
        if (!mask_info.present) {
            GGML_DL_FATTN_DEBUG_PRINT("INFO: mask metadata not present\n");
            // When mask metadata is not present, infer is_causal from mask data
            if (mask != nullptr) {
                mask_info.is_causal = infer_is_causal_from_mask_data(const_cast<ggml_tensor*>(mask));
                GGML_DL_FATTN_DEBUG_PRINT("INFO: inferred is_causal=%d from mask data\n", mask_info.is_causal);
                // Set window parameters for causal attention
                // window_left=-1 means no limit on left (can see all past tokens)
                // window_right=0 means cannot see future tokens
                if (mask_info.is_causal) {
                    mask_info.window_left = -1;
                    mask_info.window_right = 0;
                }
            } else {
                // No mask tensor means non-causal (full attention)
                mask_info.is_causal = false;
                // For non-causal attention, both window_left and window_right should be -1
                // This allows each query token to attend to all key tokens
                mask_info.window_left = -1;
                mask_info.window_right = -1;
                GGML_DL_FATTN_DEBUG_PRINT("INFO: no mask tensor, setting is_causal=false, window_left=-1, window_right=-1\n");
            }
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
    // IMPORTANT: Prefer using mask_info.seqlen_k_real from host side if available,
    // to ensure synchronization between host and CUDA kernel across multi-round conversations.
    auto & decode_state = ggml_dl::flash_attn_ext_dldnn_decode_state().seq_k_real_by_k_ptr;
    const uintptr_t k_key = reinterpret_cast<uintptr_t>(pack.K->data);
    int64_t & seqlen_k_real = decode_state[k_key];

    // If mask_params exists and has a valid seqlen_k_real, use it as the authoritative source
    // This ensures host-side accumulation (in set_flash_attn_runtime) is respected
    if (have_mask_params && mask_info.seqlen_k_real > 0) {
        // Initialize or sync decode_state with host-side value
        if (seqlen_k_real <= 0 || seqlen_k_real != mask_info.seqlen_k_real) {
            seqlen_k_real = mask_info.seqlen_k_real;
        }
    }

    GGMLTensorDescriptor k_desc_trunc;
    GGMLTensorDescriptor v_desc_trunc;
    const GGMLTensorDescriptor * k_desc_ptr = &pack.k_desc;
    const GGMLTensorDescriptor * v_desc_ptr = &pack.v_desc;

    auto infer_seq_k_from_mask = [&](const ggml_tensor * tensor, int64_t token_idx) -> int64_t {
        const int64_t mask_sk = tensor->ne[0];
        const size_t elem_size = ggml_type_size(tensor->type);
        const size_t row_bytes = (size_t) mask_sk * elem_size;
        std::vector<uint8_t> row_buf(row_bytes);
        ggml_backend_tensor_get(tensor, row_buf.data(), token_idx * tensor->nb[1], row_bytes);

        auto read_val = [&](int64_t col) {
            switch (tensor->type) {
                case GGML_TYPE_F32:
                    return reinterpret_cast<float *>(row_buf.data())[col];
                case GGML_TYPE_F16:
                    return ggml_fp16_to_fp32(reinterpret_cast<ggml_fp16_t *>(row_buf.data())[col]);
                case GGML_TYPE_BF16:
                    return ggml_bf16_to_fp32(reinterpret_cast<ggml_bf16_t *>(row_buf.data())[col]);
                default:
                    return -INFINITY;
            }
        };

        int64_t last_valid = -1;
        for (int64_t c = mask_sk - 1; c >= 0; --c) {
            const float val = read_val(c);
            if (!(std::isinf(val) && val < 0)) {
                last_valid = c;
                break;
            }
        }
        return last_valid + 1;
    };

    if (mask != nullptr) {
        auto set_trunc_desc = [&](const ggml_tensor * tensor, GGMLTensorDescriptor & desc, int64_t trunc_seq_len) {
            const int64_t clamped_seq = std::min<int64_t>(trunc_seq_len, tensor->ne[1]);
            GGML_ASSERT(clamped_seq > 0);
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
            // IMPORTANT: If mask_info.seqlen_k_real exists, it's already been accumulated
            // on the host side (in set_flash_attn_runtime), so use it directly.
            // Otherwise, fall back to local accumulation logic.
            if (have_mask_params && mask_info.seqlen_k_real > 0) {
                // Host side has already done the accumulation, use it
                seqlen_k_real = clamp_seq(mask_info.seqlen_k_real);
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
            //     "for debug: prefill seqlen_k_real : %lld (mask_params=%lld, seq_q=%lld, seq_k=%lld)\n",
            //     (long long) seqlen_k_real,
            //     (long long) (have_mask_params ? mask_info.seqlen_k_real : -1),
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
            // Decode: similar to prefill, prefer mask_info.seqlen_k_real if available
            // Host side has already accumulated: prev_seqlen_k_real + seqlen_q
            if (have_mask_params && mask_info.seqlen_k_real > 0) {
                // Host side has already done the accumulation, use it
                seqlen_k_real = clamp_seq(mask_info.seqlen_k_real);
            } else {
                GGML_ASSERT(GGML_IS_TEST && "only happen in ut test.");
                // Fallback: local accumulation (shouldn't happen in normal flow)
                int64_t inferred = infer_seq_k_from_mask(mask, /*token_idx*/ 0);
                int64_t prev = seqlen_k_real;
                if (prev <= 0 || prev > seq_k) {
                    prev = seq_q;
                }
                int64_t candidate = prev + seq_q;
                if (inferred > 0 && inferred <= seq_k) {
                    candidate = std::max<int64_t>(candidate, inferred);
                }
                seqlen_k_real = clamp_seq(candidate);
            }
            GGML_DL_FATTN_DEBUG_PRINT(
                "decode seqlen_k_real updated for layer %p: new=%lld (mask_params=%lld, max=%lld)\n",
                (const void *) pack.K->data,
                (long long) seqlen_k_real,
                (long long) (have_mask_params ? mask_info.seqlen_k_real : -1),
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

// Helper function to infer actual sequence length from mask data
// The mask is structured as:
// - Row i: mask for token i (can attend to positions 0 to i for causal attention)
// - Rows beyond actual token count are filled with -inf
// This function finds the last valid row and returns the sequence length
static int64_t infer_from_mask_last_row(const ggml_tensor * mask, int64_t hint_token_idx) {
    // If mask is a copy/cast operation (F16), trace back to the original F32 tensor
    // This is necessary because the cast operation hasn't been executed yet during prepare_varlen_buffers
    // ggml_cast uses GGML_OP_CPY internally
    const ggml_tensor * mask_to_read = mask;
    if (mask->op == GGML_OP_CPY && mask->src[0] != nullptr && mask->src[0]->type == GGML_TYPE_F32) {
        mask_to_read = mask->src[0];
    }

    const int64_t mask_sk = mask_to_read->ne[0];
    const int64_t mask_sq = mask_to_read->ne[1];
    const size_t elem_size = ggml_type_size(mask_to_read->type);
    const size_t row_bytes = (size_t) mask_sk * elem_size;
    std::vector<uint8_t> row_buf(row_bytes);

    auto read_val = [&](int64_t col) {
        switch (mask_to_read->type) {
            case GGML_TYPE_F32:
                return reinterpret_cast<float *>(row_buf.data())[col];
            case GGML_TYPE_F16:
                return ggml_fp16_to_fp32(reinterpret_cast<ggml_fp16_t *>(row_buf.data())[col]);
            case GGML_TYPE_BF16:
                return ggml_bf16_to_fp32(reinterpret_cast<ggml_bf16_t *>(row_buf.data())[col]);
            default:
                return -INFINITY;
        }
    };

    // Find the last valid row by scanning from the end
    // A valid row has 0.0 at position 0 (can attend to first token)
    // An invalid row (beyond actual token count) has -inf at position 0
    int64_t last_valid_row = -1;
    for (int64_t row = mask_sq - 1; row >= 0; --row) {
        ggml_backend_tensor_get(mask_to_read, row_buf.data(), row * mask_to_read->nb[1], row_bytes);
        const float val = read_val(0);
        // A valid mask row should have 0.0 at position 0 (can attend to first token)
        if (val == 0.0f) {
            last_valid_row = row;
            break;
        }
    }

    if (last_valid_row < 0) {
        // No valid row found, return -1 to indicate failure
        return -1;
    }

    // Read the last valid row to find the sequence length
    ggml_backend_tensor_get(mask_to_read, row_buf.data(), last_valid_row * mask_to_read->nb[1], row_bytes);

    // Find the last valid (non-inf) position in this row
    int64_t last_valid_col = -1;
    for (int64_t c = mask_sk - 1; c >= 0; --c) {
        const float val = read_val(c);
        if (!(std::isinf(val) && val < 0)) {
            last_valid_col = c;
            break;
        }
    }

    // Debug: print first few values of the last valid row
    // fprintf(stderr, "[DEBUG] infer_from_mask_last_row: mask_to_read=%p, mask_sq=%lld, last_valid_row=%lld, first 4 values: ",
    //         (void*)mask_to_read, (long long)mask_sq, (long long)last_valid_row);
    // for (int64_t c = 0; c < std::min((int64_t)4, mask_sk); ++c) {
    //     fprintf(stderr, "%.2f ", (double)read_val(c));
    // }
    // fprintf(stderr, "\n");

    // Task 4 debug prints: infer_from_mask_last_row key variables
    GGML_DL_FATTN_DEBUG_PRINT("[TASK4-DEBUG] infer_from_mask_last_row: mask_to_read=%p, mask_sk=%lld, mask_sq=%lld\n",
            (void*)mask_to_read, (long long)mask_sk, (long long)mask_sq);
    GGML_DL_FATTN_DEBUG_PRINT("[TASK4-DEBUG] infer_from_mask_last_row: hint_token_idx=%lld, last_valid_row=%lld, last_valid_col=%lld\n",
            (long long)hint_token_idx, (long long)last_valid_row, (long long)last_valid_col);
    GGML_DL_FATTN_DEBUG_PRINT("[TASK4-DEBUG] infer_from_mask_last_row: returning seqlen_k_real=%lld\n",
            (long long)(last_valid_col + 1));

    return last_valid_col + 1;
}

// Helper function to check if mask data is valid (properly initialized)
// Returns true if mask data is valid, false if it appears uninitialized
// During warmup, mask tensor may be allocated but not initialized with actual data
// A valid mask should have:
// 1. The row at index (seqlen_q - 1) should have 0.0 at position 0 (can attend to first token)
// 2. The row at index (seqlen_q - 1) should have at least seqlen_q non-inf values (causal mask)
static bool is_mask_data_valid(const ggml_tensor * mask, int seqlen_q) {
    if (!mask) {
        fprintf(stderr, "[DEBUG] is_mask_data_valid: no mask tensor\n");
        return false;
    }

    // If mask is a copy/cast operation (F16), trace back to the original F32 tensor
    const ggml_tensor * mask_to_read = mask;
    fprintf(stderr, "[DEBUG] is_mask_data_valid: mask=%p, op=%d, type=%d\n", (void*)mask, mask->op, mask->type);
    if (mask->src[0] != nullptr) {
        fprintf(stderr, "[DEBUG] is_mask_data_valid: mask->src[0]=%p, op=%d, type=%d\n",
                (void*)mask->src[0], mask->src[0]->op, mask->src[0]->type);
    }

    // Trace back through copy operations to find the original source
    while (mask_to_read->op == GGML_OP_CPY && mask_to_read->src[0] != nullptr) {
        fprintf(stderr, "[DEBUG] is_mask_data_valid: tracing back from %p (op=%d) to %p (op=%d)\n",
                (void*)mask_to_read, mask_to_read->op, (void*)mask_to_read->src[0], mask_to_read->src[0]->op);
        mask_to_read = mask_to_read->src[0];
    }

    fprintf(stderr, "[DEBUG] is_mask_data_valid: final mask_to_read=%p, op=%d, type=%d\n",
            (void*)mask_to_read, mask_to_read->op, mask_to_read->type);

    const int64_t mask_sk = mask_to_read->ne[0];
    const int64_t mask_sq = mask_to_read->ne[1];

    if (mask_sk <= 0 || mask_sq <= 0) {
        fprintf(stderr, "[DEBUG] is_mask_data_valid: invalid mask dimensions (mask_sk=%lld, mask_sq=%lld)\n",
                (long long)mask_sk, (long long)mask_sq);
        return false;
    }

    // Check if the mask tensor has a valid buffer
    if (!mask_to_read->buffer) {
        fprintf(stderr, "[DEBUG] is_mask_data_valid: mask has no buffer\n");
        return false;
    }

    // Debug: print buffer info
    const char * buft_name = mask_to_read->buffer ? ggml_backend_buffer_name(mask_to_read->buffer) : "null";
    fprintf(stderr, "[DEBUG] is_mask_data_valid: mask buffer name=%s, data=%p\n", buft_name, mask_to_read->data);

    // Read the row at index (seqlen_q - 1) to check if it's properly initialized
    // For a valid causal mask, this row should have 0.0 at position 0 and at least seqlen_q non-inf values
    const int64_t target_row = seqlen_q - 1;
    if (target_row < 0 || target_row >= mask_sq) {
        fprintf(stderr, "[DEBUG] is_mask_data_valid: target_row=%lld out of range [0, %lld)\n",
                (long long)target_row, (long long)mask_sq);
        return false;
    }

    const size_t elem_size = ggml_type_size(mask_to_read->type);
    const size_t row_bytes = (size_t) mask_sk * elem_size;
    std::vector<uint8_t> row_buf(row_bytes);

    auto read_val = [&](int64_t col) {
        switch (mask_to_read->type) {
            case GGML_TYPE_F32:
                return reinterpret_cast<float *>(row_buf.data())[col];
            case GGML_TYPE_F16:
                return ggml_fp16_to_fp32(reinterpret_cast<ggml_fp16_t *>(row_buf.data())[col]);
            case GGML_TYPE_BF16:
                return ggml_bf16_to_fp32(reinterpret_cast<ggml_bf16_t *>(row_buf.data())[col]);
            default:
                return -INFINITY;
        }
    };

    // Read the target row
    ggml_backend_tensor_get(mask_to_read, row_buf.data(), target_row * mask_to_read->nb[1], row_bytes);

    // Check if the first value is 0.0 (can attend to first token)
    // Use tolerance-based comparison to handle -0.0 and small floating point errors
    const float first_val = read_val(0);
    const float tolerance = 1e-6f;

    // Debug: print mask dimensions and first few values of multiple rows
    fprintf(stderr, "[DEBUG] is_mask_data_valid: seqlen_q=%d, mask_sq=%lld, mask_sk=%lld, nb[0]=%zu, nb[1]=%zu\n",
            seqlen_q, (long long)mask_sq, (long long)mask_sk, mask_to_read->nb[0], mask_to_read->nb[1]);
    fprintf(stderr, "[DEBUG] is_mask_data_valid: target_row=%lld, first 8 values: ", (long long)target_row);
    for (int64_t c = 0; c < std::min((int64_t)8, mask_sk); ++c) {
        fprintf(stderr, "%.4f ", (double)read_val(c));
    }
    fprintf(stderr, "\n");

    // Also read row 0 to see the pattern
    ggml_backend_tensor_get(mask_to_read, row_buf.data(), 0, row_bytes);
    fprintf(stderr, "[DEBUG] is_mask_data_valid: row 0, first 8 values: ");
    for (int64_t c = 0; c < std::min((int64_t)8, mask_sk); ++c) {
        fprintf(stderr, "%.4f ", (double)read_val(c));
    }
    fprintf(stderr, "\n");

    // Re-read target row for validation
    ggml_backend_tensor_get(mask_to_read, row_buf.data(), target_row * mask_to_read->nb[1], row_bytes);

    if (std::fabs(first_val) > tolerance) {
        fprintf(stderr, "[DEBUG] is_mask_data_valid: row %lld first value is %.6f (expected ~0.0), mask likely uninitialized, seqlen_q=%d\n",
                (long long)target_row, first_val, seqlen_q);
        return false;
    }

    // Count non-inf values in this row
    // For a valid causal mask at row (seqlen_q - 1), there should be at least seqlen_q non-inf values
    int64_t non_inf_count = 0;
    for (int64_t c = 0; c < mask_sk; ++c) {
        const float val = read_val(c);
        if (!(std::isinf(val) && val < 0)) {
            non_inf_count++;
        }
    }

    // For causal attention, the row at index (seqlen_q - 1) should have at least seqlen_q non-inf values
    // This is because token at position (seqlen_q - 1) can attend to all previous tokens (0 to seqlen_q - 1)
    if (non_inf_count < seqlen_q) {
        fprintf(stderr, "[DEBUG] is_mask_data_valid: row %lld has only %lld non-inf values (expected at least %d), mask likely uninitialized\n",
                (long long)target_row, (long long)non_inf_count, seqlen_q);
        return false;
    }

    // fprintf(stderr, "[DEBUG] is_mask_data_valid: mask is valid, row %lld has %lld non-inf values (seqlen_q=%d)\n",
    //         (long long)target_row, (long long)non_inf_count, seqlen_q);
    return true;
}

// Helper function to read is_causal from mask params
// Returns true (causal) by default if mask or mask->extra is null
// Infer is_causal from mask data by checking if the mask is a causal mask
// A causal mask has -inf for positions where j > i (future positions)
static bool infer_is_causal_from_mask_data(ggml_tensor * mask) {
    if (mask == nullptr || mask->type != GGML_TYPE_F16) {
        GGML_DL_FATTN_DEBUG_PRINT("infer_is_causal_from_mask_data: mask is null or not F16, defaulting to causal\n");
        return true;  // Default to causal if we can't read mask data
    }

    // Read a few samples from the mask to determine if it's causal
    // For a causal mask, mask[i][j] should be -inf when j > i
    const int64_t sk = mask->ne[0];  // key sequence length
    const int64_t sq = mask->ne[1];  // query sequence length (padded)

    if (sk <= 0 || sq <= 0) {
        GGML_DL_FATTN_DEBUG_PRINT("infer_is_causal_from_mask_data: invalid dimensions sk=%lld sq=%lld, defaulting to causal\n",
                (long long)sk, (long long)sq);
        return true;  // Invalid dimensions, default to causal
    }

    // Sample a few positions to check
    // For a causal mask: mask[0][1] should be -inf (position 1 is future for position 0)
    // For a non-causal mask: mask[0][1] should be 0 or finite

    // Check if we need to sample at all - if sk <= 1 or sq <= 1, we can't determine causality
    if (sk <= 1 || sq <= 1) {
        GGML_DL_FATTN_DEBUG_PRINT("infer_is_causal_from_mask_data: sk=%lld or sq=%lld <= 1, defaulting to non-causal\n",
                (long long)sk, (long long)sq);
        return false;  // Can't determine, assume non-causal
    }

    // Read mask data from GPU to CPU
    // Allocate buffer for a small sample of mask data
    const size_t sample_size = std::min<size_t>(16, sk * sq);  // Sample up to 16 elements
    std::vector<ggml_fp16_t> mask_sample(sample_size);

    // Read first few elements: mask[0][0], mask[0][1], mask[1][0], mask[1][1], etc.
    // Mask layout: [sk, sq, 1, nr23[1]] where sk is innermost dimension
    const size_t bytes_to_read = sample_size * sizeof(ggml_fp16_t);
    ggml_backend_tensor_get(mask, mask_sample.data(), 0, bytes_to_read);

    // Check if mask[0][1] is -inf (causal) or finite (non-causal)
    // mask[0][1] is at index: 1 * sk + 0 = sk (since sk is innermost)
    // Wait, the layout is [sk, sq, ...], so mask[iq][ik] is at index: ik + iq * sk
    // So mask[0][1] (iq=0, ik=1) is at index 1

    if (sample_size > 1) {
        const float val_0_1 = ggml_fp16_to_fp32(mask_sample[1]);  // mask[iq=0][ik=1]
        const bool is_causal = std::isinf(val_0_1) && val_0_1 < 0;  // -inf means causal

        GGML_DL_FATTN_DEBUG_PRINT("infer_is_causal_from_mask_data: sk=%lld sq=%lld, mask[0][1]=%.2f, is_causal=%d\n",
                (long long)sk, (long long)sq, val_0_1, is_causal);

        return is_causal;
    }

    GGML_DL_FATTN_DEBUG_PRINT("infer_is_causal_from_mask_data: sample_size=%zu too small, defaulting to non-causal\n",
            sample_size);
    return false;  // Can't determine, assume non-causal
}

// For non-causal attention (no mask tensor), returns false
static bool get_is_causal_from_mask(ggml_tensor * mask, bool has_mask) {
    if (!has_mask) {
        // No mask tensor means non-causal attention
        return false;
    }
    if (mask == nullptr || mask->extra == nullptr) {
        return true;  // Default to causal for conversation mode
    }
    auto * params = static_cast<ggml_flash_attn_mask_params*>(mask->extra);

    // Check if params are properly initialized
    // When params->present is false, the mask params are not properly set up
    // In this case, we need to infer is_causal from the mask data
    if (!params->present) {
        // Mask params not properly initialized - infer from mask data
        return infer_is_causal_from_mask_data(mask);
    }

    return params->is_causal;
}

// Helper function to get state key from flash attention operation
// The state key combines two components:
// 1. llama_kv_cache pointer (from K tensor's extra field) - identifies the context
// 2. mask pointer - identifies the specific slot/session within the context
//
// NEW: Use seq_id as state_key to avoid state pollution across slots
// Each slot has a unique seq_id (slot.id), which provides proper isolation
// NOTE: We use an offset to ensure seq_id=0 doesn't become nullptr
static inline void * get_state_key(int32_t seq_id) {
    // Use an offset to ensure seq_id=0 doesn't become nullptr
    // Adding 1024 ensures all valid seq_ids (>=0) map to non-null pointers
    return (void*)(intptr_t)(seq_id + 1024);
}

// Helper function to get llama_kv_cache pointer from K tensor's extra field
// Handles both direct tensors and view tensors by traversing view_src chain
// The llama_kv_cache pointer is stored in the base tensor's extra field when the KV cache is created
static inline void * get_kv_cache_key(ggml_tensor * k) {
    while (k != nullptr && k->view_src != nullptr) {
        k = k->view_src;
    }
    return k ? k->extra : nullptr;
}

// Set flash attention runtime parameters (called after set_inputs, before prepare_varlen_buffers)
// REFACTORED: This function now ONLY computes host data and stores it in g_varlen_host_data_map
// It does NOT set op_params - that is done by prepare_flash_attn_varlen_buffers
//
// This function implements the accumulation logic from the baseline:
// - seqlen_k_real = prev_seqlen_k_real + seqlen_q (accumulated across calls)
// - prefill_seqlen_q is saved during prefill and used for cu_seqlens_q in decode
//
// Key design: Uses seq_id as state key to avoid state pollution across slots
// Each slot has a unique seq_id (equal to slot.id), providing proper isolation
static void set_flash_attn_runtime(
        ggml_tensor * attn,
        int batch,
        int seqlen_q_hint,
        int32_t seq_id) {
    GGML_ASSERT(batch == 1 && "DL TODO: nowadays batch is always 1, figure out how to handle multiple sequences.");
    GGML_ASSERT(attn && attn->op == GGML_OP_FLASH_ATTN_EXT);

    // Get mask and K from flash attention operation's source tensors
    // attn->src[0] = q, attn->src[1] = k, attn->src[2] = v, attn->src[3] = mask
    ggml_tensor * mask = attn->src[3];
    ggml_tensor * k = attn->src[1];

    // For non-causal attention (no mask), we still need to set up varlen params
    // but with is_causal=false. We'll use the attn tensor itself as the key for
    // storing host data, and infer dimensions from K tensor.
    bool has_mask = (mask != nullptr);
    if (!has_mask) {
        // Use attn tensor as the key for non-causal attention
        mask = attn;
    }

    // NEW: Use seq_id as state key instead of mask/K->extra
    // This prevents state pollution across different slots/sessions
    void * state_key = get_state_key(seq_id);

    // Get KV sequence length from mask (if present) or K tensor (if no mask)
    const int64_t mask_sk = has_mask ? mask->ne[0] : k->ne[1];  // key sequence length
    const int64_t mask_sq = has_mask ? mask->ne[1] : k->ne[1];  // query sequence length (same as key for non-causal)

    // [TEST-DEBUG] Task 1 & 2: Print at function entry to understand test behavior
    // printf("[TEST-DEBUG] set_flash_attn_runtime: mask=%p, has_mask=%d, seqlen_q=%d, mask_sk=%lld, mask_sq=%lld\n",
    //        (void*)mask, has_mask, seqlen_q_hint, (long long)mask_sk, (long long)mask_sq);

    // During warmup, mask might not be properly initialized yet
    // In this case, don't set varlen params
    if (mask_sk <= 0 || mask_sq <= 0) {
        GGML_DL_FATTN_DEBUG_PRINT("set_flash_attn_runtime: mask dimensions invalid (mask_sk=%lld, mask_sq=%lld), skipping varlen params\n",
                                   (long long)mask_sk, (long long)mask_sq);
        return;
    }

    // Use seqlen_q_hint directly from Q->ne[1]
    const int seqlen_q = seqlen_q_hint;

    printf("[DLFA-DEBUG] set_flash_attn_runtime ENTRY: seq_id=%d, state_key=%p, seqlen_q=%d\n", seq_id, state_key, seqlen_q);

    GGML_DL_FATTN_DEBUG_PRINT("set_flash_attn_runtime: seqlen_q=%d (from Q->ne[1])\n", seqlen_q);

    // Check if mask data is properly initialized using actual_seqlen_q
    // For decode steps (seqlen_q == 1), we skip validation because:
    // 1. The mask data might be different from prefill (different attention pattern)
    // 2. We need to update the accumulation state regardless
    // 3. The warmup detection is only needed for prefill
    //
    // SIMPLIFIED: Skip mask validation entirely for now
    // The mask validation was causing issues because the mask data format
    // is different than expected. Instead, we rely on the fact that:
    // - During warmup, the mask dimensions might be invalid or the buffer might not be ready
    // - During actual inference, we trust that the mask is correct
    // if (seqlen_q > 1 && !is_mask_data_valid(mask, seqlen_q)) {
    //     // Mask data not valid (likely warmup phase), skip varlen params
    //     return;
    // }

    // ========== REFACTORED: Hybrid approach for seqlen_k_real calculation ==========
    // The key insight is that we need to handle two scenarios:
    // 1. Normal operation: accumulate seqlen_k_real across calls
    // 2. New conversation/warmup transition: reset to mask-inferred value
    //
    // Design decision:
    // - Use accumulation logic as the PRIMARY source (baseline behavior)
    // - Use mask inference to DETECT when to reset (new conversation)
    // - If inferred < accumulated, it means mask data reflects a new/reset state
    //
    // This handles:
    // - Warmup → First prefill: inferred=5 < accumulated=7 → use inferred=5
    // - Multi-turn conversation: inferred=43 >= accumulated=43 → use accumulated=43
    // - New conversation after KV clear: inferred=15 < accumulated=100 → use inferred=15

    int prev_seqlen_k_real = 0;
    int prefill_seqlen_q = 0;
    {
        std::lock_guard<std::mutex> lock(g_accumulation_mutex);

        // Get prev_seqlen_k_real from global map (keyed by K->data)
        auto it_prev = g_prev_seqlen_k_real_map.find(state_key);
        if (it_prev != g_prev_seqlen_k_real_map.end()) {
            prev_seqlen_k_real = it_prev->second;
        }

        // Get prefill_seqlen_q from global map (keyed by K->data)
        auto it_prefill = g_prefill_seqlen_q_map.find(state_key);
        if (it_prefill != g_prefill_seqlen_q_map.end()) {
            prefill_seqlen_q = it_prefill->second;
        }
    }

    // Calculate accumulated value (baseline behavior for conversations)
    int accumulated = (prev_seqlen_k_real > 0 ? prev_seqlen_k_real : 0) + seqlen_q;

    // Try to infer seqlen_k_real from mask data (for debugging/sanity check)
    int inferred = (int) infer_from_mask_last_row(mask, seqlen_q - 1);

    // Determine final seqlen_k_real
    // Key insight: We need to distinguish between two scenarios:
    // 1. test-backend-ops: Each test has unique mask pointer → prev_seqlen_k_real=0
    //    → Tests don't call set_flash_attn_runtime, they use non-varlen path
    //    → If we get here with prev_seqlen_k_real=0, it's a conversation first prefill
    // 2. Conversations: Mask pointer is reused → prev_seqlen_k_real>0
    //    → seqlen_k_real should accumulate
    //
    // Detection logic:
    // - If prev_seqlen_k_real == 0: First prefill in conversation
    //   → Use seqlen_q (the prompt length)
    // - If prev_seqlen_k_real > 0 && seqlen_q > prev_seqlen_k_real: New conversation
    //   → Reset to seqlen_q (warmup→first prefill transition)
    // - If prev_seqlen_k_real is much larger than seqlen_q (e.g., prev > 2*seqlen_q + 100):
    //   → Likely a new short conversation after a long one, reset to seqlen_q
    //   → This handles multi-session scenarios where slot is reused
    // - Otherwise: Normal conversation accumulation
    //   → Use accumulated value
    int seqlen_k_real;
    bool is_new_conversation = false;

    printf("[DLFA-DEBUG] set_flash_attn_runtime: seqlen_q=%d, prev_seqlen_k_real=%d, accumulated=%d, inferred=%d, condition_check: prefill && prev > q+50 ? %d > %d ? %d\n",
            seqlen_q, prev_seqlen_k_real, accumulated, inferred, seqlen_q > 1, prev_seqlen_k_real > seqlen_q + 50, seqlen_q > 1 && prev_seqlen_k_real > seqlen_q + 50);

    if (prev_seqlen_k_real == 0) {
        // First prefill in conversation - use seqlen_q (the prompt length)
        // Note: Tests don't call set_flash_attn_runtime, they use non-varlen path
        // So if we get here with prev_seqlen_k_real=0, it's definitely a conversation
        seqlen_k_real = seqlen_q;
        GGML_DL_FATTN_DEBUG_PRINT("set_flash_attn_runtime: first prefill, using seqlen_q=%d\n",
                seqlen_q);
    } else if (seqlen_q > prev_seqlen_k_real) {
        // New conversation detected - start fresh with seqlen_q
        seqlen_k_real = seqlen_q;
        is_new_conversation = true;
        GGML_DL_FATTN_DEBUG_PRINT("set_flash_attn_runtime: detected new conversation (seqlen_q=%d > prev_seqlen_k_real=%d), starting fresh\n",
                seqlen_q, prev_seqlen_k_real);
    } else if (false /* DISABLED: Multi-session detection causes issues with KV cache reuse
                      The condition prev_seqlen_k_real > seqlen_q * 2 incorrectly triggers
                      when KV cache is reused (seqlen_q is only new tokens, not total).
                      Since we now use seq_id as state_key, each slot has isolated state. */) {
        // DISABLED: Multi-session detection logic removed
        // This detection was originally meant to handle multi-session scenarios where
        // different slots share the same seq_id. However, since we now use seq_id as
        // state_key, each slot has its own isolated state, making this detection
        // unnecessary and harmful in KV cache reuse scenarios.
        //
        // Problem: In KV cache reuse, seqlen_q is only the new tokens (e.g., 14),
        // while prev_seqlen_k_real is the actual KV cache size (e.g., 39).
        // The condition prev > seqlen_q * 2 would incorrectly trigger, causing
        // seqlen_k_real to reset to 14 while the actual KV cache has 39 tokens.
        seqlen_k_real = seqlen_q;
        is_new_conversation = true;
        printf("[DLFA-NEW-SESSION] Disabled - this should never be printed\n");
    } else {
        // Normal conversation operation - accumulate
        seqlen_k_real = accumulated;
        printf("[DLFA-DEBUG] set_flash_attn_runtime: normal conversation, using accumulated=%d (prev=%d + seqlen_q=%d)\n",
                accumulated, prev_seqlen_k_real, seqlen_q);

        // DISABLED: KV cache cleanup detection via mask inference
        // The callback mechanism (dlfa_kv_cache_removal_hook) already synchronizes
        // prev_seqlen_k_real when KV cache is cleaned up. The accumulated value is
        // already correct and should not be overridden by the inferred value from mask.
        //
        // Original logic (disabled):
        // if (inferred > 0 && inferred < seqlen_k_real) {
        //     seqlen_k_real = inferred;  // This was overriding correct values with incorrect inferred values
        // }
    }

    // Log mask inference result for debugging
    if (inferred > 0 && inferred <= mask_sk) {
        GGML_DL_FATTN_DEBUG_PRINT("set_flash_attn_runtime: mask inference returned %d (seqlen_k_real=%d)\n",
                inferred, seqlen_k_real);
    }

    // Debug: log the comparison between inferred and accumulated values
    GGML_DL_FATTN_DEBUG_PRINT("set_flash_attn_runtime: seqlen_q=%d, mask_sq=%lld, mask_sk=%lld, prev_seqlen_k_real=%d, inferred=%d, accumulated=%d, final seqlen_k_real=%d, is_new_conversation=%d\n",
            seqlen_q, (long long)mask_sq, (long long)mask_sk, prev_seqlen_k_real, inferred, accumulated, seqlen_k_real, is_new_conversation);

    // Clamp to valid range
    if (seqlen_k_real <= 0 || seqlen_k_real > mask_sk) {
        GGML_DL_FATTN_DEBUG_PRINT("set_flash_attn_runtime: seqlen_k_real invalid (%d), falling back to mask_sk=%lld\n",
                seqlen_k_real, (long long)mask_sk);
        seqlen_k_real = (int) mask_sk;
    }

    GGML_ASSERT(seqlen_k_real > 0);

    // Save seqlen_k_real for next call
    {
        std::lock_guard<std::mutex> lock(g_accumulation_mutex);
        g_prev_seqlen_k_real_map[state_key] = seqlen_k_real;

        // Handle prefill_seqlen_q:
        // - If new conversation detected, reset prefill_seqlen_q
        // - Otherwise, update only during prefill phase (seqlen_q > 1)
        if (is_new_conversation) {
            // Reset prefill_seqlen_q for new conversation
            if (seqlen_q > 1) {
                g_prefill_seqlen_q_map[state_key] = seqlen_q;
                GGML_DL_FATTN_DEBUG_PRINT("set_flash_attn_runtime: new conversation, saved prefill_seqlen_q=%d for state_key=%p (mask=%p)\n",
                        seqlen_q, state_key, (void*)mask);
            } else {
                // Decode phase in new conversation - clear prefill_seqlen_q
                g_prefill_seqlen_q_map.erase(state_key);
                prefill_seqlen_q = 0;
                GGML_DL_FATTN_DEBUG_PRINT("set_flash_attn_runtime: new conversation decode phase, cleared prefill_seqlen_q for state_key=%p (mask=%p)\n",
                        state_key, (void*)mask);
            }
        } else if (seqlen_q > 1) {
            // Normal prefill - update prefill_seqlen_q
            g_prefill_seqlen_q_map[state_key] = seqlen_q;
            GGML_DL_FATTN_DEBUG_PRINT("set_flash_attn_runtime: saved prefill_seqlen_q=%d for state_key=%p (mask=%p)\n",
                    seqlen_q, state_key, (void*)mask);
        }
    }

    // block table stride is built from padded kv size (n_kv = mask_sk) with 256 page blocks
    const int page_block = 256;
    const int bt_stride = (mask_sk + page_block - 1) / page_block;

    // Determine seqlen_q_for_cu: use prefill_seqlen_q if available (baseline behavior)
    // IMPORTANT: When is_new_conversation=true and seqlen_q > 1 (prefill), we need to use
    // seqlen_q directly because prefill_seqlen_q still holds the OLD value from warmup.
    // The prefill_seqlen_q is updated AFTER this point, so we can't rely on it here.
    int seqlen_q_for_cu = seqlen_q;
    if (seqlen_q > 1) {
        // Prefill phase (first turn or subsequent turn) - use current seqlen_q
        seqlen_q_for_cu = seqlen_q;
        GGML_DL_FATTN_DEBUG_PRINT("set_flash_attn_runtime: prefill phase, using seqlen_q=%d for cu_seqlens_q\n",
                seqlen_q);
    } else if (prefill_seqlen_q > 0) {
        // Decode phase - use stored prefill_seqlen_q to keep cu_seqlens_q constant
        seqlen_q_for_cu = prefill_seqlen_q;
        GGML_DL_FATTN_DEBUG_PRINT("set_flash_attn_runtime: decode phase, using prefill_seqlen_q=%d for cu_seqlens_q (current seqlen_q=%d)\n",
                prefill_seqlen_q, seqlen_q);
    }

    // ========== NEW: Store host data in g_varlen_host_data_map ==========
    // This data is shared by all 48 FA layers (keyed by K->data)
    {
        std::lock_guard<std::mutex> lock(g_varlen_host_data_mutex);

        // Get or create host data for this state_key
        flash_attn_varlen_host_data * host_data = nullptr;
        auto it = g_varlen_host_data_map.find(state_key);
        if (it == g_varlen_host_data_map.end()) {
            host_data = new flash_attn_varlen_host_data();
            g_varlen_host_data_map[state_key] = host_data;
            GGML_DL_FATTN_DEBUG_PRINT("set_flash_attn_runtime: created new host_data for state_key=%p (mask=%p)\n", state_key, (void*)mask);
        } else {
            host_data = it->second;
        }

        // Store scalar values
        // seqlen_q: actual seqlen_q (1 for decode, prefill seqlen_q for prefill)
        // seqlen_q_for_cu: used for cu_seqlens_q (prefill seqlen_q, kept constant in decode)
        host_data->seqlen_q = seqlen_q;
        host_data->seqlen_q_for_cu = seqlen_q_for_cu;
        host_data->seqlen_k_real = seqlen_k_real;
        host_data->block_table_stride = bt_stride;
        host_data->is_causal = get_is_causal_from_mask(mask, has_mask);  // Read from mask params or infer from has_mask
        host_data->is_valid = true;

        // Compute and store cu_seqlens_q (cumulative sequence lengths for Q)
        // Uses seqlen_q_for_cu to keep cu_seqlens_q constant across decode steps
        host_data->cu_seqlens_q.resize(batch + 1);
        host_data->cu_seqlens_q[0] = 0;
        for (int b = 0; b < batch; ++b) {
            host_data->cu_seqlens_q[b + 1] = host_data->cu_seqlens_q[b] + seqlen_q_for_cu;
        }

        // Compute and store seqused_k (actual used length for each sequence in K)
        host_data->seqused_k.resize(batch);
        for (int b = 0; b < batch; ++b) {
            host_data->seqused_k[b] = seqlen_k_real;
        }

        // Compute and store block_table (paged KV cache block mapping)
        host_data->block_table.resize(batch * bt_stride);
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < bt_stride; ++i) {
                host_data->block_table[b * bt_stride + i] = b * bt_stride + i;
            }
        }

        GGML_DL_FATTN_DEBUG_PRINT("set_flash_attn_runtime: stored host_data for state_key=%p (mask=%p): seqlen_q=%d, seqlen_q_for_cu=%d, seqlen_k_real=%d, bt_stride=%d, cu_seqlens_q=[%d,%d], seqused_k=[%d]\n",
                state_key, (void*)mask, seqlen_q, seqlen_q_for_cu, seqlen_k_real, bt_stride,
                host_data->cu_seqlens_q[0], host_data->cu_seqlens_q[1], host_data->seqused_k[0]);
    }

    // NOTE: We no longer set op_params here - that is done by prepare_flash_attn_varlen_buffers
    // This separation allows set_flash_attn_runtime to be called once (after set_inputs)
    // and prepare_flash_attn_varlen_buffers to be called for each FA node

    GGML_DL_FATTN_DEBUG_PRINT(
        "set_flash_attn_runtime: batch=%d, seqlen_q=%d, seqlen_q_for_cu=%d, seqlen_k_real=%d, bt_stride=%d\n",
        batch, seqlen_q, seqlen_q_for_cu, seqlen_k_real, bt_stride);

    // Task 4 debug prints: set_flash_attn_runtime key variables
    GGML_DL_FATTN_DEBUG_PRINT("[TASK4-DEBUG] set_flash_attn_runtime: seqlen_q=%d, seqlen_q_for_cu=%d, seqlen_k_real=%d, block_table_stride=%d\n",
            seqlen_q, seqlen_q_for_cu, seqlen_k_real, bt_stride);
    GGML_DL_FATTN_DEBUG_PRINT("[TASK4-DEBUG] set_flash_attn_runtime: mask_sk=%lld, mask_sq=%lld, prev_seqlen_k_real=%d, state_key=%p, mask=%p\n",
            (long long)mask_sk, (long long)mask_sq, prev_seqlen_k_real, state_key, (void*)mask);
}

static void flash_attn_ext_dldnn_mha_varlen_forward(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_DL_FATTN_DEBUG_PRINT("\n========== ENTERING %s ==========\n", __FUNCTION__);

    const int id = ctx.device;  // Use device from context, not ggml_cuda_get_device()

    ggml_cuda_pool_alloc<float> softmax_lse_mem(ctx.pool(id));
    ggml_cuda_pool_alloc<uint8_t> workspace_mem(ctx.pool(id));
    const ggml_tensor * mask = dst->src[3];

    // Read varlen scalar params from operation tensor's op_params
    // NOTE: batch is always 1, hardcoded (not stored in op_params)
    // These are non-const because they may be updated if set_flash_attn_runtime is called
    const int batch = 1;
    int seqlen_q = ggml_get_op_params_i32(dst, GGML_FLASH_ATTN_PARAM_VARLEN_SEQLEN_Q_I32);
    int seqlen_k_real = ggml_get_op_params_i32(dst, GGML_FLASH_ATTN_PARAM_VARLEN_SEQLEN_K_REAL_I32);
    int block_table_stride = ggml_get_op_params_i32(dst, GGML_FLASH_ATTN_PARAM_VARLEN_BT_STRIDE_I32);
    bool has_varlen_params = ggml_get_op_params_i32(dst, GGML_FLASH_ATTN_PARAM_VARLEN_HAS_PARAMS_I32) != 0;

    // [MULTI-GPU-DEBUG] Task 1.3: Print after reading op_params (commented out after fix verified)
    // printf("[MULTI-GPU-DEBUG] varlen_forward: device=%d, dst=%p, seqlen_q=%d, seqlen_k_real=%d, has_varlen_params=%d\n",
    //        id, (void*)dst, seqlen_q, seqlen_k_real, has_varlen_params ? 1 : 0);

    // Task 4 debug prints: op_params values read from dst (commented out after fix verified)
    // static int debug_counter = 0;
    // if (debug_counter % 48 == 0) {
    //     printf("[TASK4-DEBUG] varlen_forward: op_params read - seqlen_q=%d, seqlen_k_real=%d, block_table_stride=%d, has_varlen_params=%d\n",
    //             seqlen_q, seqlen_k_real, block_table_stride, has_varlen_params ? 1 : 0);
    // }
    // debug_counter++;
    GGML_DL_FATTN_DEBUG_PRINT("[TASK4-DEBUG] varlen_forward: device=%d, dst=%p\n", id, (void*)dst);

    // For test-backend-ops compatibility and warmup phase:
    // When has_varlen_params=0, it means either:
    // - Warmup phase (K->data is nullptr or mask data not valid)
    // - set_flash_attn_runtime was not called (e.g., test-backend-ops)
    // - Tensor pointer mismatch (test framework uses different tensor instance)
    //
    // In these cases, fall back to non-varlen path which uses standard GGML tensor layout.
    // The varlen path requires paged KV cache layout which tests don't use.
    //
    // NOTE: This can happen in non-test environments (e.g., llama-server warmup phase)
    // when K->data is nullptr, so we don't assert GGML_IS_TEST here.
    if (!has_varlen_params) {
        GGML_DL_FATTN_DEBUG_PRINT("varlen_forward: has_varlen_params=0, falling back to non-varlen dldnn path\n");
        flash_attn_ext_dldnn_mha_forward(ctx, dst);
        return;
    }

    const char* env_dl_fattn_debug = getenv("GGML_DL_FATTN_DEBUG");
    ggml_flash_attn_mask_params mask_info{};
    bool have_mask_params = ggml_flash_attn_ext_get_mask_params(dst, &mask_info);

    // Debug: print mask_info.is_causal value and seqlen values
    // Commented out after fix verified
    // static int is_causal_debug_counter = 0;
    // // Print for every prefill (seqlen_q > 1) to debug multi-turn issue
    // if (is_causal_debug_counter % 28 == 0 || seqlen_q > 1) {
    //     printf("[IS_CAUSAL_DEBUG] varlen_forward: have_mask_params=%d, is_causal=%d, seqlen_q=%d, seqlen_k_real=%d, layer=%d\n",
    //            have_mask_params ? 1 : 0, mask_info.is_causal ? 1 : 0, seqlen_q, seqlen_k_real, is_causal_debug_counter % 28);
    // }
    // is_causal_debug_counter++;

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

        // Varlen params are now guaranteed to be set (checked above)
        const int max_seqlen_q = (int) S_q;
        const int max_seqlen_k = (int) S_k;
        const int num_blocks_per_seq_runtime = block_table_stride;

        // Device buffers should already be prepared and copied in flash_attn_ext_dldnn
        // (called before this function for async overlap). Just retrieve pointers from cache.
        int * cu_seqlens_q_dev = nullptr;
        int * seqused_k_dev    = nullptr;
        int * block_table_dev  = nullptr;

        {
            // Retrieve varlen_data from global map using operation tensor pointer
            flash_attn_varlen_data * varlen_data = nullptr;
            {
                std::lock_guard<std::mutex> lock(g_varlen_data_mutex);
                auto it = g_varlen_data_map.find(dst);
                GGML_ASSERT(it != g_varlen_data_map.end() && "varlen data must be prepared before graph capture");
                varlen_data = it->second;
            }

            GGML_ASSERT((int) batch == (int) B);
            GGML_ASSERT((int) varlen_data->cu_seqlens_q.size() == (int) B + 1);
            GGML_ASSERT((int) varlen_data->seqused_k.size() == (int) B);

            // Device cache MUST be initialized by flash_attn_ext_dldnn_prepare_varlen_buffers
            // before graph capture. No dynamic re-preparation allowed inside graph execution.
            // Retrieve cache for current device from the per-device cache map

            auto it = varlen_data->device_cache_map.find(id);
            GGML_ASSERT(it != varlen_data->device_cache_map.end() && "device cache must be prepared before graph capture");

            flash_attn_device_layout_cache * cache = it->second;

            // Verify device matches - this should always pass if prepare was called correctly
            GGML_ASSERT(cache->device == id && "device cache must match current device");

            cu_seqlens_q_dev = cache->cu_seqlens_ptr;
            seqused_k_dev = cache->seqused_ptr;
            block_table_dev = cache->block_table_ptr;

            // Debug: print cu_seqlens_q and seqused_k values during prefill
            // Commented out after fix verified
            // if (seqlen_q > 1) {
            //     printf("[PREFILL_DEBUG] varlen_forward: cu_seqlens_q=[%d,%d], seqused_k=[%d], seqlen_q=%d, seqlen_k_real=%d\n",
            //            varlen_data->cu_seqlens_q[0], varlen_data->cu_seqlens_q[1],
            //            varlen_data->seqused_k[0], seqlen_q, seqlen_k_real);
            // }

            // Task 4 debug prints: cu_seqlens_q and seqused_k values from varlen_data
            GGML_DL_FATTN_DEBUG_PRINT("[TASK4-DEBUG] varlen_forward: cu_seqlens_q values (size=%zu): ",
                    varlen_data->cu_seqlens_q.size());
            for (size_t i = 0; i < varlen_data->cu_seqlens_q.size() && i < 8; ++i) {
                GGML_DL_FATTN_DEBUG_PRINT("%d ", varlen_data->cu_seqlens_q[i]);
            }
            GGML_DL_FATTN_DEBUG_PRINT("\n");
            GGML_DL_FATTN_DEBUG_PRINT("[TASK4-DEBUG] varlen_forward: seqused_k values (size=%zu): ",
                    varlen_data->seqused_k.size());
            for (size_t i = 0; i < varlen_data->seqused_k.size() && i < 8; ++i) {
                GGML_DL_FATTN_DEBUG_PRINT("%d ", varlen_data->seqused_k[i]);
            }
            GGML_DL_FATTN_DEBUG_PRINT("\n");
            GGML_DL_FATTN_DEBUG_PRINT("[TASK4-DEBUG] varlen_forward: device pointers - cu_seqlens_q_dev=%p, seqused_k_dev=%p, block_table_dev=%p\n",
                    (void*)cu_seqlens_q_dev, (void*)seqused_k_dev, (void*)block_table_dev);

            // Phase 3 verification: Assert device pointers are valid
            // These assertions validate that prepare_varlen_buffers correctly prepared device buffers
            GGML_ASSERT(cu_seqlens_q_dev != nullptr && "cu_seqlens_q_dev must be prepared before graph execution");
            GGML_ASSERT(seqused_k_dev != nullptr && "seqused_k_dev must be prepared before graph execution");
            GGML_ASSERT(block_table_dev != nullptr && "block_table_dev must be prepared before graph execution");

            // Task 2 debug: Verify device buffer contents by copying back to host (commented out after fix verified)
            // This is critical for debugging multi-GPU CUDA Graph issues
            // {
            //     std::vector<int> cu_seqlens_q_host(B + 1);
            //     std::vector<int> seqused_k_host(B);
            //     cudaMemcpy(cu_seqlens_q_host.data(), cu_seqlens_q_dev, (B + 1) * sizeof(int), cudaMemcpyDeviceToHost);
            //     cudaMemcpy(seqused_k_host.data(), seqused_k_dev, B * sizeof(int), cudaMemcpyDeviceToHost);
            //     printf("[TASK2-DEBUG] varlen_forward device=%d: DEVICE buffer contents - cu_seqlens_q=[%d,%d], seqused_k=[%d]\n",
            //            id, cu_seqlens_q_host[0], cu_seqlens_q_host[1], seqused_k_host[0]);
            //     printf("[TASK2-DEBUG] varlen_forward device=%d: HOST varlen_data - cu_seqlens_q=[%d,%d], seqused_k=[%d]\n",
            //            id, varlen_data->cu_seqlens_q[0], varlen_data->cu_seqlens_q[1], varlen_data->seqused_k[0]);
            //
            //     // Check if device and host data match
            //     bool data_mismatch = false;
            //     if (cu_seqlens_q_host[0] != varlen_data->cu_seqlens_q[0] ||
            //         cu_seqlens_q_host[1] != varlen_data->cu_seqlens_q[1] ||
            //         seqused_k_host[0] != varlen_data->seqused_k[0]) {
            //         data_mismatch = true;
            //         printf("[TASK2-DEBUG] *** DATA MISMATCH DETECTED on device %d! ***\n", id);
            //     }
            // }
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
        // Use tensor->nb to compute strides for Q descriptor so varlen layout is supported.
        // dims = [total_q_max, H, D]  (total_q_max = batch * max_seqlen_q)
        // We expect GGML tensor layout ne[0]=D, ne[1]=S, ne[2]=H, ne[3]=B with nb[] in bytes.
        // Strides must be provided in element units (not bytes) relative to nb[0].
        auto set_desc_3d_q_packed = [&](cudnnTensorDescriptor_t d, const ggml_tensor * tensor, int dim0, int dim1, int dim2, cudnnDataType_t dt) {
            int dims[3] = { dim0, dim1, dim2 };
            int strides[3] = {
                static_cast<int>(tensor->nb[1] / tensor->nb[0]), // stride for total_q (seq stride in elements)
                static_cast<int>(tensor->nb[2] / tensor->nb[0]), // stride for H (head stride in elements)
                1                                                // stride for D
            };
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
        set_desc_3d_q_packed(q_desc.desc, Q, total_q_max, (int) H_q, (int) D, ggml_type_to_cudnn_type(Q->type));
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

bool flash_attn_dldnn_available(const ggml_tensor * dst) {

    const char *env_force_no_dlfa = getenv("GGML_FORCE_NO_DLFA");
    if (env_force_no_dlfa != nullptr && strcmp(env_force_no_dlfa, "1") == 0) {
        return false;
    }

    const struct ggml_tensor * Q = dst->src[0];
    const struct ggml_tensor * K = dst->src[1];
    const struct ggml_tensor * V = dst->src[2];
    const struct ggml_tensor * mask = dst->src[3];
    const struct ggml_tensor * sinks = dst->src[4];

    if(sinks){
        // DL-TODO: support attention sinks later.
        return false;
    }

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

    // Get mask parameters using the standard function
    ggml_flash_attn_mask_params mask_info{};
    bool have_mask_params = ggml_flash_attn_ext_get_mask_params(dst, &mask_info);

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
        } else if (!mask_info.present && !have_mask_params) {
            // If mask exists but mask_info.present is false AND we don't have mask params at all,
            // it means mask params haven't been set yet (e.g., during warmup).
            // In this case, we should still allow DLDNN to be used.
            GGML_DL_FATTN_DEBUG_PRINT("mask metadata not yet set, allowing DLDNN\n");
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
// REFACTORED: This function now reads host data from g_varlen_host_data_map[seq_id]
// and only handles device buffer allocation and data transfer
// It also sets op_params (which was previously done by set_flash_attn_runtime)
void flash_attn_ext_dldnn_prepare_varlen_buffers(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int32_t seq_id) {
    const int id = ctx.device;  // Use device from context, not ggml_cuda_get_device()

    // [MULTI-GPU-DEBUG] Task 1.2: Print at function entry (commented out after fix verified)
    // ggml_tensor * mask_for_debug = dst ? dst->src[3] : nullptr;
    // printf("[MULTI-GPU-DEBUG] prepare_varlen_buffers: device=%d, dst=%p, mask=%p\n", id, (void*)dst, (void*)mask_for_debug);

    // Ensure we're on the correct device BEFORE any CUDA operations
    ggml_cuda_set_device(id);

    // Get mask and K from flash attention operation's source tensors
    ggml_tensor * mask = dst->src[3];
    if (!mask) {
        // No mask, no varlen params needed
        return;
    }

    ggml_tensor * k = dst->src[1];

    // NEW: Use seq_id as state_key (same as set_flash_attn_runtime)
    // This ensures proper isolation across different slots/sessions
    void * state_key = get_state_key(seq_id);

    // IMPORTANT: Always initialize has_varlen_params to 0 at the start
    // This ensures that if we return early (e.g., due to nullptr), has_varlen_params is 0
    ggml_set_op_params_i32(dst, GGML_FLASH_ATTN_PARAM_VARLEN_HAS_PARAMS_I32, 0);

    // ========== NEW: Read host data from g_varlen_host_data_map[state_key] ==========
    flash_attn_varlen_host_data * host_data = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_varlen_host_data_mutex);
        auto it = g_varlen_host_data_map.find(state_key);
        if (it == g_varlen_host_data_map.end() || !it->second->is_valid) {
            // Host data not available (set_flash_attn_runtime not called yet or state_key invalid)
            // has_varlen_params is already 0, so just return
            GGML_DL_FATTN_DEBUG_PRINT("prepare_varlen_buffers: host_data not available for seq_id=%d (state_key=%p), returning early (has_varlen_params=0 set)\n", seq_id, state_key);
            return;
        }
        host_data = it->second;
    }

    // Read values from host_data
    // seqlen_q: actual seqlen_q (1 for decode, prefill seqlen_q for prefill)
    // seqlen_q_for_cu: used for cu_seqlens_q (prefill seqlen_q, kept constant in decode)
    const int batch = 1;  // Always 1
    const int seqlen_q = host_data->seqlen_q;
    const int seqlen_q_for_cu = host_data->seqlen_q_for_cu;
    const int seqlen_k_real = host_data->seqlen_k_real;
    const int num_blocks_per_seq = host_data->block_table_stride;

    // [MULTI-GPU-DEBUG] Task 1.2: Print after reading host_data (commented out after fix verified)
    // printf("[MULTI-GPU-DEBUG] prepare_varlen_buffers: device=%d, seqlen_q=%d, seqlen_k_real=%d, cu_seqlens_q=[%d,%d]\n",
    //        id, seqlen_q, seqlen_k_real, host_data->cu_seqlens_q[0], host_data->cu_seqlens_q[1]);

    GGML_DL_FATTN_DEBUG_PRINT("prepare_varlen_buffers: read from host_data - seqlen_q=%d, seqlen_q_for_cu=%d, seqlen_k_real=%d, bt_stride=%d (state_key=%p, mask=%p)\n",
            seqlen_q, seqlen_q_for_cu, seqlen_k_real, num_blocks_per_seq, state_key, (void*)mask);

    // ========== NEW: Set op_params (previously done by set_flash_attn_runtime) ==========
    ggml_set_op_params_i32(dst, GGML_FLASH_ATTN_PARAM_VARLEN_SEQLEN_Q_I32, seqlen_q);
    ggml_set_op_params_i32(dst, GGML_FLASH_ATTN_PARAM_VARLEN_SEQLEN_K_REAL_I32, seqlen_k_real);
    ggml_set_op_params_i32(dst, GGML_FLASH_ATTN_PARAM_VARLEN_BT_STRIDE_I32, num_blocks_per_seq);
    ggml_set_op_params_i32(dst, GGML_FLASH_ATTN_PARAM_VARLEN_HAS_PARAMS_I32, 1);

    // Set mask magic value and present flag so ggml_flash_attn_ext_get_mask_params returns true
    ggml_set_op_params_i32(dst, GGML_FLASH_ATTN_PARAM_MASK_MAGIC_I32, GGML_FLASH_ATTN_PARAM_MASK_MAGIC_VALUE);
    ggml_set_op_params_i32(dst, GGML_FLASH_ATTN_PARAM_MASK_PRESENT_I32, 1);
    // Set is_causal from host_data (always true for autoregressive models)
    ggml_set_op_params_i32(dst, GGML_FLASH_ATTN_PARAM_MASK_CAUSAL_I32, host_data->is_causal ? 1 : 0);

    const int64_t B = dst->src[0]->ne[3];
    const int page_block_size = 256;

    GGML_ASSERT((int) batch == (int) B);

    // Use global map to get/create varlen_data for this operation tensor
    flash_attn_varlen_data * varlen_data = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_varlen_data_mutex);
        auto it = g_varlen_data_map.find(dst);
        if (it == g_varlen_data_map.end()) {
            // Create new varlen data structure
            varlen_data = new flash_attn_varlen_data();
            g_varlen_data_map[dst] = varlen_data;
            GGML_DL_FATTN_DEBUG_PRINT("prepare_varlen_buffers: created new varlen_data for dst=%p\n", (void*)dst);
        } else {
            varlen_data = it->second;
        }

        // Copy host data from g_varlen_host_data_map to varlen_data
        // This is needed because varlen_forward reads from varlen_data
        varlen_data->cu_seqlens_q = host_data->cu_seqlens_q;
        varlen_data->seqused_k = host_data->seqused_k;
        varlen_data->block_table = host_data->block_table;
    }

    GGML_ASSERT((int) varlen_data->cu_seqlens_q.size() == (int) B + 1);
    GGML_ASSERT((int) varlen_data->seqused_k.size() == (int) B);

    const size_t cu_bytes = varlen_data->cu_seqlens_q.size() * sizeof(int32_t);
    const size_t su_bytes = varlen_data->seqused_k.size() * sizeof(int32_t);
    const size_t bt_elems = (size_t)((seqlen_k_real + page_block_size - 1) / page_block_size);
    const size_t bt_bytes = bt_elems * sizeof(int32_t);

    // Get or create cache for current device using per-device cache map
    flash_attn_device_layout_cache * cache = nullptr;
    auto it = varlen_data->device_cache_map.find(id);
    if (it == varlen_data->device_cache_map.end()) {
        // Create new cache for this device
        cache = new flash_attn_device_layout_cache();
        cache->device = id;
        varlen_data->device_cache_map[id] = cache;
        GGML_DL_FATTN_DEBUG_PRINT("prepare_varlen_buffers: created new cache for device %d (dst=%p)\n", id, (void*)dst);
    } else {
        cache = it->second;
    }

    // Verify device field is correctly set
    GGML_ASSERT(cache->device == id && "cache device must match current device");

    const size_t total_elems = varlen_data->cu_seqlens_q.size() + varlen_data->seqused_k.size() + num_blocks_per_seq;
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
    cache->seqused_ptr = cache->cu_seqlens_ptr + varlen_data->cu_seqlens_q.size();
    cache->block_table_ptr = cache->seqused_ptr + varlen_data->seqused_k.size();

    // Phase 3 verification: Assert device buffers are properly allocated
    GGML_ASSERT(cache->cu_seqlens_ptr != nullptr && "cu_seqlens_ptr must be allocated");
    GGML_ASSERT(cache->seqused_ptr != nullptr && "seqused_ptr must be allocated");
    GGML_ASSERT(cache->block_table_ptr != nullptr && "block_table_ptr must be allocated");

    // Start async copies early for better overlap with subsequent operations
    int * host_ptr = cache->host_combined_ptr;
    memcpy(host_ptr, varlen_data->cu_seqlens_q.data(), cu_bytes);
    host_ptr += varlen_data->cu_seqlens_q.size();
    memcpy(host_ptr, varlen_data->seqused_k.data(), su_bytes);
    host_ptr += varlen_data->seqused_k.size();
    memcpy(host_ptr, varlen_data->block_table.data(), bt_bytes);

    // Ensure device is set before memory copy
    ggml_cuda_set_device(id);

    // Use cudaMemcpyAsync for better performance
    cudaStream_t stream = ctx.stream();
    CUDA_CHECK(cudaMemcpyAsync(cache->combined_ptr, cache->host_combined_ptr, total_elems * sizeof(int), cudaMemcpyHostToDevice, stream));
    // CUDA_CHECK(cudaStreamSynchronize(stream)); // no need to cudaStreamSynchronize, just for debug.

    // Phase 3 verification: Log device buffer preparation completion
    GGML_DL_FATTN_DEBUG_PRINT(
        "[DEBUG] prepare_varlen_buffers: device %d buffers prepared (async, no sync) - "
        "cu_seqlens_ptr=%p, seqused_ptr=%p, block_table_ptr=%p, total_elems=%zu\n",
        id, (void*)cache->cu_seqlens_ptr, (void*)cache->seqused_ptr,
        (void*)cache->block_table_ptr, total_elems);
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

    // if (ggml_dlfa_graphs_enabled()) {
    // use mha_varlen_forward always.
    flash_attn_ext_dldnn_mha_varlen_forward(ctx, dst);
    // } else {
        // flash_attn_ext_dldnn_mha_forward(ctx, dst);
    // }
}

// Reset accumulation state (called when starting a new inference session)
// This clears the prev_seqlen_k_real and prefill_seqlen_q maps
void flash_attn_ext_dldnn_reset_accumulation_state() {
    std::lock_guard<std::mutex> lock(g_accumulation_mutex);
    g_prev_seqlen_k_real_map.clear();
    g_prefill_seqlen_q_map.clear();
}

// Cleanup function to free all varlen_data and device memory
void flash_attn_ext_dldnn_cleanup_varlen_data() {
    // Clear accumulation maps first
    {
        std::lock_guard<std::mutex> lock(g_accumulation_mutex);
        g_prev_seqlen_k_real_map.clear();
        g_prefill_seqlen_q_map.clear();
    }

    std::lock_guard<std::mutex> lock(g_varlen_data_mutex);

    for (auto & pair : g_varlen_data_map) {
        flash_attn_varlen_data * varlen_data = pair.second;
        if (varlen_data) {
            // Free device memory in each device cache
            for (auto & cache_pair : varlen_data->device_cache_map) {
                flash_attn_device_layout_cache * cache = cache_pair.second;
                if (cache) {
                    const int device_id = cache->device;
                    if (device_id >= 0) {
                        ggml_cuda_set_device(device_id);

                        // Free device memory
                        if (cache->combined_ptr) {
                            CUDA_CHECK(cudaFree(cache->combined_ptr));
                            cache->combined_ptr = nullptr;
                        }

                        // Free host pinned memory
                        if (cache->host_combined_ptr) {
                            CUDA_CHECK(cudaFreeHost(cache->host_combined_ptr));
                            cache->host_combined_ptr = nullptr;
                        }
                    }
                    delete cache;
                }
            }
            varlen_data->device_cache_map.clear();
            delete varlen_data;
        }
    }

    g_varlen_data_map.clear();
}

// DLFA state cleanup function for KV cache synchronization
// Called when llama_memory_seq_rm is invoked to clean up KV cache
// This synchronizes DLFA's internal state with the actual KV cache size
void flash_attn_ext_dldnn_clear_state_for_seq_id(llama_seq_id seq_id, llama_pos new_kv_size) {
    void * state_key = get_state_key(seq_id);

    std::lock_guard<std::mutex> lock(g_accumulation_mutex);

    // Update g_prev_seqlen_k_real_map to the new KV cache size
    g_prev_seqlen_k_real_map[state_key] = new_kv_size;

    // Also clear prefill_seqlen_q to ensure next prefill sets it correctly
    g_prefill_seqlen_q_map.erase(state_key);

    GGML_DL_FATTN_DEBUG_PRINT("dlfa_clear_state: seq_id=%d, state_key=%p, new_kv_size=%d\n",
            seq_id, state_key, new_kv_size);
    printf("[DLFA-STATE-CLEANUP] Cleared state for seq_id=%d, state_key=%p, new_kv_size=%d\n",
           seq_id, state_key, new_kv_size);
}

} // namespace ggml_dl

// C linkage wrappers for proc_address registration
// NEW: Updated to accept seq_id parameter for proper slot isolation
extern "C" void ggml_dl_flash_attn_ext_dldnn_prepare_varlen_buffers(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int32_t seq_id) {
    ggml_dl::flash_attn_ext_dldnn_prepare_varlen_buffers(ctx, dst, seq_id);
}

extern "C" void ggml_dl_flash_attn_ext_dldnn_cleanup_varlen_data() {
    ggml_dl::flash_attn_ext_dldnn_cleanup_varlen_data();
}

extern "C" void ggml_dl_flash_attn_ext_dldnn_reset_accumulation_state() {
    ggml_dl::flash_attn_ext_dldnn_reset_accumulation_state();
}

// NEW: C linkage wrapper for clearing DLFA state for a specific sequence
// This is called when llama_memory_seq_rm is invoked to clean up KV cache
extern "C" void dlfa_clear_state(llama_seq_id seq_id, llama_pos new_kv_size) {
    ggml_dl::flash_attn_ext_dldnn_clear_state_for_seq_id(seq_id, new_kv_size);
}

// NEW: C linkage wrapper for set_flash_attn_runtime
// This is called once after set_inputs to compute host data (shared by all FA layers)
extern "C" void ggml_dl_flash_attn_ext_dldnn_set_runtime(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int seqlen_q_hint, int32_t seq_id) {
    GGML_UNUSED(ctx);  // Context not needed for set_flash_attn_runtime (only computes host data)
    set_flash_attn_runtime(dst, 1, seqlen_q_hint, seq_id);
}

// Backend variant for use with ggml_backend_reg_get_proc_address
extern "C" void flash_attn_ext_dldnn_set_runtime_backend(ggml_backend_t backend, ggml_tensor * dst, int seqlen_q_hint, int32_t seq_id) {
    GGML_UNUSED(backend);  // Backend not needed for set_flash_attn_runtime (only computes host data)
    set_flash_attn_runtime(dst, 1, seqlen_q_hint, seq_id);
}

// NEW: C linkage wrapper for copying host data from one key to another
// NOTE: After changing to use K->data as the state key, this function is essentially a no-op
// because in multi-GPU scenarios, all GPUs share the same KV cache, and therefore the same
// K->data pointer. The host data is already shared across all GPUs through the unified key.
// This function is kept for backward compatibility but no longer performs meaningful copying.
extern "C" void ggml_dl_flash_attn_ext_dldnn_copy_host_data(ggml_backend_cuda_context & ctx, void * src_mask, void * dst_mask) {
    GGML_UNUSED(ctx);  // Context not needed for copying host data

    // With K->data as the unified state key, src_mask and dst_mask should resolve to the same
    // K->data pointer in multi-GPU scenarios (since KV cache is shared). This copy operation
    // is now a no-op - both GPUs naturally access the same host data through the same key.
    //
    // We keep this function for backward compatibility with existing calling code, but it
    // no longer needs to do any actual copying.
    //
    // Future optimization: Remove this function and its call sites once the change is verified.

    std::lock_guard<std::mutex> lock(g_varlen_host_data_mutex);

    // With the new K->data key scheme, src_mask and dst_mask are no longer the keys.
    // The actual key is K->data, which is the same across all GPUs.
    // This function is now a no-op - both "source" and "destination" resolve to the same entry.
    //
    // We verify that both src_mask and dst_mask exist in the map (they should map to the same
    // K->data internally via set_flash_attn_runtime), but we don't need to copy anything.

    auto it_src = g_varlen_host_data_map.find(src_mask);
    auto it_dst = g_varlen_host_data_map.find(dst_mask);

    if (it_src != g_varlen_host_data_map.end() && it_dst != g_varlen_host_data_map.end()) {
        // Both entries exist - with K->data as the key, they should be the same entry or
        // point to equivalent data. No copying needed.
        GGML_DL_FATTN_DEBUG_PRINT("copy_host_data: no-op with K->data key scheme (src_mask=%p, dst_mask=%p)\n",
                src_mask, dst_mask);
    } else {
        // One or both entries don't exist - this shouldn't happen with the new scheme
        GGML_DL_FATTN_DEBUG_PRINT("copy_host_data: warning - src_mask=%p (%s), dst_mask=%p (%s)\n",
                src_mask, it_src != g_varlen_host_data_map.end() ? "found" : "not found",
                dst_mask, it_dst != g_varlen_host_data_map.end() ? "found" : "not found");
    }
}

// Backend variant for use with ggml_backend_reg_get_proc_address
// Note: This function is called from llama-context.cpp with (backend, node, seq_id)
// We extract the device from the tensor's buffer backend
extern "C" void flash_attn_ext_dldnn_prepare_varlen_buffers_backend(ggml_backend_t backend, ggml_tensor * dst, int32_t seq_id) {
    // Get device ID from backend
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    int device = 0;  // Default to device 0
    if (dev) {
        // Use ggml_backend_cuda_get_device_id if available, otherwise use default
        // For now, extract device from string name
        const char * dev_name = ggml_backend_dev_name(dev);
        if (dev_name && strstr(dev_name, "CUDA") != nullptr) {
            // Parse device ID from name (format: "CUDA0", "CUDA1", etc.)
            device = atoi(dev_name + 4);
        }
    }
    // Create CUDA context with device ID
    ggml_backend_cuda_context ctx(device);
    ggml_dl::flash_attn_ext_dldnn_prepare_varlen_buffers(ctx, dst, seq_id);
}

#endif // GGML_USE_DLFA