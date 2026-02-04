# GGML_OP_MOE_SUM Performance Benchmark

## Overview

This tool benchmarks the `GGML_OP_MOE_SUM` operator, which is used for aggregating expert outputs in Mixture of Experts (MoE) models.

## What is moe_sum?

The `moe_sum` operator reduces a 3D tensor `[hidden_dim, n_expert_used, n_tokens]` to a 2D tensor `[hidden_dim, n_tokens]` by summing along the expert dimension:

```
output[d][t] = Σ input[d][k][t] for k in 0..n_expert_used-1
```

This is a key operation in MoE models where each token is processed by multiple experts (top-k routing), and their outputs need to be combined.

## Build

```bash
cd build
cmake ..
make bench-moe-sum
```

## Usage

```bash
./bench-moe-sum [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-h, --hidden <n>` | Hidden dimension | 4096 |
| `-e, --experts <n>` | Number of experts (top-k) | 4 |
| `-t, --tokens <n>` | Number of tokens | 256 |
| `-i, --iterations <n>` | Number of iterations | 100 |
| `-v, --verbose` | Enable verbose output | - |
| `--help` | Show help message | - |

### Example

```bash
# Benchmark with default settings
./bench-moe-sum

# Benchmark with custom dimensions
./bench-moe-sum --hidden 2048 --experts 8 --tokens 512 --iterations 50 --verbose
```

## Output

The benchmark compares two approaches:

1. **moe_sum**: The specialized `GGML_OP_MOE_SUM` operator (DLCU only)
2. **add_loop**: Traditional approach using sequential ADD operations

Example output:
```
=================================================
GGML_OP_MOE_SUM Performance Benchmark
=================================================
Configuration:
  Hidden dimension: 4096
  Number of experts: 4
  Number of tokens: 256
  Iterations: 100
=================================================

Using DLCU backend

Results (averaged over 100 iterations):
=================================================
  moe_sum:       12.34 ms  (  123.40 us/iter)
  add_loop:      45.67 ms  (  456.70 us/iter)

  Speedup:       3.70x
  moe_sum bandwidth: 45.67 GB/s
=================================================
```

## Implementation Notes

The `moe_sum` operator is only available when `GGML_USE_DLCU` is defined. On other configurations, only the `add_loop` benchmark will run.

See `ggml/src/ggml-dlcu/dl-moesum.cu` for the GPU kernel implementation, which includes:
- Vectorized FP16 kernel (16-element loads)
- Warp-per-token kernels
- Specialized kernels for topk = 2, 4, 8, 9
- General fallback kernel
