# Orpheus-TTS Golden References

This directory contains golden reference data for Orpheus-TTS testing.

## Directory Structure

```
golden/orpheus/
├── tokens/          # Token sequences (JSON format)
├── audio/           # Reference audio files (WAV format)
├── metrics/         # Expected audio metrics (JSON format)
└── scripts/         # Generation scripts
```

## Token Sequences

Token sequences are stored as JSON files with the following structure:

```json
{
  "name": "zeros_100frames",
  "n_frames": 100,
  "seed": null,
  "heads": [
    [0, 0, 0, ...],  // head0: 4 tokens per frame
    [0, 0, ...],     // head1: 2 tokens per frame
    [0, ...]         // head2: 1 token per frame
  ]
}
```

### Available Token Sequences

| Name | Description | Purpose |
|------|-------------|---------|
| `zeros_100frames.json` | All zeros | Minimal activation test |
| `ones_100frames.json` | All ones | Basic non-zero test |
| `sequential_100frames.json` | 0,1,2,...,4095 | Codebook coverage test |
| `random_seed42_100frames.json` | Fixed seed random | Deterministic random test |

## Audio Metrics

Expected metrics are stored as JSON files:

```json
{
  "name": "zeros_100frames",
  "metrics": {
    "neg_ratio": {"min": 35.0, "max": 65.0},
    "dc_bias": {"min": -0.01, "max": 0.01},
    "energy_0_200Hz": {"max": 40.0},
    "peak_amplitude": {"min": 0.1, "max": 1.0}
  }
}
```

### Metric Thresholds

| Metric | Target | Acceptable Range | Description |
|--------|--------|------------------|-------------|
| `neg_ratio` | 50% | 35-65% | Percentage of negative samples (symmetric waveform) |
| `dc_bias` | 0 | \|mean\| < 0.01 | DC offset after tanh |
| `energy_0_200Hz` | <20% | <40% | Low frequency energy (not dominant) |
| `peak_amplitude` | 0.85 | 0.7-0.95 | Peak signal level |

## Generating Golden References

### From chatllm.cpp (Golden Reference)

```bash
# Build chatllm.cpp
cd /path/to/chatllm.cpp
cmake -B build && cmake --build build -j$(nproc)

# Generate reference audio
./build/bin/chatllm -m orpheus-3b.gguf --model-vocoder snac-24khz.gguf \
    -p "Hello world" -o golden_hello.wav

# Extract metrics
python3 scripts/extract_metrics.py golden_hello.wav metrics/hello.json
```

### From py-gguf TTS.cpp (Alternative Reference)

```bash
cd /path/to/TTS.cpp/py-gguf
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Generate reference audio
python3 tts_orpheus.py --model orpheus-3b.gguf --vocoder snac-24khz.gguf \
    -p "Hello world" -o golden_hello_py.wav
```

### Vocoder-Only Testing (No LLM Required)

```bash
# Test vocoder with deterministic tokens
./build/bin/llama-orpheus-tts \
    --model-vocoder snac-24khz-f16.gguf \
    --test-vocoder \
    --test-frames 100 \
    --seed 42 \
    -o test_vocoder_seed42.wav
```

## Test Case Categories

### 1. Unit Tests (No Model Required)
- Snake activation function
- Conv1D / ConvTranspose1D operations
- Codebook lookup
- These tests use synthetic data and verify mathematical correctness

### 2. Vocoder Tests (SNAC Model Only)
- Random token sequences
- Deterministic token sequences
- Audio quality metrics validation
- These tests verify the SNAC vocoder produces valid audio

### 3. End-to-End Tests (Full Pipeline)
- Text-to-speech generation
- Voice selection
- Long-form synthesis
- These tests verify the complete TTS pipeline

## Adding New Golden References

1. **Generate the reference output**:
   ```bash
   ./build/bin/llama-orpheus-tts -m model.gguf --model-vocoder vocoder.gguf \
       -p "Test phrase" -o output.wav
   ```

2. **Extract metrics**:
   ```bash
   python3 scripts/extract_metrics.py output.wav metrics/test_phrase.json
   ```

3. **Store reference**:
   ```bash
   cp output.wav audio/test_phrase.wav
   ```

4. **Document the test case** in this README

## Cross-Platform Considerations

- Floating-point differences may cause small variations in output
- Use tolerance-based comparison (not bit-exact)
- Focus on audio metrics rather than raw samples
- Allow platform-specific thresholds in metrics
