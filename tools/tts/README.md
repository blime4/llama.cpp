# llama.cpp/example/tts

This example demonstrates Text To Speech features using multiple TTS models.

## Available TTS Systems

1. **OuteTTS** - Basic TTS from [OuteAI](https://www.outeai.com/)
2. **Orpheus-TTS** - Emotion-expressive TTS (see dedicated section below)

---

## OuteTTS Quickstart

If you have built llama.cpp with SSL support you can simply run the
following command and the required models will be downloaded automatically:

```console
$ build/bin/llama-tts --tts-oute-default -p "Hello world" && aplay output.wav
```

For details about the models and how to convert them to the required format
see the following sections.

### OuteTTS Model Conversion

Checkout or download the model that contains the LLM model:

```console
$ pushd models
$ git clone --branch main --single-branch --depth 1 https://huggingface.co/OuteAI/OuteTTS-0.2-500M
$ cd OuteTTS-0.2-500M && git lfs install && git lfs pull
$ popd
```

Convert the model to .gguf format:

```console
(venv) python convert_hf_to_gguf.py models/OuteTTS-0.2-500M \
    --outfile models/outetts-0.2-0.5B-f16.gguf --outtype f16
```

The generated model will be `models/outetts-0.2-0.5B-f16.gguf`.

We can optionally quantize this to Q8_0 using the following command:

```console
$ build/bin/llama-quantize models/outetts-0.2-0.5B-f16.gguf \
    models/outetts-0.2-0.5B-q8_0.gguf q8_0
```

Next we do something similar for the audio decoder. First download or checkout
the model for the voice decoder:

```console
$ pushd models
$ git clone --branch main --single-branch --depth 1 https://huggingface.co/novateur/WavTokenizer-large-speech-75token
$ cd WavTokenizer-large-speech-75token && git lfs install && git lfs pull
$ popd
```

This model file is a PyTorch checkpoint (.ckpt) and we first need to convert it to
huggingface format:

```console
(venv) python tools/tts/convert_pt_to_hf.py \
    models/WavTokenizer-large-speech-75token/wavtokenizer_large_speech_320_24k.ckpt
```

Then we can convert the huggingface format to gguf:

```console
(venv) python convert_hf_to_gguf.py models/WavTokenizer-large-speech-75token \
    --outfile models/wavtokenizer-large-75-f16.gguf --outtype f16
```

### Running OuteTTS

With both of the models generated, the LLM model and the voice decoder model,
we can run the example:

```console
$ build/bin/llama-tts -m ./models/outetts-0.2-0.5B-q8_0.gguf \
    -mv ./models/wavtokenizer-large-75-f16.gguf \
    -p "Hello world"
...
main: audio written to file 'output.wav'
```

The output.wav file will contain the audio of the prompt. This can be heard
by playing the file with a media player. On Linux the following command will
play the audio:

```console
$ aplay output.wav
```

### Running OuteTTS with llama-server

Running this example with `llama-server` is also possible and requires two
server instances to be started. One will serve the LLM model and the other
will serve the voice decoder model.

The LLM model server can be started with the following command:

```console
$ ./build/bin/llama-server -m ./models/outetts-0.2-0.5B-q8_0.gguf --port 8020
```

And the voice decoder model server can be started using:

```console
./build/bin/llama-server -m ./models/wavtokenizer-large-75-f16.gguf --port 8021 --embeddings --pooling none
```

Then we can run [tts-outetts.py](tts-outetts.py) to generate the audio.

First create a virtual environment for python and install the required
dependencies (this in only required to be done once):

```console
$ python3 -m venv venv
$ source venv/bin/activate
(venv) pip install requests numpy
```

And then run the python script using:

```console
(venv) python ./tools/tts/tts-outetts.py http://localhost:8020 http://localhost:8021 "Hello world"
spectrogram generated: n_codes: 90, n_embd: 1282
converting to audio ...
audio generated: 28800 samples
audio written to file "output.wav"
```

---

## Orpheus-TTS

Orpheus-TTS is an emotion-expressive Text-to-Speech model that uses:
- A LLaMA-based LLM to generate discrete audio tokens
- A SNAC neural vocoder to convert tokens to waveform

### Building Orpheus-TTS

```bash
# From llama.cpp root directory
cmake -B build -DLLAMA_CURL=OFF
cmake --build build --target llama-orpheus-tts -j$(nproc)

# Executable location: build/bin/llama-orpheus-tts
```

### Model Conversion

#### 1. Orpheus LLM Model

Download the Orpheus model:

```bash
# From llama.cpp root directory
cd models
huggingface-cli download Canstralian/Orpheus-3b-0.1-ft --local-dir orpheus-3b
cd ..
```

Convert to GGUF format:

```bash
python tools/tts/convert_hf_to_gguf_orpheus.py models/orpheus-3b \
    --outfile models/orpheus-3b-f16.gguf --outtype f16
```

Optionally quantize:

```bash
./build/bin/llama-quantize models/orpheus-3b-f16.gguf \
    models/orpheus-3b-q8_0.gguf q8_0
```

#### 2. SNAC Vocoder Model

Download the SNAC vocoder:

```bash
cd models
huggingface-cli download hubertsiuzdak/snac_24khz --local-dir snac_24khz
cd ..
```

Convert to GGUF format:

```bash
python tools/tts/convert_hf_to_gguf_snac.py models/snac_24khz \
    --outfile models/snac-24khz-f16.gguf --outtype f16
```

### Running Orpheus-TTS

#### Basic Usage

```bash
./build/bin/llama-orpheus-tts \
    -m models/orpheus-3b-f16.gguf \
    --model-vocoder models/snac-24khz-f16.gguf \
    -p "Hello, how are you?" \
    -o output.wav \
    -t 8
```

#### With Voice Selection

```bash
./build/bin/llama-orpheus-tts \
    -m models/orpheus-3b-f16.gguf \
    --model-vocoder models/snac-24khz-f16.gguf \
    -p "Hello, how are you today?" \
    -v tara \
    -o output.wav
```

Available voices include: `tara`, `leo`, `zoe`, `jad`, `bria`, `leah`, `dan`, `mimi`, `jess`, `carly`, etc.

#### Test Vocoder Only (Without LLM)

You can test the SNAC vocoder independently without running the LLM:

```bash
./build/bin/llama-orpheus-tts \
    --model-vocoder models/snac-24khz-f16.gguf \
    --test-vocoder \
    -o test.wav
```

This generates audio from random tokens, useful for verifying the vocoder works correctly.

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model` | Path to LLM GGUF model | Required |
| `--model-vocoder` | Path to SNAC vocoder GGUF | Required |
| `-p, --prompt` | Text to synthesize | Required |
| `-o, --output` | Output WAV file | output.wav |
| `-v, --voice` | Voice/speaker name | None |
| `-t, --threads` | Number of threads | Auto |
| `-n, --n-predict` | Max tokens to generate | 2048 |
| `--temp` | Temperature | 0.1 |
| `--top-k` | Top-k sampling | 40 |
| `--top-p` | Top-p sampling | 0.9 |
| `--test-vocoder` | Test vocoder only | False |
| `--test-frames` | Frames for vocoder test | 10 |

### TTS Acceptance Criteria

The following metrics are used to validate TTS output quality.

#### Zero Crossing Rate (ZCR) - Primary Quality Metric

ZCR measures how often the audio signal crosses zero. Clean speech has a characteristic ZCR range.

| Quality Level | ZCR Range | Description |
|---------------|-----------|-------------|
| **GOOD** | 0.02 - 0.07 | Clean, natural speech |
| **MARGINAL** | 0.07 - 0.10 | Acceptable, slight noise |
| **DISTORTED** | > 0.10 | Noisy, artifacts present |

#### Test Commands

```bash
# Generate test audio
./build/bin/llama-orpheus-tts \
  -m models/orpheus-3b-f16.gguf \
  --model-vocoder models/snac-24khz-f16.gguf \
  -p "Hello, how are you doing today?" \
  -o test_output.wav \
  --temp 0.1 --top-k 40 --top-p 0.9

# Calculate ZCR
python3 -c "
import numpy as np
import wave
with wave.open('test_output.wav', 'rb') as f:
    audio = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(float) / 32768
zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
print(f'ZCR: {zcr:.4f} ({\"GOOD\" if zcr < 0.07 else \"MARGINAL\" if zcr < 0.10 else \"DISTORTED\"})')
"
```

### Complete Example

```bash
./build/bin/llama-orpheus-tts \
    -m models/orpheus-3b-f16.gguf \
    --model-vocoder models/snac-24khz-f16.gguf \
    -p "This is a test of the Orpheus text to speech system. It supports emotional expression and natural prosody." \
    -v tara \
    -o test_output.wav \
    -t 8 \
    -n 4096 \
    --temp 0.1 \
    --top-k 40 \
    --top-p 0.9
```
