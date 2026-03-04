#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert SNAC vocoder model to GGUF format.

SNAC (Multi-Scale Neural Audio Codec) is a neural audio codec that converts
discrete tokens to waveform. This script converts the snac_24khz model from
HuggingFace (hubertsiuzdak/snac_24khz) to GGUF format.

Usage:
    python convert_hf_to_gguf_snac.py /path/to/snac-model --outfile snac-24khz-f16.gguf --outtype f16

Example:
    # Download model first
    huggingface-cli download hubertsiuzdak/snac_24khz --local-dir models/snac

    # Convert to GGUF
    python convert_hf_to_gguf_snac.py models/snac --outfile models/snac/snac-24khz-f16.gguf
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Add gguf-py to path
if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent.parent.parent / 'gguf-py'))

import gguf

logger = logging.getLogger("snac-to-gguf")

# SNAC 24kHz model constants (from hubertsiuzdak/snac_24khz)
SNAC_SAMPLING_RATE = 24000
SNAC_DECODER_DIM = 1536
SNAC_DECODER_RATES = [8, 8, 3, 2]
SNAC_CODEBOOK_SIZE = 4096
SNAC_CODEBOOK_DIM = 8
SNAC_VQ_STRIDES = [8, 4, 2, 1]
SNAC_N_QUANTIZERS = 4
SNAC_NOISE = True
SNAC_DEPTHWISE = True


class SnacConverter:
    """Convert SNAC vocoder model to GGUF format."""

    def __init__(
        self,
        dir_model: Path,
        fname_out: Path,
        ftype: gguf.LlamaFileType = gguf.LlamaFileType.MOSTLY_F16,
    ):
        self.dir_model = dir_model
        self.fname_out = fname_out
        self.ftype = ftype

        # Load model config
        self.config = self._load_config()

        # Initialize GGUF writer
        self.gguf_writer = gguf.GGUFWriter(
            path=fname_out,
            arch="snac",
        )

    def _load_config(self) -> dict[str, Any]:
        """Load SNAC configuration from config.json."""
        config_path = self.dir_model / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                logger.info(f"Loaded config from {config_path}")
                return config

        logger.warning("config.json not found, using defaults")
        return {
            "sampling_rate": SNAC_SAMPLING_RATE,
            "decoder_dim": SNAC_DECODER_DIM,
            "decoder_rates": SNAC_DECODER_RATES,
            "codebook_size": SNAC_CODEBOOK_SIZE,
            "codebook_dim": SNAC_CODEBOOK_DIM,
            "vq_strides": SNAC_VQ_STRIDES,
            "noise": SNAC_NOISE,
            "depthwise": SNAC_DEPTHWISE,
        }

    def _load_tensors(self) -> dict[str, torch.Tensor]:
        """Load all tensors from the model.

        Handles weight normalization decomposition:
        - parametrizations.weight.original0 * original1 -> combined weight
        - weight_g * weight_v / ||weight_v|| -> combined weight

        Also maps PyTorch tensor names to GGUF format.
        """
        raw_tensors = {}

        # Load from pytorch_model.bin
        pytorch_file = self.dir_model / "pytorch_model.bin"
        if pytorch_file.exists():
            data = torch.load(pytorch_file, map_location="cpu", weights_only=True)
            raw_tensors.update(data)
            logger.info(f"Loaded {len(data)} tensors from {pytorch_file}")

        # Also try safetensors format
        safetensor_files = list(self.dir_model.glob("*.safetensors"))
        if safetensor_files:
            from safetensors.torch import load_file
            for sf in safetensor_files:
                data = load_file(sf)
                raw_tensors.update(data)
                logger.info(f"Loaded {len(data)} tensors from {sf}")

        if not raw_tensors:
            raise ValueError(f"No model weights found in {self.dir_model}")

        # Process tensors: combine weight normalization and remap names
        tensors = {}
        weight_norm_pairs = {}  # Maps base_name -> (g, v) for parametrizations format
        weight_gv_pairs = {}    # Maps base_name -> (g, v) for _g/_v format

        for name, tensor in raw_tensors.items():
            # Skip encoder tensors - we only need decoder for TTS
            if name.startswith("encoder."):
                continue

            if name.endswith(".parametrizations.weight.original0"):
                base_name = name[:-33]  # Remove ".parametrizations.weight.original0" (33 chars)
                if base_name not in weight_norm_pairs:
                    weight_norm_pairs[base_name] = [None, None]
                weight_norm_pairs[base_name][0] = tensor
            elif name.endswith(".parametrizations.weight.original1"):
                base_name = name[:-33]  # Remove ".parametrizations.weight.original1" (33 chars)
                if base_name not in weight_norm_pairs:
                    weight_norm_pairs[base_name] = [None, None]
                weight_norm_pairs[base_name][1] = tensor
            elif name.endswith("_g"):
                # weight_g format (snac_24khz uses this)
                base_name = name[:-2]  # Remove "_g"
                if base_name not in weight_gv_pairs:
                    weight_gv_pairs[base_name] = [None, None]
                weight_gv_pairs[base_name][0] = tensor
            elif name.endswith("_v"):
                # weight_v format (snac_24khz uses this)
                base_name = name[:-2]  # Remove "_v"
                if base_name not in weight_gv_pairs:
                    weight_gv_pairs[base_name] = [None, None]
                weight_gv_pairs[base_name][1] = tensor
            else:
                # Non-weight-normalized tensor (e.g., bias, alpha)
                gguf_name = self._map_tensor_name(name)
                if gguf_name:
                    tensors[gguf_name] = tensor

        # Combine weight normalization pairs (parametrizations format)
        for base_name, (original0, original1) in weight_norm_pairs.items():
            # Strip trailing dot from base_name if present
            if base_name.endswith("."):
                base_name = base_name[:-1]

            gguf_name = self._map_tensor_name(base_name + ".weight")
            if gguf_name is None:
                continue

            if original0 is not None and original1 is not None:
                # Weight normalization: w = g * (v / ||v||)
                v = original1
                v_norm = torch.norm(v.reshape(v.shape[0], -1), dim=1, keepdim=True)
                for _ in range(v.dim() - 2):
                    v_norm = v_norm.unsqueeze(-1)
                combined = original0 * v / v_norm
                tensors[gguf_name] = combined
                logger.debug(f"Combined weight norm for {gguf_name}: {combined.shape}")
            elif original1 is not None:
                tensors[gguf_name] = original1
            elif original0 is not None:
                tensors[gguf_name] = original0

        # Combine weight_g/weight_v pairs (snac_24khz format)
        for base_name, (g, v) in weight_gv_pairs.items():
            gguf_name = self._map_tensor_name(base_name)
            if gguf_name is None:
                continue

            if g is not None and v is not None:
                # Weight normalization: w = g * (v / ||v||)
                v_norm = torch.norm(v.reshape(v.shape[0], -1), dim=1, keepdim=True)
                for _ in range(v.dim() - 2):
                    v_norm = v_norm.unsqueeze(-1)
                combined = g * v / v_norm
                tensors[gguf_name] = combined
                logger.debug(f"Combined _g/_v for {gguf_name}: {combined.shape}")
            elif v is not None:
                tensors[gguf_name] = v
            elif g is not None:
                tensors[gguf_name] = g

        logger.info(f"Processed {len(tensors)} tensors (from {len(raw_tensors)} raw)")
        return tensors

    def _map_tensor_name(self, pytorch_name: str) -> str | None:
        """Map PyTorch tensor names to GGUF format.

        PyTorch structure (snac_24khz from hubertsiuzdak/snac_24khz):
        Two naming formats exist:
        - Format 1: decoder.model.layers.X.*
        - Format 2: decoder.model.X.* (direct indexing)

        Structure:
        - decoder.model.{0/layers.0}.*              -> Input conv (depthwise)
        - decoder.model.{1/layers.1}.*              -> Up conv (1x1)
        - decoder.model.{2-5/layers.2-5}.block.*    -> Decoder layers
        - decoder.model.{6/layers.6}.alpha          -> Snake alpha (final)
        - decoder.model.{7/layers.7}.*              -> Output conv
        - quantizer.quantizers.{0-2}.*              -> Quantizers (3 quantizers)

        GGUF format:
        - decoder.in_conv.*
        - decoder.up_conv.*
        - decoder.layers.{i}.*
        - decoder.alpha_out
        - decoder.out_conv.*
        - quantizers.{i}.*
        """
        # Remove parametrizations suffix for mapping
        name = pytorch_name.replace(".parametrizations.weight.original0", ".weight")
        name = name.replace(".parametrizations.weight.original1", ".weight")

        # Normalize naming: convert decoder.model.X to decoder.model.layers.X
        # This handles both naming formats
        direct_match = re.match(r"decoder\.model\.(\d+)\.(.+)", name)
        if direct_match and not name.startswith("decoder.model.layers."):
            layer_idx = direct_match.group(1)
            rest = direct_match.group(2)
            name = f"decoder.model.layers.{layer_idx}.{rest}"

        # Input conv (decoder.model.layers.0)
        if name.startswith("decoder.model.layers.0."):
            rest = name[len("decoder.model.layers.0."):]
            return f"decoder.in_conv.{rest}"

        # Up conv (decoder.model.layers.1)
        if name.startswith("decoder.model.layers.1."):
            rest = name[len("decoder.model.layers.1."):]
            return f"decoder.up_conv.{rest}"

        # Output alpha (decoder.model.layers.6.alpha)
        if name == "decoder.model.layers.6.alpha":
            return "decoder.alpha_out"

        # Output conv (decoder.model.layers.7)
        if name.startswith("decoder.model.layers.7."):
            rest = name[len("decoder.model.layers.7."):]
            return f"decoder.out_conv.{rest}"

        # Decoder layers (decoder.model.layers.{2,3,4,5}.block.layers.*)
        dec_match = re.match(r"decoder\.model\.layers\.([2-5])\.block\.layers\.(.+)", name)
        if dec_match:
            layer_idx = int(dec_match.group(1)) - 2  # Map 2->0, 3->1, 4->2, 5->3
            rest = dec_match.group(2)
            return self._map_decoder_layer_tensor(layer_idx, rest)

        # Quantizers (quantizer.quantizers.{0-2}.*)
        quant_match = re.match(r"quantizer\.quantizers\.(\d+)\.(.+)", name)
        if quant_match:
            quant_idx = quant_match.group(1)
            rest = quant_match.group(2)
            return f"quantizers.{quant_idx}.{rest}"

        return None

    def _map_decoder_layer_tensor(self, layer_idx: int, rest: str) -> str | None:
        """Map decoder layer tensor names to GGUF format.

        Structure (actual snac_24khz):
        - block.0.alpha            -> .alpha (snake alpha)
        - block.1.bias             -> .conv_t.bias
        - block.1.weight           -> .conv_t.weight
        - block.2.linear.weight    -> .noise_proj.weight
        - block.{3,4,5}.block.*    -> .residual_units.{0,1,2}.*
        """
        # Snake alpha
        if rest == "0.alpha":
            return f"decoder.layers.{layer_idx}.alpha"

        # ConvTranspose1d
        if rest.startswith("1."):
            suffix = rest[2:]
            if suffix == "bias":
                return f"decoder.layers.{layer_idx}.conv_t.bias"
            elif suffix == "weight" or suffix.endswith(".weight"):
                return f"decoder.layers.{layer_idx}.conv_t.weight"

        # Noise projection
        if rest.startswith("2.linear."):
            suffix = rest[9:]
            if suffix == "weight":
                return f"decoder.layers.{layer_idx}.noise_proj.weight"

        # Residual units ({3,4,5}.block.layers.*)
        res_match = re.match(r"([3-5])\.block\.layers\.(.+)", rest)
        if res_match:
            unit_idx = int(res_match.group(1)) - 3  # Map 3->0, 4->1, 5->2
            inner_rest = res_match.group(2)
            return self._map_residual_unit_tensor(layer_idx, unit_idx, inner_rest)

        return None

    def _map_residual_unit_tensor(self, layer_idx: int, unit_idx: int, rest: str) -> str | None:
        """Map residual unit tensor names to GGUF format.

        Structure (actual snac_24khz):
        - 0.alpha            -> .in_alpha
        - 1.bias             -> .in_conv.bias
        - 1.weight           -> .in_conv.weight
        - 2.alpha            -> .out_alpha
        - 3.bias             -> .out_conv.bias
        - 3.weight           -> .out_conv.weight
        """
        prefix = f"decoder.layers.{layer_idx}.residual_units.{unit_idx}"

        if rest == "0.alpha":
            return f"{prefix}.in_alpha"
        if rest.startswith("1."):
            suffix = rest[2:]
            if suffix == "bias":
                return f"{prefix}.in_conv.bias"
            elif suffix == "weight" or suffix.endswith(".weight"):
                return f"{prefix}.in_conv.weight"
        if rest == "2.alpha":
            return f"{prefix}.out_alpha"
        if rest.startswith("3."):
            suffix = rest[2:]
            if suffix == "bias":
                return f"{prefix}.out_conv.bias"
            elif suffix == "weight" or suffix.endswith(".weight"):
                return f"{prefix}.out_conv.weight"

        return None

    def _set_gguf_parameters(self):
        """Set GGUF metadata for the SNAC model."""
        self.gguf_writer.add_name("SNAC-24kHz")

        # SNAC-specific metadata
        cfg = self.config
        self.gguf_writer.add_uint32("snac.sampling_rate", cfg.get("sampling_rate", SNAC_SAMPLING_RATE))
        self.gguf_writer.add_uint32("snac.decoder_dim", cfg.get("decoder_dim", SNAC_DECODER_DIM))
        self.gguf_writer.add_uint32("snac.codebook_size", cfg.get("codebook_size", SNAC_CODEBOOK_SIZE))
        self.gguf_writer.add_uint32("snac.codebook_dim", cfg.get("codebook_dim", SNAC_CODEBOOK_DIM))
        self.gguf_writer.add_uint32("snac.n_quantizers", len(cfg.get("vq_strides", SNAC_VQ_STRIDES)))
        self.gguf_writer.add_bool("snac.noise", cfg.get("noise", SNAC_NOISE))
        self.gguf_writer.add_bool("snac.depthwise", cfg.get("depthwise", SNAC_DEPTHWISE))

        # Decoder rates as array
        decoder_rates = cfg.get("decoder_rates", SNAC_DECODER_RATES)
        for i, rate in enumerate(decoder_rates):
            self.gguf_writer.add_uint32(f"snac.decoder_rate_{i}", rate)

        # VQ strides as array
        vq_strides = cfg.get("vq_strides", SNAC_VQ_STRIDES)
        for i, stride in enumerate(vq_strides):
            self.gguf_writer.add_uint32(f"snac.vq_stride_{i}", stride)

    def convert(self):
        """Perform the conversion."""
        logger.info(f"Converting SNAC model from {self.dir_model}")

        # Set metadata
        self._set_gguf_parameters()
        logger.info("Set GGUF parameters")

        # Load and process tensors
        tensors = self._load_tensors()

        # Write tensors to GGUF
        count = 0
        for gguf_name, tensor in tensors.items():
            data = tensor.numpy()
            original_shape = data.shape

            # Special handling for out_conv.weight: needs transpose
            # PyTorch: [1, 64, 7] = [out, in, kernel]
            # GGML ggml_conv_1d expects: ne[0]=K, ne[1]=IC, ne[2]=OC = [kernel, in, out]
            # For our case: K=7, IC=64, OC=1 -> numpy shape [7, 64, 1]
            if gguf_name == "decoder.out_conv.weight":
                # PyTorch shape: [out, in, kernel] = [1, 64, 7]
                # Transpose to: [kernel, in, out] = [7, 64, 1]
                # GGML expects ne[0]=K=7, ne[1]=IC=64, ne[2]=OC=1
                data = np.transpose(data, (2, 1, 0))  # [kernel, in, out]
                logger.debug(f"Transposed out_conv.weight: {original_shape} -> {data.shape}")
            else:
                # Squeeze all dimensions of size 1 (e.g., [out, 1, k] -> [out, k], [out, in, 1] -> [out, in])
                # But preserve 1D tensors like bias [n] and don't turn [1] into scalar
                data = np.squeeze(data)  # Removes all dimensions of size 1
                # Ensure we don't turn a 1D tensor into a scalar
                if data.ndim == 0:
                    data = data.reshape(1)
                # For 3D tensors that should remain 3D (like ConvTranspose1D), restore them
                if len(original_shape) == 3 and original_shape[1] != 1 and original_shape[2] != 1:
                    data = data.reshape(original_shape)

            self.gguf_writer.add_tensor(gguf_name, data)
            count += 1
            if count <= 10 or count % 20 == 0:
                logger.debug(f"Added tensor: {gguf_name} {tensor.shape} -> {data.shape}")

        logger.info(f"Wrote {count} tensors to GGUF")

        # Write GGUF file
        self.gguf_writer.write_header_to_file()
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file()
        self.gguf_writer.close()

        logger.info(f"GGUF file written to {self.fname_out}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SNAC vocoder model to GGUF format"
    )
    parser.add_argument(
        "model",
        type=Path,
        help="Path to the SNAC model directory (containing pytorch_model.bin)",
    )
    parser.add_argument(
        "--outfile",
        type=Path,
        default=None,
        help="Output GGUF file path (default: snac-24khz-f16.gguf)",
    )
    parser.add_argument(
        "--outtype",
        choices=["f16", "bf16", "q8_0", "q4_k_m", "q5_k_m"],
        default="f16",
        help="Output tensor type (default: f16)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Determine output type
    ftype_map = {
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "q4_k_m": gguf.LlamaFileType.MOSTLY_Q4_K_M,
        "q5_k_m": gguf.LlamaFileType.MOSTLY_Q5_K_M,
    }
    ftype = ftype_map[args.outtype]

    # Determine output filename
    if args.outfile is None:
        args.outfile = Path(f"snac-24khz-{args.outtype}.gguf")

    # Run conversion
    converter = SnacConverter(
        dir_model=args.model,
        fname_out=args.outfile,
        ftype=ftype,
    )
    converter.convert()

    logger.info("Conversion complete!")


if __name__ == "__main__":
    main()
