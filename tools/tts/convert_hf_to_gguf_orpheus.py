#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Orpheus-TTS LLM model to GGUF format.

This script only handles the LLM part of the Orpheus-TTS model.
For SNAC vocoder conversion, use convert_hf_to_gguf_snac.py instead.

Usage:
    python convert_hf_to_gguf_orpheus.py /path/to/orpheus-model --outfile orpheus-3b-f16.gguf --outtype f16
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Add gguf-py to path
if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent.parent.parent / 'gguf-py'))

import gguf

logger = logging.getLogger("orpheus-to-gguf")

# Orpheus-TTS model constants
ORPHEUS_VOCAB_SIZE = 156940
ORPHEUS_N_ATTN_HEADS = 24
ORPHEUS_N_KV_ATTN_HEADS = 8
ORPHEUS_HEAD_DIM = 128
ORPHEUS_HIDDEN_SIZE = 3072
ORPHEUS_N_LAYERS = 28
ORPHEUS_STOPPING_TOKEN_ID = 128258
ORPHEUS_EOS_TOKEN_ID = 128001
ORPHEUS_BOS_TOKEN_ID = 128000


class OrpheusConverter:
    """Convert Orpheus-TTS LLM model to GGUF format."""

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
        self.llm_config = self._load_llm_config()

        # Initialize GGUF writer with custom architecture
        self.gguf_writer = gguf.GGUFWriter(
            path=fname_out,
            arch="orpheus",
        )

    def _load_llm_config(self) -> dict[str, Any]:
        """Load LLM configuration from config.json."""
        config_path = self.dir_model / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _load_tensors(self) -> dict[str, torch.Tensor]:
        """Load all tensors from the model."""
        tensors = {}

        # Load LLM tensors (safetensors format)
        safetensor_files = list(self.dir_model.glob("*.safetensors"))
        if safetensor_files:
            from safetensors.torch import load_file
            for sf in safetensor_files:
                data = load_file(sf)
                tensors.update(data)

        # Also try pytorch_model.bin
        pytorch_files = list(self.dir_model.glob("pytorch_model*.bin"))
        for pf in pytorch_files:
            data = torch.load(pf, map_location="cpu", weights_only=True)
            tensors.update(data)

        return tensors

    def _get_llm_tensor_name(self, hf_name: str) -> str | None:
        """Map HuggingFace tensor names to GGUF format for LLM."""
        # Handle prefix variations
        name = hf_name
        if name.startswith("model."):
            name = name[6:]  # Remove "model." prefix

        # Embedding
        if name == "embed_tokens.weight":
            return "token_embd.weight"

        # Output norm and LM head
        if name == "norm.weight":
            return "output_norm.weight"
        if name == "lm_head.weight":
            return "output.weight"  # lm_head -> output

        # Layer mappings
        import re
        layer_match = re.match(r"layers\.(\d+)\.(.+)", name)
        if layer_match:
            layer_idx = layer_match.group(1)
            rest = layer_match.group(2)

            # Attention layers
            if rest == "self_attn.q_proj.weight":
                return f"blk.{layer_idx}.attn_q.weight"
            if rest == "self_attn.k_proj.weight":
                return f"blk.{layer_idx}.attn_k.weight"
            if rest == "self_attn.v_proj.weight":
                return f"blk.{layer_idx}.attn_v.weight"
            if rest == "self_attn.o_proj.weight":
                return f"blk.{layer_idx}.attn_output.weight"

            # Norm layers
            if rest == "input_layernorm.weight":
                return f"blk.{layer_idx}.attn_norm.weight"
            if rest == "post_attention_layernorm.weight":
                return f"blk.{layer_idx}.ffn_norm.weight"

            # MLP layers
            if rest == "mlp.gate_proj.weight":
                return f"blk.{layer_idx}.ffn_gate.weight"
            if rest == "mlp.up_proj.weight":
                return f"blk.{layer_idx}.ffn_up.weight"
            if rest == "mlp.down_proj.weight":
                return f"blk.{layer_idx}.ffn_down.weight"

        return None

    def _permute_qk(self, tensor: torch.Tensor, n_head: int, n_head_kv: int) -> torch.Tensor:
        """Permute Q/K tensors for LLaMA-style models."""
        if n_head_kv is not None and n_head != n_head_kv:
            n_head = n_head_kv
        return (tensor.reshape(n_head, 2, tensor.shape[0] // n_head // 2, *tensor.shape[1:])
                .swapaxes(1, 2)
                .reshape(tensor.shape))

    def _set_gguf_parameters(self):
        """Set GGUF metadata for the model."""
        self.gguf_writer.add_name("Orpheus-TTS")
        self.gguf_writer.add_vocab_size(ORPHEUS_VOCAB_SIZE)
        self.gguf_writer.add_context_length(4096)  # Default context
        self.gguf_writer.add_embedding_length(ORPHEUS_HIDDEN_SIZE)
        self.gguf_writer.add_block_count(ORPHEUS_N_LAYERS)
        self.gguf_writer.add_feed_forward_length(8192)  # Standard LLaMA FFN
        self.gguf_writer.add_head_count(ORPHEUS_N_ATTN_HEADS)
        self.gguf_writer.add_head_count_kv(ORPHEUS_N_KV_ATTN_HEADS)
        self.gguf_writer.add_rope_dimension_count(ORPHEUS_HEAD_DIM)
        self.gguf_writer.add_rope_freq_base(500000.0)  # Orpheus uses high rope theta
        self.gguf_writer.add_layer_norm_rms_eps(1e-5)

        # Orpheus-specific metadata
        self.gguf_writer.add_uint32("orpheus.stopping_token_id", ORPHEUS_STOPPING_TOKEN_ID)
        self.gguf_writer.add_uint32("orpheus.hidden_size", ORPHEUS_HIDDEN_SIZE)
        self.gguf_writer.add_uint32("orpheus.kv_hidden_size", ORPHEUS_HIDDEN_SIZE * ORPHEUS_N_KV_ATTN_HEADS // ORPHEUS_N_ATTN_HEADS)

        # Token IDs
        self.gguf_writer.add_bos_token_id(ORPHEUS_BOS_TOKEN_ID)
        self.gguf_writer.add_eos_token_id(ORPHEUS_EOS_TOKEN_ID)

    def _set_vocab(self):
        """Set up the tokenizer vocabulary."""
        # Orpheus uses LLaMA-3 style tokenizer (sentencepiece)
        tokenizer_model_path = self.dir_model / "tokenizer.model"
        tokenizer_json_path = self.dir_model / "tokenizer.json"

        if tokenizer_model_path.exists():
            # SentencePiece tokenizer (LLaMA style)
            self._set_vocab_sentencepiece()
        elif tokenizer_json_path.exists():
            # BPE tokenizer fallback
            self._set_vocab_bpe()
        else:
            logger.warning("No tokenizer found, skipping vocab")

    def _set_vocab_sentencepiece(self):
        """Set vocabulary from sentencepiece tokenizer."""
        # Use LlamaHfVocab for LLaMA-style sentencepiece tokenizer
        vocab = gguf.LlamaHfVocab(self.dir_model)
        tokens = []
        for i in range(len(vocab)):
            tokens.append(vocab.get_token(i))
        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_token_list(tokens)

        # Add special vocab with proper tokenizer model
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=False)
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_bpe(self):
        """Set vocabulary from BPE tokenizer."""
        # GPT-2 style BPE tokenizer
        self.gguf_writer.add_tokenizer_model("gpt2")

        # Load vocabulary from tokenizer.json (HuggingFace format)
        tokenizer_json_path = self.dir_model / "tokenizer.json"
        vocab_json_path = self.dir_model / "vocab.json"

        tokens = []
        if tokenizer_json_path.exists():
            with open(tokenizer_json_path, "r", encoding="utf-8") as f:
                tokenizer_data = json.load(f)
                if "model" in tokenizer_data and "vocab" in tokenizer_data["model"]:
                    # Sort by value (index) to get tokens in order
                    vocab = tokenizer_data["model"]["vocab"]
                    tokens = [k for k, v in sorted(vocab.items(), key=lambda x: x[1])]
                    logger.info(f"Loaded {len(tokens)} tokens from tokenizer.json")

        if not tokens and vocab_json_path.exists():
            with open(vocab_json_path, "r", encoding="utf-8") as f:
                vocab = json.load(f)
                tokens = [k for k, v in sorted(vocab.items(), key=lambda x: x[1])]
                logger.info(f"Loaded {len(tokens)} tokens from vocab.json")

        # Orpheus extends the vocabulary with audio tokens (128000-156939)
        # These are special tokens for SNAC audio codec codes
        while len(tokens) < ORPHEUS_VOCAB_SIZE:
            token_id = len(tokens)
            # Audio tokens are named like: <audio_token_XXXX>
            tokens.append(f"<audio_{token_id}>")
        logger.info(f"Extended vocabulary to {len(tokens)} tokens for Orpheus audio tokens")

        if tokens:
            self.gguf_writer.add_token_list(tokens)
        else:
            logger.warning("No vocabulary found for BPE tokenizer")

        # Add special vocab
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)

    def convert(self):
        """Perform the conversion."""
        logger.info(f"Converting Orpheus-TTS model from {self.dir_model}")

        # Set metadata
        self._set_gguf_parameters()
        logger.info("Set GGUF parameters")

        # Set vocabulary
        self._set_vocab()
        logger.info("Set vocabulary")

        # Load LLM tensors
        tensors = self._load_tensors()
        logger.info(f"Loaded {len(tensors)} LLM tensors")

        # Process tensors
        n_head = ORPHEUS_N_ATTN_HEADS
        n_head_kv = ORPHEUS_N_KV_ATTN_HEADS

        llm_count = 0
        skipped_count = 0

        for hf_name, tensor in tensors.items():
            # Try LLM mapping
            gguf_name = self._get_llm_tensor_name(hf_name)

            if gguf_name:
                # Apply QK permutation for attention weights
                if gguf_name.endswith("attn_q.weight"):
                    tensor = self._permute_qk(tensor, n_head, n_head)
                elif gguf_name.endswith("attn_k.weight"):
                    tensor = self._permute_qk(tensor, n_head, n_head_kv)

                # Convert to numpy
                data = tensor.numpy().squeeze()
                self.gguf_writer.add_tensor(gguf_name, data)
                llm_count += 1
                continue

            # Log unmapped tensors (only if they look like model weights)
            if any(x in hf_name for x in ['weight', 'bias', 'alpha', 'codebook', 'proj']):
                logger.debug(f"Unmapped tensor: {hf_name} {tensor.shape}")
                skipped_count += 1

        logger.info(f"Converted {llm_count} LLM tensors, skipped {skipped_count}")

        # Write GGUF file
        self.gguf_writer.write_header_to_file()
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file()
        self.gguf_writer.close()

        logger.info(f"GGUF file written to {self.fname_out}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Orpheus-TTS LLM model to GGUF format"
    )
    parser.add_argument(
        "model",
        type=Path,
        help="Path to the Orpheus-TTS model directory",
    )
    parser.add_argument(
        "--outfile",
        type=Path,
        default=None,
        help="Output GGUF file path (default: orpheus-3b-f16.gguf)",
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
        args.outfile = Path(f"orpheus-3b-{args.outtype}.gguf")

    # Run conversion
    converter = OrpheusConverter(
        dir_model=args.model,
        fname_out=args.outfile,
        ftype=ftype,
    )
    converter.convert()

    logger.info("Conversion complete!")


if __name__ == "__main__":
    main()
