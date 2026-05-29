#!/usr/bin/env python3
"""Generate text with a tiny GPT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train_gpt import CharTokenizer, GPTConfig, TinyGPT, pick_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from a tiny GPT-style model.")
    parser.add_argument("--checkpoint", default="runs/tiny-gpt/checkpoint.pt", help="checkpoint path")
    parser.add_argument("--prompt", default="The model", help="text prompt")
    parser.add_argument("--tokens", type=int, default=300, help="characters to generate")
    parser.add_argument("--temperature", type=float, default=0.9, help="sampling temperature")
    parser.add_argument("--top-k", type=int, default=20, help="sample from the k most likely chars")
    parser.add_argument("--device", default=None, help="cpu, mps, or cuda")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = args.device or pick_device()
    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    tokenizer = CharTokenizer.from_json(checkpoint["tokenizer"])
    config = GPTConfig(**checkpoint["config"])
    model = TinyGPT(config).to(device)
    model.load_state_dict(checkpoint["model"])

    unknown = sorted(set(args.prompt) - set(tokenizer.chars))
    if unknown:
        raise ValueError(f"prompt contains characters missing from training data: {unknown}")

    start = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    output = model.generate(
        start,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
