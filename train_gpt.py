#!/usr/bin/env python3
"""Train a tiny GPT-style character language model."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(channels, dim=2)
        q = q.view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:, :, :steps, :steps] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        y = weights @ v
        y = y.transpose(1, 2).contiguous().view(batch, steps, channels)
        return self.resid_dropout(self.proj(y))


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.token_embedding.weight = self.head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, steps = tokens.shape
        if steps > self.config.block_size:
            raise ValueError("sequence is longer than block_size")
        positions = torch.arange(steps, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        x = self.dropout(x)
        x = self.blocks(x)
        logits = self.head(self.ln(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.9,
        top_k: int | None = 20,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            tokens_cond = tokens[:, -self.config.block_size :]
            logits, _ = self(tokens_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens


class CharTokenizer:
    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self.stoi = {char: index for index, char in enumerate(self.chars)}
        self.itos = {index: char for char, index in self.stoi.items()}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[char] for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.itos[token] for token in tokens)

    def to_json(self) -> dict[str, list[str]]:
        return {"chars": self.chars}

    @classmethod
    def from_json(cls, data: dict[str, list[str]]) -> "CharTokenizer":
        tokenizer = cls.__new__(cls)
        tokenizer.chars = data["chars"]
        tokenizer.stoi = {char: index for index, char in enumerate(tokenizer.chars)}
        tokenizer.itos = {index: char for char, index in tokenizer.stoi.items()}
        return tokenizer


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[start : start + block_size] for start in starts])
    y = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: TinyGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    eval_iters: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    losses = {}
    for split, data in (("train", train_data), ("val", val_data)):
        values = torch.zeros(eval_iters)
        for index in range(eval_iters):
            x, y = get_batch(data, batch_size, model.config.block_size, device)
            _, loss = model(x, y)
            values[index] = loss.item()
        losses[split] = values.mean().item()
    model.train()
    return losses


def command_train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    text = Path(args.input).read_text(encoding="utf-8")
    if len(text) < args.block_size + 2:
        raise ValueError("training text must be longer than block_size")

    tokenizer = CharTokenizer(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_index = max(int(0.9 * len(encoded)), args.block_size + 2)
    train_data = encoded[:split_index]
    val_data = encoded[split_index:]
    if len(val_data) < args.block_size + 2:
        val_data = train_data

    config = GPTConfig(
        vocab_size=len(tokenizer.chars),
        block_size=args.block_size,
        n_layer=args.layers,
        n_head=args.heads,
        n_embd=args.embedding,
        dropout=args.dropout,
    )
    device = args.device or pick_device()
    model = TinyGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    started = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_interval == 0:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                args.batch_size,
                args.eval_iters,
                device,
            )
            print(
                f"step {step:5d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | {time.time() - started:.1f}s"
            )

        x, y = get_batch(train_data, args.batch_size, config.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    checkpoint = {
        "config": asdict(config),
        "tokenizer": tokenizer.to_json(),
        "model": model.state_dict(),
    }
    torch.save(checkpoint, output / "checkpoint.pt")
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    print(f"saved checkpoint to {output / 'checkpoint.pt'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a tiny GPT-style language model.")
    parser.add_argument("--input", default="data/tiny_corpus.txt", help="training text file")
    parser.add_argument("--output", default="runs/tiny-gpt", help="checkpoint directory")
    parser.add_argument("--steps", type=int, default=400, help="training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--block-size", type=int, default=128, help="context length")
    parser.add_argument("--layers", type=int, default=4, help="Transformer blocks")
    parser.add_argument("--heads", type=int, default=4, help="attention heads")
    parser.add_argument("--embedding", type=int, default=128, help="embedding width")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout probability")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate")
    parser.add_argument("--eval-interval", type=int, default=100, help="steps between loss reports")
    parser.add_argument("--eval-iters", type=int, default=10, help="batches per loss estimate")
    parser.add_argument("--device", default=None, help="cpu, mps, or cuda")
    parser.add_argument("--seed", type=int, default=1337, help="random seed")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    command_train(args)


if __name__ == "__main__":
    main()

