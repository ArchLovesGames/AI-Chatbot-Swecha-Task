#!/usr/bin/env python3
"""A tiny dependency-free character language model."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TinyLLM:
    order: int
    counts: dict[str, Counter[str]]
    vocabulary: list[str]

    @classmethod
    def train(cls, text: str, order: int) -> "TinyLLM":
        if order < 1:
            raise ValueError("order must be at least 1")
        if not text:
            raise ValueError("training text is empty")

        counts: dict[str, Counter[str]] = defaultdict(Counter)
        vocabulary = sorted(set(text))

        padded = "\n" * order + text
        for index in range(order, len(padded)):
            next_char = padded[index]
            for context_size in range(0, order + 1):
                context = padded[index - context_size : index]
                counts[context][next_char] += 1

        return cls(order=order, counts=dict(counts), vocabulary=vocabulary)

    def save(self, path: Path) -> None:
        data = {
            "order": self.order,
            "vocabulary": self.vocabulary,
            "counts": {
                context: dict(counter)
                for context, counter in sorted(self.counts.items())
            },
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TinyLLM":
        data = json.loads(path.read_text(encoding="utf-8"))
        counts = {
            context: Counter(counter)
            for context, counter in data["counts"].items()
        }
        return cls(
            order=int(data["order"]),
            counts=counts,
            vocabulary=list(data["vocabulary"]),
        )

    def generate(
        self,
        prompt: str,
        tokens: int,
        temperature: float,
        top_k: int | None,
        seed: int | None,
    ) -> str:
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0")
        if seed is not None:
            random.seed(seed)

        output = list(prompt)
        for _ in range(tokens):
            context = "".join(output[-self.order :])
            next_char = self._sample_next(context, temperature, top_k)
            output.append(next_char)
        return "".join(output)

    def _sample_next(
        self,
        context: str,
        temperature: float,
        top_k: int | None,
    ) -> str:
        counter = self._counter_for_context(context)
        items = counter.most_common()
        if top_k is not None and top_k > 0:
            items = items[:top_k]

        chars = [char for char, _ in items]
        weights = [count for _, count in items]
        adjusted = apply_temperature(weights, temperature)
        return random.choices(chars, weights=adjusted, k=1)[0]

    def _counter_for_context(self, context: str) -> Counter[str]:
        for size in range(min(len(context), self.order), -1, -1):
            candidate = context[-size:] if size else ""
            counter = self.counts.get(candidate)
            if counter:
                return counter
        return Counter({char: 1 for char in self.vocabulary})


def apply_temperature(weights: list[int], temperature: float) -> list[float]:
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    logits = [math.log(probability) / temperature for probability in probabilities]
    max_logit = max(logits)
    return [math.exp(logit - max_logit) for logit in logits]


def command_train(args: argparse.Namespace) -> None:
    text = Path(args.input).read_text(encoding="utf-8")
    model = TinyLLM.train(text=text, order=args.order)
    model.save(Path(args.model))
    print(f"trained {len(model.counts)} contexts from {len(text)} characters")
    print(f"saved model to {args.model}")


def command_generate(args: argparse.Namespace) -> None:
    model = TinyLLM.load(Path(args.model))
    result = model.generate(
        prompt=args.prompt,
        tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and sample a tiny local language model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train a model from a text file")
    train.add_argument("--input", required=True, help="training text file")
    train.add_argument("--model", default="model.json", help="output model path")
    train.add_argument("--order", type=int, default=5, help="maximum context length")
    train.set_defaults(func=command_train)

    generate = subparsers.add_parser("generate", help="generate text from a trained model")
    generate.add_argument("--model", default="model.json", help="model JSON path")
    generate.add_argument("--prompt", default="", help="starting text")
    generate.add_argument("--tokens", type=int, default=300, help="characters to generate")
    generate.add_argument("--temperature", type=float, default=0.9, help="sampling randomness")
    generate.add_argument("--top-k", type=int, default=12, help="sample from the k most likely tokens")
    generate.add_argument("--seed", type=int, default=None, help="random seed for repeatable output")
    generate.set_defaults(func=command_generate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
