#!/usr/bin/env python3
"""Interactive CLI demo – type sentences and watch SRAM/HBM tier changes.

Usage::

    python demo.py                       # GPT-2 on available device
    python demo.py --model gpt2-medium
    python demo.py --sram 64
"""

from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_cache_manager import CacheManager, ImportanceScorer, ManagedGenerator


def _print_tier(label: str, entries, tokenizer, max_show: int = 20):
    """Print a compact view of a cache tier."""
    if not entries:
        print(f"  {label}: (empty)")
        return
    # Sort by importance descending
    sorted_entries = sorted(entries, key=lambda e: e.importance, reverse=True)
    shown = sorted_entries[:max_show]
    tokens = [tokenizer.decode([e.token_id]) for e in shown]
    print(f"  {label} ({len(entries)} tokens):")
    for e, tok in zip(shown, tokens):
        bar = "█" * int(e.importance * 20)
        print(f"    pos={e.position:>4d}  imp={e.importance:.3f} {bar:20s}  refs={e.refs}  '{tok}'")
    if len(entries) > max_show:
        print(f"    … and {len(entries) - max_show} more")


def main():
    parser = argparse.ArgumentParser(description="KV Cache Manager – Interactive Demo")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--sram", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device} …")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    dim = model.config.n_embd
    scorer = ImportanceScorer()
    cache = CacheManager(sram_capacity=args.sram, dim=dim, device=device, scorer=scorer)
    gen = ManagedGenerator(model, tokenizer, cache)

    print(f"\nReady.  SRAM capacity = {args.sram} tokens")
    print("Commands:  :stats  :tiers  :ppl <text>  :reset  :quit\n")

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input == ":quit":
            break

        if user_input == ":reset":
            cache.reset()
            print("Cache cleared.")
            continue

        if user_input == ":stats":
            for k, v in cache.stats().items():
                print(f"  {k}: {v}")
            continue

        if user_input == ":tiers":
            _print_tier("SRAM", cache.sram, tokenizer)
            _print_tier("HBM ", cache.hbm, tokenizer)
            continue

        if user_input.startswith(":ppl "):
            text = user_input[5:]
            print("Computing perplexity (this may take a moment) …")
            result = gen.evaluate_perplexity(text)
            print(f"  Baseline PPL : {result['baseline_ppl']:.2f}")
            print(f"  Managed PPL  : {result['managed_ppl']:.2f}")
            print(f"  Ratio        : {result['ppl_ratio']:.4f}")
            for k, v in result["cache_stats"].items():
                print(f"  {k}: {v}")
            continue

        # Default: generate continuation
        cache.reset()
        print("Generating …")
        output = gen.generate(
            user_input,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"\n{user_input}{output}\n")

        # Show tier summary
        stats = cache.stats()
        print(
            f"[SRAM {stats['sram_capacity_used']}  "
            f"HBM {stats['hbm_count']}  "
            f"hit_rate {stats['sram_hit_rate']}]"
        )


if __name__ == "__main__":
    main()
