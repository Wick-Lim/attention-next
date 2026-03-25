"""Benchmark: managed KV cache vs full attention on GPT-2.

Usage::

    python -m kv_cache_manager.benchmark                  # defaults
    python -m kv_cache_manager.benchmark --sram 64 --len 256
    python -m kv_cache_manager.benchmark --dataset wikitext
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .cache_manager import CacheManager
from .importance_scorer import ImportanceScorer
from .attention_hook import ManagedGenerator


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load_sample_texts(source: str, num_samples: int, max_tokens: int, tokenizer) -> list[str]:
    """Return a list of text samples for evaluation."""
    if source == "wikitext":
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        texts = []
        for row in ds:
            text = row["text"].strip()
            if not text:
                continue
            ids = tokenizer.encode(text)
            if len(ids) >= max_tokens:
                texts.append(text)
            if len(texts) >= num_samples:
                break
        return texts

    # Fallback: built-in samples for quick testing without datasets.
    return [
        (
            "The history of artificial intelligence began in antiquity, with myths, "
            "stories and rumors of artificial beings endowed with intelligence or "
            "consciousness by master craftsmen. The seeds of modern AI were planted by "
            "philosophers who attempted to describe the process of human thinking as "
            "the mechanical manipulation of symbols. This work culminated in the "
            "invention of the programmable digital computer in the 1940s, a machine "
            "based on the abstract essence of mathematical reasoning. This device and "
            "the ideas behind it inspired a handful of scientists to begin seriously "
            "discussing the possibility of building an electronic brain."
        ),
        (
            "In computer science, a hash table is a data structure that implements an "
            "associative array, also called a dictionary. A hash table uses a hash "
            "function to compute an index, also called a hash code, into an array of "
            "buckets or slots, from which the desired value can be found. During "
            "lookup, the key is hashed and the resulting hash indicates where the "
            "corresponding value is stored. Ideally, the hash function will assign "
            "each key to a unique bucket, but most hash table designs employ an "
            "imperfect hash function, which might cause hash collisions."
        ),
    ]


def _measure_memory(device: str) -> float:
    """Return current GPU memory allocated in MiB, or 0 for CPU."""
    if device == "cpu":
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024 ** 2)


# ------------------------------------------------------------------
# Core benchmark
# ------------------------------------------------------------------

def run_benchmark(args: argparse.Namespace) -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model & tokenizer
    print(f"Loading {args.model} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    num_layers = model.config.n_layer
    num_heads = model.config.n_head
    head_dim = model.config.n_embd // num_heads
    dim = model.config.n_embd
    print(f"Model: layers={num_layers}  heads={num_heads}  head_dim={head_dim}  dim={dim}")

    # Samples
    texts = _load_sample_texts(args.dataset, args.samples, args.len, tokenizer)
    print(f"Evaluating on {len(texts)} sample(s), max_length={args.len}")

    # Scorer & manager
    scorer = ImportanceScorer(
        sim_threshold=args.sim_thresh,
        boost=args.boost,
        decay=args.decay,
        promote_thresh=args.promote_thresh,
    )
    cache = CacheManager(sram_capacity=args.sram, dim=dim, device=device, scorer=scorer)
    gen = ManagedGenerator(model, tokenizer, cache)

    results = []
    for idx, text in enumerate(tqdm(texts, desc="Samples")):
        mem_before = _measure_memory(device)
        t0 = time.perf_counter()
        r = gen.evaluate_perplexity(text, max_length=args.len)
        t1 = time.perf_counter()
        mem_after = _measure_memory(device)

        r["wall_time_s"] = round(t1 - t0, 3)
        r["memory_delta_mib"] = round(mem_after - mem_before, 2)
        results.append(r)

        print(
            f"  [{idx+1}/{len(texts)}]  "
            f"baseline_ppl={r['baseline_ppl']:.2f}  "
            f"managed_ppl={r['managed_ppl']:.2f}  "
            f"ratio={r['ppl_ratio']:.4f}  "
            f"time={r['wall_time_s']}s"
        )

    # Aggregate
    avg_ratio = sum(r["ppl_ratio"] for r in results) / len(results)
    avg_baseline = sum(r["baseline_ppl"] for r in results) / len(results)
    avg_managed = sum(r["managed_ppl"] for r in results) / len(results)

    summary = {
        "model": args.model,
        "sram_capacity": args.sram,
        "max_length": args.len,
        "num_samples": len(texts),
        "avg_baseline_ppl": round(avg_baseline, 2),
        "avg_managed_ppl": round(avg_managed, 2),
        "avg_ppl_ratio": round(avg_ratio, 4),
        "target_ratio": "≤ 1.05",
        "pass": avg_ratio <= 1.05,
        "per_sample": results,
    }

    print("\n" + "=" * 60)
    print(f"Average baseline PPL : {avg_baseline:.2f}")
    print(f"Average managed PPL  : {avg_managed:.2f}")
    print(f"Average ratio        : {avg_ratio:.4f}  (target ≤ 1.05)")
    print(f"Result               : {'PASS' if summary['pass'] else 'FAIL'}")
    print("=" * 60)

    return summary


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="KV Cache Manager Benchmark")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name")
    parser.add_argument("--sram", type=int, default=128, help="SRAM capacity (tokens)")
    parser.add_argument("--len", type=int, default=256, help="Max sequence length")
    parser.add_argument("--samples", type=int, default=2, help="Number of text samples")
    parser.add_argument("--dataset", default="builtin", choices=["builtin", "wikitext"])
    parser.add_argument("--sim-thresh", type=float, default=0.30)
    parser.add_argument("--boost", type=float, default=0.35)
    parser.add_argument("--decay", type=float, default=0.04)
    parser.add_argument("--promote-thresh", type=float, default=0.68)
    parser.add_argument("--output", type=str, default=None, help="Save JSON results to file")
    args = parser.parse_args()

    summary = run_benchmark(args)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
