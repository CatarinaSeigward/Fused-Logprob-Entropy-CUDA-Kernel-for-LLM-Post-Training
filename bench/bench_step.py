"""End-to-end step-level bench: full GRPO step latency + peak VRAM.

Complements bench_micro.py (kernel-level isolation) and bench_backward.py
(backward-only). This file answers: "in a real training step, how much
does K1 actually save?"

Runs N steps with stock TRL and our trainer on the SAME recipe; reports
step latency, peak VRAM, and isolates the logp/entropy segment.

Output: bench/results/bench_step_<timestamp>.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from kernel_opt import KernelOptGRPOTrainer


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 1234


def reward_dummy(completions, **kwargs):
    return [0.5 for _ in completions]


def make_dataset(n=16):
    ds = load_dataset("openai/gsm8k", "main", split="train").select(range(n))
    def fmt(ex):
        return {"prompt": [{"role": "user", "content": ex["question"]}],
                "answer": ex["answer"]}
    return ds.map(fmt, remove_columns=ds.column_names)


def build(trainer_cls, num_generations, max_completion, num_steps):
    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    cfg = GRPOConfig(
        output_dir=f"D:/Projects/kernel-opt/build/bench_step_{trainer_cls.__name__}",
        per_device_train_batch_size=num_generations,
        num_generations=num_generations,
        max_completion_length=max_completion,
        max_steps=num_steps,
        learning_rate=1e-5,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        beta=0.04,
        temperature=0.9,
        seed=SEED,
        data_seed=SEED,
    )
    return trainer_cls(
        model=model, args=cfg,
        train_dataset=make_dataset(),
        reward_funcs=[reward_dummy],
        peft_config=lora,
    )


def run_one_recipe(trainer_cls, num_generations, max_completion, num_steps=2):
    """Returns dict with per-step latencies, peak VRAM, step_time_metric_avg."""
    print(f"\n--- {trainer_cls.__name__}  G={num_generations}  max_completion={max_completion}  steps={num_steps} ---")
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    trainer = build(trainer_cls, num_generations, max_completion, num_steps)
    t_total_start = time.perf_counter()
    trainer.train()
    t_total = time.perf_counter() - t_total_start
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)

    # Use total / num_steps as the reliable average. (TRL's `_metrics` is
    # cleared in some code paths; total wall is robust.)
    avg_step_s = t_total / num_steps if num_steps > 0 else None

    result = {
        "trainer": trainer_cls.__name__,
        "num_generations": num_generations,
        "max_completion": max_completion,
        "num_steps": num_steps,
        "total_wall_s": round(t_total, 3),
        "avg_step_s": round(avg_step_s, 3) if avg_step_s else None,
        "peak_alloc_mb": round(peak_mb, 1),
        "peak_reserved_mb": round(reserved_mb, 1),
    }
    print(json.dumps(result, indent=2))

    # Free everything before next run
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--max-completion", type=int, default=192)
    ap.add_argument("--num-steps", type=int, default=2)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    out_path = args.out or os.path.join(
        os.path.dirname(__file__), "results",
        f"bench_step_{int(time.time())}.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    results = []
    # Run STOCK first (might OOM on tighter recipes; we want the data anyway)
    try:
        results.append(run_one_recipe(GRPOTrainer, args.num_generations,
                                       args.max_completion, args.num_steps))
    except torch.cuda.OutOfMemoryError as e:
        print(f"  STOCK OOM: {e}")
        results.append({"trainer": "GRPOTrainer", "error": "OOM"})

    try:
        results.append(run_one_recipe(KernelOptGRPOTrainer, args.num_generations,
                                       args.max_completion, args.num_steps))
    except torch.cuda.OutOfMemoryError as e:
        print(f"  OURS OOM: {e}")
        results.append({"trainer": "KernelOptGRPOTrainer", "error": "OOM"})

    # Compare
    print("\n" + "=" * 70)
    print("Comparison")
    print("=" * 70)
    if all("error" not in r for r in results):
        s, o = results
        print(f"{'metric':>22}  {'stock':>14}  {'ours':>14}  {'ratio/diff':>14}")
        print(f"{'avg step time (s)':>22}  {s['avg_step_s']:>14.2f}  "
              f"{o['avg_step_s']:>14.2f}  {s['avg_step_s']/o['avg_step_s']:>13.2f}x")
        print(f"{'peak alloc (MB)':>22}  {s['peak_alloc_mb']:>14.0f}  "
              f"{o['peak_alloc_mb']:>14.0f}  "
              f"{s['peak_alloc_mb']-o['peak_alloc_mb']:>13.0f}MB")
        print(f"{'peak reserved (MB)':>22}  {s['peak_reserved_mb']:>14.0f}  "
              f"{o['peak_reserved_mb']:>14.0f}  "
              f"{s['peak_reserved_mb']-o['peak_reserved_mb']:>13.0f}MB")

    with open(out_path, "w") as f:
        json.dump({
            "config": {"num_generations": args.num_generations,
                       "max_completion": args.max_completion,
                       "num_steps": args.num_steps},
            "results": results,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
