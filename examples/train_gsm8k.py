"""GRPO training demo using KernelOptGRPOTrainer (K1 swapped in).

This is `scripts/smoke_grpo.py` but with our trainer subclass instead of
stock TRL. Same model, same dataset, same recipe. Drop-in replacement,
1-line diff in the trainer construction.

Run from PowerShell with:
    cmd /c "scripts\dev_env.bat && python examples\train_gsm8k.py"
"""

import os
import re
import time

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig

from kernel_opt import KernelOptGRPOTrainer


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# Recipe locked to fit 8 GB on RTX 4060 (Stage 1 found peak ~9.4 GB at G=8/256;
# this version tightens to comfortably fit including the transformer backward).
NUM_GENERATIONS = 4
PER_DEV_BATCH = NUM_GENERATIONS
MAX_COMPLETION_LEN = 192
NUM_STEPS = 5


def extract_answer(text: str):
    m = re.search(r"####\s*([-+]?\d[\d,]*\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")
    m = re.findall(r"-?\d+\.?\d*", text)
    return m[-1] if m else None


def reward_correctness(completions, answer, **kwargs):
    rewards = []
    for c, gold in zip(completions, answer):
        text = c if isinstance(c, str) else c[0]["content"]
        gold_num = extract_answer(gold)
        pred_num = extract_answer(text)
        rewards.append(1.0 if (pred_num is not None and pred_num == gold_num) else 0.0)
    return rewards


def reward_format(completions, **kwargs):
    pat = re.compile(r"<think>.+?</think>.*<answer>.+?</answer>", re.DOTALL)
    rewards = []
    for c in completions:
        text = c if isinstance(c, str) else c[0]["content"]
        rewards.append(0.2 if pat.search(text) else 0.0)
    return rewards


def make_dataset():
    ds = load_dataset("openai/gsm8k", "main", split="train").select(range(64))
    sys = ("Solve the math problem. Put reasoning in <think>...</think> "
           "and the final numeric answer in <answer>...</answer>.")
    def fmt(ex):
        return {
            "prompt": [
                {"role": "system", "content": sys},
                {"role": "user", "content": ex["question"]},
            ],
            "answer": ex["answer"],
        }
    return ds.map(fmt, remove_columns=ds.column_names)


def main():
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    print(f"loading {MODEL} ...")
    _tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)

    print("preparing dataset ...")
    ds = make_dataset()

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    cfg = GRPOConfig(
        output_dir="D:/Projects/kernel-opt/build/train_gsm8k_out",
        per_device_train_batch_size=PER_DEV_BATCH,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LEN,
        max_steps=NUM_STEPS,
        learning_rate=1e-5,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=True,
        gradient_checkpointing=True,   # ~2× activation memory reduction
        optim="adamw_8bit",
        beta=0.04,
        temperature=0.9,
    )

    trainer = KernelOptGRPOTrainer(    # <-- the only diff from smoke_grpo.py
        model=model,
        args=cfg,
        train_dataset=ds,
        reward_funcs=[reward_correctness, reward_format],
        peft_config=lora,
    )

    print("starting training with KernelOptGRPOTrainer ...")
    t_start = time.perf_counter()
    trainer.train()
    t_end = time.perf_counter()

    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    print()
    print("=== KernelOptGRPOTrainer demo result ===")
    print(f"steps:               {NUM_STEPS}")
    print(f"wall time:           {t_end - t_start:.1f}s "
          f"({(t_end - t_start) / NUM_STEPS:.1f}s/step)")
    print(f"peak allocated VRAM: {peak_mb:.0f} MB")
    print(f"total elapsed:       {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
