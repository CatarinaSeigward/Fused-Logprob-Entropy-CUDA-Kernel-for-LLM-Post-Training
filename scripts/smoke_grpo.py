"""Stage 1 exit-criterion: 5-step stock TRL GRPO on Qwen2.5-0.5B + GSM8K.

Goal: confirm the planned recipe fits in 8 GB VRAM. If OOM, this script is
the harness we'll trim against (drop LoRA rank, num_generations, max_seq, etc.)
to lock the recipe.

NOT a deliverable. Becomes examples/train_gsm8k.py later, with our custom
trainer swapped in.
"""

import os
import re
import time

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
NUM_GENERATIONS = 8       # G (responses per prompt)
# In TRL 1.4, per_device_train_batch_size counts SEQUENCES (gens), not prompts.
# Set it to G so one prompt is fully processed per gradient step.
PER_DEV_BATCH = NUM_GENERATIONS
MAX_COMPLETION_LEN = 256  # TRL 1.4 dropped max_prompt_length; only completion is bounded
NUM_STEPS = 5


def extract_answer(text: str) -> str | None:
    """Pull the final number from text. Tolerates GSM8K '#### NN' or plain digits."""
    m = re.search(r"####\s*([-+]?\d[\d,]*\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")
    m = re.findall(r"-?\d+\.?\d*", text)
    return m[-1] if m else None


def reward_correctness(completions, answer, **kwargs):
    """+1 if extracted answer matches gold, else 0. Robust to list/string completions."""
    rewards = []
    for c, gold in zip(completions, answer):
        text = c if isinstance(c, str) else c[0]["content"]
        gold_num = extract_answer(gold)
        pred_num = extract_answer(text)
        rewards.append(1.0 if (pred_num is not None and pred_num == gold_num) else 0.0)
    return rewards


def reward_format(completions, **kwargs):
    """Small bonus for using <think>...</think><answer>...</answer> structure."""
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
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    print(f"  model loaded, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    print("preparing dataset ...")
    ds = make_dataset()

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    cfg = GRPOConfig(
        output_dir="D:/Projects/kernel-opt/build/smoke_grpo_out",
        per_device_train_batch_size=PER_DEV_BATCH,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LEN,
        max_steps=NUM_STEPS,
        learning_rate=1e-5,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=True,
        gradient_checkpointing=False,    # would slow down; keep off for VRAM observation
        optim="adamw_8bit",              # bnb 8-bit Adam to save VRAM
        beta=0.04,                       # KL coefficient (DeepSeek default)
        temperature=0.9,
    )

    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        reward_funcs=[reward_correctness, reward_format],
        peft_config=lora,
    )

    print("starting 5-step training ...")
    t_start = time.perf_counter()
    trainer.train()
    t_end = time.perf_counter()

    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)
    print()
    print(f"=== Stage 1 smoke result ===")
    print(f"steps:                {NUM_STEPS}")
    print(f"wall time:            {t_end - t_start:.1f}s ({(t_end - t_start)/NUM_STEPS:.1f}s/step)")
    print(f"peak allocated VRAM:  {peak_mb:.0f} MB")
    print(f"peak reserved VRAM:   {reserved_mb:.0f} MB  <- this is what nvidia-smi would show")
    print(f"total elapsed (incl. load): {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
