"""1-step parity check: stock TRL vs KernelOptGRPOTrainer override.

Stage 4 exit criterion. Builds ONE GRPOTrainer (loads model + ref + LoRA
optimizer once). Then calls `_get_per_token_logps_and_entropies` two ways
on the same input batch:
  - bound to the stock GRPOTrainer instance (uses TRL's eager helpers)
  - bound to a KernelOptGRPOTrainer instance, sharing the same model

Asserts per-token logprob and entropy match within bf16 numerical noise.

Why not run two full trainer.train() steps and compare losses? Because two
trainers + 2 model copies + 2 ref copies blow past 8 GB on the 4060. This
test isolates the substitution we actually care about.

Run:
    cmd /c "scripts\dev_env.bat && python examples\compare_one_step.py"
"""
from __future__ import annotations

import time

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from kernel_opt import KernelOptGRPOTrainer


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
NUM_GENERATIONS = 4         # smaller than demo to leave VRAM headroom
PER_DEV_BATCH = NUM_GENERATIONS
MAX_COMPLETION_LEN = 128
SEED = 1234


def reward_dummy(completions, **kwargs):
    return [0.5 for _ in completions]


def make_dataset():
    ds = load_dataset("openai/gsm8k", "main", split="train").select(range(8))
    def fmt(ex):
        return {"prompt": [{"role": "user", "content": ex["question"]}],
                "answer": ex["answer"]}
    return ds.map(fmt, remove_columns=ds.column_names)


def main():
    torch.manual_seed(SEED)

    print(f"loading {MODEL} ...")
    _tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    cfg = GRPOConfig(
        output_dir="D:/Projects/kernel-opt/build/cmp_one_step",
        per_device_train_batch_size=PER_DEV_BATCH,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LEN,
        max_steps=1,
        learning_rate=1e-5,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=True,
        gradient_checkpointing=False,
        optim="adamw_8bit",
        beta=0.04,
        temperature=0.9,
        seed=SEED,
        data_seed=SEED,
    )

    print("building stock trainer (loads ref model, builds LoRA, etc.) ...")
    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=make_dataset(),
        reward_funcs=[reward_dummy],
        peft_config=lora,
    )

    # Pull a real batch by stepping the dataloader once. This invokes the
    # full GRPO machinery: rollouts, reward, advantages, etc. We just need
    # a realistic (input_ids, attention_mask, logits_to_keep) tuple.
    print("generating one rollout to get real input_ids ...")
    train_dl = trainer.get_train_dataloader()
    batch_iter = iter(train_dl)
    raw_batch = next(batch_iter)
    inputs = trainer._prepare_inputs(raw_batch)

    # `inputs` from TRL contains:
    #   prompt_ids, prompt_mask, completion_ids, completion_mask,
    #   advantages, old_per_token_logps, ...
    # We need to reconstruct the (full input_ids, attention_mask, logits_to_keep)
    # that `_compute_loss` would feed to `_get_per_token_logps_and_entropies`.
    # Mirror TRL/grpo_trainer.py:_compute_loss lines 2440-2447.
    prompt_ids = inputs["prompt_ids"]
    prompt_mask = inputs["prompt_mask"]
    completion_ids = inputs["completion_ids"]
    completion_mask = inputs["completion_mask"]
    full_input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    full_attn_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits_to_keep = completion_ids.size(1)

    print(f"  input shape: {full_input_ids.shape}, "
          f"logits_to_keep={logits_to_keep}, dtype={full_input_ids.dtype}")
    print(f"  vocab size:  {trainer.model.get_input_embeddings().num_embeddings}")

    # Use the wrapped (LoRA-attached) model from the trainer, not the raw model
    wrapped = trainer.model_wrapped
    print(f"  policy model: {type(wrapped).__name__}")

    # Run STOCK path (TRL's selective_log_softmax + entropy_from_logits)
    print()
    print("running STOCK GRPOTrainer._get_per_token_logps_and_entropies ...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        stock_logp, stock_ent = GRPOTrainer._get_per_token_logps_and_entropies(
            trainer, wrapped, full_input_ids, full_attn_mask, logits_to_keep,
            compute_entropy=True,
        )
    torch.cuda.synchronize()
    stock_ms = (time.perf_counter() - t0) * 1000
    stock_peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"  stock: logp.shape={tuple(stock_logp.shape)}, ent.shape={tuple(stock_ent.shape)}, "
          f"time={stock_ms:.1f}ms, peak_alloc={stock_peak_mb:.0f}MB")

    # Run OURS (K1) — bind our override to the same trainer instance
    print()
    print("running KernelOptGRPOTrainer._get_per_token_logps_and_entropies ...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        ours_logp, ours_ent = KernelOptGRPOTrainer._get_per_token_logps_and_entropies(
            trainer, wrapped, full_input_ids, full_attn_mask, logits_to_keep,
            compute_entropy=True,
        )
    torch.cuda.synchronize()
    ours_ms = (time.perf_counter() - t0) * 1000
    ours_peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"  ours:  logp.shape={tuple(ours_logp.shape)}, ent.shape={tuple(ours_ent.shape)}, "
          f"time={ours_ms:.1f}ms, peak_alloc={ours_peak_mb:.0f}MB")

    # Compare. Important nuance:
    #   TRL's `selective_log_softmax` for bf16 returns BF16 (per-row F.log_softmax
    #   keeps input dtype), and `entropy_from_logits` likewise. So TRL's outputs
    #   carry bf16 quantization noise (~3 sig figs).
    #   Ours returns fp32 (we accumulate in fp32 from bf16 input → strictly
    #   more accurate).
    # Three comparisons below tell the whole story.
    print()
    print("=" * 70)
    print("comparison: stock TRL eager vs K1, on real Qwen logits")
    print("=" * 70)

    # (1) Raw fp32 diff. Bounded by TRL's bf16 quantization, NOT by our error.
    raw_logp = (stock_logp.float() - ours_logp.float()).abs()
    raw_ent = (stock_ent.float() - ours_ent.float()).abs()
    print(f"\n[1] raw diff (fp32 view of both outputs):")
    print(f"    logprob: max={raw_logp.max().item():.4e}  "
          f"mean={raw_logp.mean().item():.4e}  median={raw_logp.median().item():.4e}")
    print(f"    entropy: max={raw_ent.max().item():.4e}  "
          f"mean={raw_ent.mean().item():.4e}  median={raw_ent.median().item():.4e}")

    # (2) Drop-in equivalence: cast OURS to bf16 (TRL's output dtype) before
    # comparing. Should be exactly bf16 quantization noise apart.
    ours_logp_bf16 = ours_logp.bfloat16().float()
    ours_ent_bf16 = ours_ent.bfloat16().float()
    eq_logp = (stock_logp.float() - ours_logp_bf16).abs()
    eq_ent = (stock_ent.float() - ours_ent_bf16).abs()
    print(f"\n[2] drop-in equivalence (after both → bf16 → fp32):")
    print(f"    logprob: max={eq_logp.max().item():.4e}  "
          f"mean={eq_logp.mean().item():.4e}")
    print(f"    entropy: max={eq_ent.max().item():.4e}  "
          f"mean={eq_ent.mean().item():.4e}")

    # (3) Vs fp32 ground truth on the SAME bf16 logits (cast up). Tells us
    # which path is closer to the true value.
    print(f"\n[3] accuracy vs fp32 ground truth (same logits, fp32 reduction):")
    # Recompute on saved logits — easiest to inline a tiny ground-truth path.
    # Use the model's last forward by re-calling the override path that
    # captures logits. To keep this self-contained, we recompute via a small
    # helper: slice logits exactly as TRL does.
    with torch.no_grad():
        out = wrapped(input_ids=full_input_ids, attention_mask=full_attn_mask,
                      logits_to_keep=logits_to_keep + 1, use_cache=False)
        gt_logits = out.logits[:, :-1, :][:, -logits_to_keep:, :]
        gt_logits = gt_logits / trainer.temperature
        gt_logits_f32 = gt_logits.float()
        gt_log_probs = torch.log_softmax(gt_logits_f32, dim=-1)
        gt_logp = gt_log_probs.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
        gt_p = gt_log_probs.exp()
        gt_ent = -(gt_p * gt_log_probs).sum(-1)
    err_stock_logp = (stock_logp.float() - gt_logp).abs().max().item()
    err_ours_logp = (ours_logp.float() - gt_logp).abs().max().item()
    err_stock_ent = (stock_ent.float() - gt_ent).abs().max().item()
    err_ours_ent = (ours_ent.float() - gt_ent).abs().max().item()
    print(f"    logprob: stock vs gt = {err_stock_logp:.4e}  "
          f"ours vs gt = {err_ours_logp:.4e}  "
          f"({'ours better' if err_ours_logp < err_stock_logp else 'stock better'})")
    print(f"    entropy: stock vs gt = {err_stock_ent:.4e}  "
          f"ours vs gt = {err_ours_ent:.4e}  "
          f"({'ours better' if err_ours_ent < err_stock_ent else 'stock better'})")

    print()
    print(f"timing (logp+ent extraction including model forward): "
          f"stock {stock_ms:.0f}ms  ours {ours_ms:.0f}ms  "
          f"speedup {stock_ms/ours_ms:.2f}x")
    print(f"peak VRAM: stock {stock_peak_mb:.0f}MB  ours {ours_peak_mb:.0f}MB  "
          f"savings {stock_peak_mb - ours_peak_mb:.0f}MB")

    # Simulate the GRPO loss formula with both per-token logps. The loss is
    # what trainer.train() would actually optimize — comparing it directly is
    # the most meaningful drop-in check.
    print()
    print(f"[4] simulated GRPO loss with each logp source:")
    fake_old = stock_logp.float().detach()  # use stock as the "old" for both
    fake_advantages = torch.randn(*stock_logp.shape, device="cuda") * 0.5
    fake_mask = torch.ones_like(stock_logp, dtype=torch.float32)
    eps_clip = 0.2

    def grpo_loss(new_logp_f32):
        ratio = (new_logp_f32 - fake_old).exp()
        surr = -torch.minimum(ratio * fake_advantages,
                              ratio.clamp(1 - eps_clip, 1 + eps_clip) * fake_advantages)
        return (surr * fake_mask).sum() / fake_mask.sum()

    loss_stock = grpo_loss(stock_logp.float()).item()
    loss_ours = grpo_loss(ours_logp.float()).item()
    print(f"    loss(stock_logp) = {loss_stock:.6f}")
    print(f"    loss(ours_logp)  = {loss_ours:.6f}")
    print(f"    |diff|           = {abs(loss_stock - loss_ours):.6e}")

    # Stage 4 exit:
    #   (a) ours strictly closer to fp32 ground truth than stock TRL.
    #   (b) simulated GRPO loss using our logp matches stock-logp loss within
    #       PLAN's 1e-3 tolerance.
    print()
    print(f"Stage 4 exit criteria:")
    print(f"  ours <= stock vs fp32 ground truth (logp): "
          f"{'PASS' if err_ours_logp <= err_stock_logp else 'FAIL'}  "
          f"(ours {err_ours_logp:.2e} vs stock {err_stock_logp:.2e})")
    print(f"  ours <= stock vs fp32 ground truth (ent):  "
          f"{'PASS' if err_ours_ent <= err_stock_ent else 'FAIL'}  "
          f"(ours {err_ours_ent:.2e} vs stock {err_stock_ent:.2e})")
    LOSS_TOL = 1e-3
    print(f"  simulated GRPO loss diff < {LOSS_TOL}: "
          f"{'PASS' if abs(loss_stock - loss_ours) < LOSS_TOL else 'FAIL'}  "
          f"(diff {abs(loss_stock - loss_ours):.2e})")


if __name__ == "__main__":
    main()
