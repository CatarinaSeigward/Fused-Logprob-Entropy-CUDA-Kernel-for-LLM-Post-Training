"""KernelOptGRPOTrainer — drop-in replacement for trl.GRPOTrainer using K1.

Overrides exactly one method: `_get_per_token_logps_and_entropies` (defined at
trl/trainer/grpo_trainer.py:1046–1125 in trl 1.4). That method is the single
entry point for ALL logprob computations in GRPO:
  - policy logprob during loss   (line 2447 in TRL, with autograd)
  - old logprob during rollout    (line 2051, no grad)
  - reference logprob             (lines 2097 / 2112, no grad)

Replacing it once swaps every logprob call to our fused kernel.

Multimodal training paths (image/pixel inputs) fall back to the parent's
eager implementation — we don't reimplement that complexity here.
"""
from __future__ import annotations

import torch
from trl import GRPOTrainer

from .ops import fused_logprob_entropy


_MULTIMODAL_KWARGS = (
    "pixel_values", "image_grid_thw", "num_images",
    "pixel_attention_mask", "image_sizes",
    "token_type_ids", "mm_token_type_ids", "image_position_ids",
)


class KernelOptGRPOTrainer(GRPOTrainer):
    """trl.GRPOTrainer with the inner logprob+entropy block replaced by K1.

    Behavior is bit-equivalent to stock TRL up to bf16 numerical noise (and in
    fact slightly *more* accurate in bf16 because we accumulate in fp32; see
    notes/stage2_findings.md).
    """

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
        mm_token_type_ids=None,
        image_position_ids=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Multimodal: defer to the parent. Our K1 fast path is text-only.
        is_multimodal = any(
            v is not None for v in (
                pixel_values, image_grid_thw, num_images,
                pixel_attention_mask, image_sizes,
                token_type_ids, mm_token_type_ids, image_position_ids,
            )
        )
        if is_multimodal:
            return super()._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, logits_to_keep,
                batch_size=batch_size, compute_entropy=compute_entropy,
                pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                num_images=num_images,
                pixel_attention_mask=pixel_attention_mask,
                image_sizes=image_sizes,
                token_type_ids=token_type_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_position_ids=image_position_ids,
            )

        # Text-only fast path — mirror TRL's chunking + slicing logic exactly,
        # then call K1 instead of (selective_log_softmax + entropy_from_logits).
        bsz = batch_size or input_ids.size(0)
        all_logps: list[torch.Tensor] = []
        all_entropies: list[torch.Tensor] = []

        for start in range(0, input_ids.size(0), bsz):
            input_ids_batch = input_ids[start : start + bsz]
            attention_mask_batch = attention_mask[start : start + bsz]

            model_inputs = {
                "input_ids": input_ids_batch,
                "attention_mask": attention_mask_batch,
            }
            # `model_kwarg_keys` is a TRL-internal cache of the model's accepted kwargs
            if "logits_to_keep" in self.model_kwarg_keys:
                # +1 because the very-last logits position is dropped below
                model_inputs["logits_to_keep"] = logits_to_keep + 1
            model_inputs["use_cache"] = False

            logits = model(**model_inputs).logits
            # Drop next-token-pred position; keep the completion window
            logits = logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :]
            # Temperature scaling (in-place, matches TRL)
            logits.div_(self.temperature)
            completion_ids = input_ids_batch[:, -logits_to_keep:]

            # IMPORTANT: the slice above leaves `logits` non-contiguous in the
            # leading dims (V dim still has stride 1, but the batch stride is
            # the *original* L*V, not the sliced (L-1)*V). Calling
            # `.contiguous()` on the full [B, L', V] would allocate ~B*L*V*2
            # bytes — 1+ GB for typical GRPO shapes — and OOM the 4060.
            #
            # Workaround: iterate per batch element. `logits[b]` selects dim 0
            # and yields a [L', V] tensor with strides (V, 1) — naturally
            # contiguous, no copy needed. K1 then runs on each row chunk.
            #
            # (A future kernel rev that accepts strided logits would let us do
            # one fused launch instead of B; noted as Stage 6 follow-up.)
            B_inner = logits.size(0)
            for b in range(B_inner):
                row_logits = logits[b]
                row_ids = completion_ids[b]
                if not row_logits.is_contiguous():
                    row_logits = row_logits.contiguous()
                if not row_ids.is_contiguous():
                    row_ids = row_ids.contiguous()
                logp_b, ent_b, _lse_b = fused_logprob_entropy(row_logits, row_ids)
                all_logps.append(logp_b.unsqueeze(0))
                if compute_entropy:
                    all_entropies.append(ent_b.detach().unsqueeze(0))

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies
