"""
masking_strategy.py -- Multiple masking strategies for per-token log probability.

Design principles:
    Different completions and ref/old/current models under the same prompt
    must use exactly the same mask positions. Therefore:
    1. The masking schedule is precomputed once after rollout.
    2. The result is cached and consumed by old/ref/current models.

Strategies:
    - sequential_masking:     Sample k steps per sample from step_map;
                              distribution controlled by sample_temperature:
                                  >= UNIFORM_TEMP_THRESHOLD (incl. +inf) -> uniform
                                  > 0  -> softmax(weight / T) weighted sampling
                                    0  -> argmax / topk
                              weight(step) = -mean_log_prob(tokens in step).
    - random_masking:         Randomly select positions by ratio, build multi-step masks.
    - coupled_random_masking: Complementary random masks, 50/50 split per sample.
    - all_masking:            Masking on all completion positions.

Each element in masking_inputs is a triple (step_id, should_mask, current_pos)(Only for Sequential Masking):
    - step_id:     step number (-1 for dummy)
    - should_mask: (gen_len,) bool, positions to mask (step_map >= step_id)
    - current_pos:  (gen_len,) bool, positions actually decoded (step_map == step_id)

Cross-rank consistency:
    After local construction, _sync_max_steps() uses dist.all_reduce(MAX)
    to align all ranks to the global maximum, then pads with dummy steps.
    This prevents NCCL deadlocks.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.distributed as dist
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")


# Constants

# Threshold above which sample_temperature triggers uniform sampling.
# Surprisal weights (-log p) typically range 0-30; at T >= 1e4 the
# softmax distribution is effectively uniform, so we skip weight
# computation and drop the token_probs dependency.
UNIFORM_TEMP_THRESHOLD: float = 1e4


def _is_effectively_uniform(temperature: float) -> bool:
    """Check whether temperature is large enough to be equivalent to uniform sampling."""
    return math.isinf(temperature) or temperature >= UNIFORM_TEMP_THRESHOLD


# MaskingSchedule data structure

@dataclass
class MaskingSchedule:
    """
    Precomputed masking schedule.

    masking_inputs: per-sample list of list of (step_id, should_mask, current_pos)
    strategy: str
    """
    masking_inputs: list
    strategy: str

    def to(self, device: torch.device) -> "MaskingSchedule":
        return MaskingSchedule(
            masking_inputs=[
                [(sid, m.to(device), u.to(device)) for sid, m, u in sample_masks]
                for sample_masks in self.masking_inputs
            ],
            strategy=self.strategy,
        )


# Cross-rank synchronization

def _sync_max_steps(local_max_steps: int) -> int:
    """
    Align local_max_steps across ranks via dist.all_reduce(MAX).
    Returns local_max_steps directly if distributed is not initialized.
    Result is at least 1 (guarantees a dummy step exists).
    """
    if not dist.is_initialized():
        return max(local_max_steps, 1)
    steps_tensor = torch.tensor([local_max_steps], dtype=torch.long, device="cuda")
    dist.all_reduce(steps_tensor, op=dist.ReduceOp.MAX)
    return max(steps_tensor.item(), 1)


def _pad_to_global_max(
    masking_inputs: list,
    global_max_steps: int,
    gen_len: int,
    rank: int,
) -> list:
    """
    Pad each sample's masking_inputs with dummy steps up to global_max_steps.
    Logs an ERROR if the length is still wrong after padding.
    """
    B = len(masking_inputs)
    for b in range(B):
        while len(masking_inputs[b]) < global_max_steps:
            masking_inputs[b].append((
                -1,
                torch.zeros(gen_len, dtype=torch.bool),
                torch.zeros(gen_len, dtype=torch.bool),
            ))
        if len(masking_inputs[b]) != global_max_steps:
            logger.error(
                f"[Rank {rank}] sample {b} has {len(masking_inputs[b])} steps "
                f"after padding, expected {global_max_steps}"
            )
    return masking_inputs


# Build interface

def build_masking_schedule(
    step_map: torch.Tensor,
    completion_mask: torch.Tensor,
    strategy: str,
    k: int = 8,
    mask_ratio: float = 0.3,
    num_generations: int = 1,
    seed: Optional[int] = None,
    token_probs: Optional[torch.Tensor] = None,
    sample_temperature: float = 1.0,
) -> MaskingSchedule:
    """
    Build a masking schedule.

    Args:
        step_map:            (B, gen_len) step assignment per token.
        completion_mask:     (B, gen_len) valid token mask.
        strategy:            Strategy name.
        k:                   Number of steps for sequential_masking / random_masking.
        mask_ratio:          Mask ratio for random_masking.
        num_generations:     Generations per prompt (interface compat, unused).
        seed:                Random seed.
        token_probs:         (B, gen_len) per-token generation probabilities from
                             rollout; required by sequential_masking when temperature
                             is in the weighted regime.
        sample_temperature:  Step sampling temperature; +inf or >= UNIFORM_TEMP_THRESHOLD
                             means uniform; 0 means argmax.
    """
    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    if strategy == "sequential_masking":
        # Only validate token_probs when temperature is in the weighted regime
        if (not _is_effectively_uniform(sample_temperature)) and token_probs is None:
            raise ValueError(
                f"sequential_masking with sample_temperature={sample_temperature} "
                f"(in weighted regime) requires token_probs. "
                f"Pass rollout_output.token_probs, or set "
                f"sample_temperature=float('inf') / >= {UNIFORM_TEMP_THRESHOLD:g} "
                f"for uniform sampling."
            )
        return _build_sequential_masking(
            step_map, completion_mask, token_probs, k, rng,
            temperature=sample_temperature,
        )
    elif strategy == "random_masking":
        return _build_random_masking(step_map, completion_mask, mask_ratio, k, rng)
    elif strategy == "coupled_random_masking":
        return _build_coupled_random_masking(step_map, completion_mask, num_generations, rng)
    elif strategy == "all_masking":
        return _build_all_masking(step_map, completion_mask)
    else:
        raise ValueError(
            f"Unknown masking strategy '{strategy}'. "
            f"Available: sequential_masking, "
            f"random_masking, coupled_random_masking, all_masking"
        )


# sequential_masking

def _build_sequential_masking(
    step_map: torch.Tensor,
    completion_mask: torch.Tensor,
    token_probs: Optional[torch.Tensor],
    k: int,
    rng: Optional[torch.Generator],
    temperature: float = 1.0,
) -> MaskingSchedule:
    """
    Sample k steps per sample. Sampling weights depend on token_probs and temperature:

        weight(step) = -mean_log_prob(tokens in step)
        p(step) ~ softmax(weight / temperature)

    Special cases:
        - temperature = +inf or >= UNIFORM_TEMP_THRESHOLD: uniform (no token_probs needed)
        - temperature = 0: argmax/topk (deterministic; pad with highest-weight step)
        - Degenerate weights: fall back to uniform

    Sampling:
        - candidates >= k: sample k without replacement
        - candidates <  k: take all, then fill remaining with replacement

    Cross-rank alignment via _sync_max_steps.
    """
    B, gen_len = step_map.shape
    rank = dist.get_rank() if dist.is_initialized() else 0

    is_uniform = _is_effectively_uniform(temperature)

    # Log when a large (but finite) temperature triggers the uniform path
    if is_uniform and (not math.isinf(temperature)) and rank == 0:
        logger.info(
            f"sequential_masking: temperature={temperature:g} >= "
            f"UNIFORM_TEMP_THRESHOLD ({UNIFORM_TEMP_THRESHOLD:g}); "
            f"Use uniform sampling. token_probs is ignored."
        )

    masking_inputs = [[] for _ in range(B)]

    for b in range(B):
        sample_step_map        = step_map[b]
        sample_completion_mask = completion_mask[b]

        max_step = sample_step_map.max().item()
        if max_step <= 0:
            logger.warning(
                f"[Rank {rank}] Sample {b}: max_step={max_step} <= 0, "
                f"will use dummy steps"
            )
            continue

        # Collect valid candidate steps and their weights
        all_candidates: List[int] = []
        step_weights:   List[float] = []  # only used when is_uniform=False
        for s in range(0, int(max_step) + 1):
            step_mask = (sample_step_map == s) & sample_completion_mask.bool()
            if step_mask.sum().item() == 0:
                continue
            all_candidates.append(s)
            if not is_uniform:
                # token_probs validated non-None by caller in weighted regime
                probs = token_probs[b][step_mask].float().clamp(min=1e-10)
                step_weights.append(-probs.log().mean().item())

        if len(all_candidates) == 0:
            logger.warning(
                f"[Rank {rank}] Sample {b}: no valid candidates, will use dummy steps"
            )
            continue

        selected_indices = _sample_k_indices(
            num_candidates=len(all_candidates),
            k=k,
            step_weights=step_weights if not is_uniform else None,
            temperature=temperature,
            rng=rng,
        )
        selected_steps = [all_candidates[i] for i in selected_indices]

        for step in selected_steps:
            should_mask = (sample_step_map >= step)
            current_pos  = (sample_step_map == step)
            masking_inputs[b].append(
                (step, should_mask.cpu(), current_pos.cpu())
            )

    # Sync step counts across ranks and pad
    local_max  = max((len(m) for m in masking_inputs), default=0)
    global_max = _sync_max_steps(local_max)
    masking_inputs = _pad_to_global_max(masking_inputs, global_max, gen_len, rank)

    if math.isinf(temperature):
        temp_str = "inf"
    elif is_uniform:
        temp_str = f"{temperature:g}(uniform)"
    else:
        temp_str = f"{temperature:g}"
    logger.info(
        f"[Rank {rank}] sequential_masking: k={k}, temperature={temp_str}, "
        f"local_max_steps={local_max}, global_max_steps={global_max}, "
    )
    _log_min_unmask(masking_inputs, rank, "sequential_masking")

    return MaskingSchedule(
        masking_inputs=masking_inputs,
        strategy="sequential_masking",
    )


def _sample_k_indices(
    num_candidates: int,
    k: int,
    step_weights: Optional[List[float]],
    temperature: float,
    rng: Optional[torch.Generator],
) -> List[int]:
    """
    Select k indices from [0, num_candidates), allowing repeats if fewer than k.

    Branches:
        - step_weights is None       -> uniform sampling
        - weights have near-zero std -> fall back to uniform
        - temperature == 0           -> argmax / topk (deterministic)
        - otherwise                  -> softmax(weights / temperature) sampling
    """
    is_uniform = step_weights is None

    if not is_uniform:
        weights = torch.tensor(step_weights, dtype=torch.float32)
        if weights.std() < 1e-8:
            is_uniform = True

    # Uniform sampling path
    if is_uniform:
        if num_candidates >= k:
            perm = torch.randperm(num_candidates, generator=rng)[:k]
            return perm.tolist()
        # Fewer than k: take all with repeats, fill remainder uniformly
        repeats   = k // num_candidates
        remainder = k % num_candidates
        selected  = list(range(num_candidates)) * repeats
        if remainder > 0:
            extra = torch.randperm(num_candidates, generator=rng)[:remainder]
            selected += extra.tolist()
        return selected

    # Weighted sampling path
    if temperature == 0:
        if num_candidates >= k:
            _, topk_idx = weights.topk(k)
            return topk_idx.tolist()
        argmax_idx = int(weights.argmax().item())
        selected   = list(range(num_candidates))
        selected  += [argmax_idx] * (k - num_candidates)
        return selected

    sampling_probs = torch.softmax(weights / temperature, dim=0)

    if num_candidates >= k:
        return torch.multinomial(
            sampling_probs, k, replacement=False, generator=rng
        ).tolist()

    selected   = list(range(num_candidates))
    remaining  = k - num_candidates
    extra = torch.multinomial(
        sampling_probs, remaining, replacement=True, generator=rng
    )
    selected += extra.tolist()
    return selected


# random_masking

def _build_random_masking(
    step_map: torch.Tensor,
    completion_mask: torch.Tensor,
    mask_ratio: float,
    k: int,
    rng: Optional[torch.Generator],
) -> MaskingSchedule:
    """
    Build k independent masks, each covering mask_ratio of valid positions.
    Masks may overlap across steps.

    Per step:
        - should_mask: independently sampled positions
        - current_pos:  same as should_mask (steps are independent)
    """
    B, gen_len = step_map.shape
    rank = dist.get_rank() if dist.is_initialized() else 0

    masking_inputs = [[] for _ in range(B)]

    for b in range(B):
        sample_valid  = (completion_mask[b] > 0)
        valid_indices = sample_valid.nonzero(as_tuple=True)[0]
        num_valid     = valid_indices.shape[0]

        if num_valid == 0:
            continue

        num_select = max(1, int(num_valid * mask_ratio))

        for step_i in range(k):
            perm = torch.randperm(num_valid, generator=rng)[:num_select]
            selected_indices = valid_indices[perm]

            should_mask = torch.zeros(gen_len, dtype=torch.bool)
            should_mask[selected_indices] = True

            current_pos = should_mask.clone()

            masking_inputs[b].append(
                (step_i, should_mask.cpu().clone(), current_pos.cpu().clone())
            )

    local_max  = max((len(m) for m in masking_inputs), default=0)
    global_max = _sync_max_steps(local_max)
    masking_inputs = _pad_to_global_max(masking_inputs, global_max, gen_len, rank)

    logger.info(
        f"[Rank {rank}] random_masking: k={k}, mask_ratio={mask_ratio}, "
        f"local_max_steps={local_max}, global_max_steps={global_max}, "
    )
    _log_min_unmask(masking_inputs, rank, "random_masking")

    return MaskingSchedule(
        masking_inputs=masking_inputs,
        strategy="random_masking",
    )


# coupled_random_masking

def _build_coupled_random_masking(
    step_map: torch.Tensor,
    completion_mask: torch.Tensor,
    num_generations: int,
    rng: Optional[torch.Generator],
) -> MaskingSchedule:
    """
    Complementary random masking: randomly split each sample's valid positions
    50/50 into two complementary sets.

    Step 0: unmask set_a, mask set_b
    Step 1: unmask set_b, mask set_a

    Together they cover all completion tokens exactly once.
    """
    B, gen_len = step_map.shape
    rank = dist.get_rank() if dist.is_initialized() else 0

    masking_inputs = [[] for _ in range(B)]

    for b in range(B):
        sample_valid  = (completion_mask[b] > 0)
        valid_indices = sample_valid.nonzero(as_tuple=True)[0]
        num_valid     = valid_indices.shape[0]

        if num_valid == 0:
            dummy = torch.zeros(gen_len, dtype=torch.bool)
            masking_inputs[b].append((-1, dummy.clone(), dummy.clone()))
            masking_inputs[b].append((-1, dummy.clone(), dummy.clone()))
            continue

        perm        = torch.randperm(num_valid, generator=rng)
        split_point = max(1, min(num_valid // 2, num_valid - 1))

        indices_a = valid_indices[perm[:split_point]]
        indices_b = valid_indices[perm[split_point:]]

        set_a = torch.zeros(gen_len, dtype=torch.bool)
        set_b = torch.zeros(gen_len, dtype=torch.bool)
        set_a[indices_a] = True
        set_b[indices_b] = True

        masking_inputs[b].append((0, set_b.cpu().clone(), set_b.cpu().clone()))
        masking_inputs[b].append((1, set_a.cpu().clone(), set_a.cpu().clone()))

    local_max  = max((len(m) for m in masking_inputs), default=0)
    global_max = _sync_max_steps(local_max)
    masking_inputs = _pad_to_global_max(masking_inputs, global_max, gen_len, rank)

    logger.info(
        f"[Rank {rank}] coupled_random_masking: "
        f"B={B}, global_max_steps={global_max}, "
        f"total completion tokens={(completion_mask > 0).sum().item()}"
    )

    return MaskingSchedule(
        masking_inputs=masking_inputs,
        strategy="coupled_random_masking",
    )


# all_masking

def _build_all_masking(
    step_map: torch.Tensor,
    completion_mask: torch.Tensor,
) -> MaskingSchedule:
    """
    All-masking baseline: mask all completion positions in a single step,
    predicting all tokens at once (absorbing diffusion training objective).

    Produces a single step per sample:
        should_mask = completion_mask  (all valid positions masked)
        current_pos  = completion_mask  (all valid tokens evaluated)

    Useful as a control experiment baseline (standard GRPO/PPO behavior)
    or for verifying the masking framework introduces no bias.

    Still goes through _sync_max_steps + pad to prevent cross-rank misalignment.
    """
    B, gen_len = step_map.shape
    rank = dist.get_rank() if dist.is_initialized() else 0

    masking_inputs = [[] for _ in range(B)]

    for b in range(B):
        valid = (completion_mask[b] > 0)

        # Keep a dummy step even with no valid tokens to maintain alignment
        if valid.sum().item() == 0:
            dummy = torch.zeros(gen_len, dtype=torch.bool)
            masking_inputs[b].append((-1, dummy.clone(), dummy.clone()))
            continue

        should_mask = valid.clone().cpu()   # mask all valid completion positions
        current_pos  = valid.clone().cpu()  # evaluate all valid positions

        # step_id=0 indicates a real (non-dummy) step
        masking_inputs[b].append((0, should_mask, current_pos))

    # Use the same sync flow as other strategies to prevent cross-rank misalignment
    local_max  = max((len(m) for m in masking_inputs), default=0)
    global_max = _sync_max_steps(local_max)
    masking_inputs = _pad_to_global_max(masking_inputs, global_max, gen_len, rank)

    logger.info(
        f"[Rank {rank}] all_masking: B={B}, "
        f"local_max_steps={local_max}, global_max_steps={global_max}, "
        f"total completion tokens={(completion_mask > 0).sum().item()}"
    )
    _log_min_unmask(masking_inputs, rank, "all_masking")

    return MaskingSchedule(
        masking_inputs=masking_inputs,
        strategy="all_masking",
    )


# Diagnostics

def _log_min_unmask(masking_inputs: list, rank: int, tag: str) -> None:
    """Log the minimum unmask token count across all non-dummy steps (for debugging)."""
    min_count = float("inf")
    min_info  = None
    for i, sample in enumerate(masking_inputs):
        for step_s, _, unmask in sample:
            if step_s == -1:
                continue
            count = unmask.sum().item()
            if count < min_count:
                min_count = count
                min_info  = (i, step_s, count)
    if min_info is not None:
        logger.info(
            f"[Rank {rank}] {tag}: min unmask tokens = {min_info[2]} "
            f"(sample {min_info[0]}, step {min_info[1]})"
        )
