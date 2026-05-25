from typing import Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist
from accelerate.logging import get_logger

from reward.rl_reward import compute_reward_per_func
from rollout.llada_rl_rollout import RolloutOutput
from trainer.logging_utils import CYAN, RESET

logger = get_logger(__name__, log_level="INFO")

# Rollout filtering

def filter_rollout_samples(
    rollout_output: RolloutOutput,
    num_generations: int,
    max_groups: int,
    reward_funcs: list,
) -> Tuple[RolloutOutput, Dict[str, float]]:
    rewards_per_func = compute_reward_per_func(
        completions=rollout_output.completions,
        ground_truths=rollout_output.ground_truths,
        reward_funcs=reward_funcs,
    )
    rewards = rewards_per_func.sum(dim=1)

    total_samples = rewards.shape[0]
    num_groups = total_samples // num_generations
    grouped_rewards = rewards.view(num_groups, num_generations)

    avg_reward_response_local = rewards.mean().item()
    avg_reward_task_local = grouped_rewards.max(dim=1).values.mean().item()

    if dist.is_initialized():
        reward_tensor = torch.tensor(
            [avg_reward_response_local, avg_reward_task_local], dtype=torch.float32, device="cuda"
        )
        dist.all_reduce(reward_tensor, op=dist.ReduceOp.AVG)
        avg_reward_response_global = reward_tensor[0].item()
        avg_reward_task_global = reward_tensor[1].item()
    else:
        avg_reward_response_global = avg_reward_response_local
        avg_reward_task_global = avg_reward_task_local

    rollout_metrics = {
        "avg_reward_response": avg_reward_response_global,
        "avg_reward_task": avg_reward_task_global,
    }

    logger.info(
        f"  {CYAN}Rollout stats: {num_groups} groups, "
        f"avg_reward_response={avg_reward_response_global:.4f}, "
        f"avg_reward_task={avg_reward_task_global:.4f}{RESET}"
    )

    if num_groups <= max_groups:
        logger.info(
            f"  Rollout filtering: {num_groups} groups <= max_groups={max_groups}, "
            f"no filtering needed."
        )
        return rollout_output, rollout_metrics

    group_std = grouped_rewards.std(dim=1)
    valid_mask = group_std > 1e-8
    valid_indices = torch.where(valid_mask)[0].tolist()
    invalid_indices = torch.where(~valid_mask)[0].tolist()

    logger.info(
        f"  Rollout filtering: {len(valid_indices)} valid groups "
        f"out of {num_groups} total, target={max_groups}"
    )

    if len(valid_indices) >= max_groups:
        selected_indices = sorted(
            np.random.choice(valid_indices, size=max_groups, replace=False).tolist()
        )
    else:
        remaining_needed = max_groups - len(valid_indices)

        all_correct_indices = []
        all_wrong_indices = []
        for g_idx in invalid_indices:
            if grouped_rewards[g_idx].mean().item() > 0:
                all_correct_indices.append(g_idx)
            else:
                all_wrong_indices.append(g_idx)

        logger.info(
            f"  Rollout filtering filler: {len(all_correct_indices)} all-correct, "
            f"{len(all_wrong_indices)} all-wrong, need {remaining_needed} fillers"
        )

        if len(all_correct_indices) >= remaining_needed:
            filler_indices = sorted(
                np.random.choice(all_correct_indices, size=remaining_needed, replace=False).tolist()
            )
        else:
            filler_indices = list(all_correct_indices)
            still_needed = remaining_needed - len(filler_indices)
            if all_wrong_indices:
                wrong_group_steps = []
                for g_idx in all_wrong_indices:
                    start = g_idx * num_generations
                    end = start + num_generations
                    avg_steps = (rollout_output.step_map[start:end].max(dim=1).values.float() + 1).mean().item()
                    wrong_group_steps.append((g_idx, avg_steps))
                wrong_group_steps.sort(key=lambda x: x[1], reverse=True)
                filler_indices += sorted(g_idx for g_idx, _ in wrong_group_steps[:still_needed])

        selected_indices = sorted(valid_indices + filler_indices)

    logger.info(
        f"  Rollout filtering: selected {len(selected_indices)} groups "
        f"(valid={min(len(valid_indices), max_groups)}, "
        f"filler={len(selected_indices) - min(len(valid_indices), max_groups)})"
    )

    sample_indices = []
    for g_idx in selected_indices:
        start = g_idx * num_generations
        sample_indices.extend(range(start, start + num_generations))
    sample_indices = torch.tensor(sample_indices, dtype=torch.long)

    filtered_output = RolloutOutput(
        prompt_ids=rollout_output.prompt_ids[sample_indices],
        prompt_mask=rollout_output.prompt_mask[sample_indices],
        completion_ids=rollout_output.completion_ids[sample_indices],
        completion_mask=rollout_output.completion_mask[sample_indices],
        step_map=rollout_output.step_map[sample_indices],
        token_probs=rollout_output.token_probs[sample_indices],
        completions=[rollout_output.completions[i] for i in sample_indices],
        ground_truths=[rollout_output.ground_truths[i] for i in sample_indices],
    )

    return filtered_output, rollout_metrics


# Rollout sample step/prob analysis

def analyze_rollout_step_probs(
    step_map: torch.Tensor,
    token_probs: torch.Tensor,
    completion_mask: torch.Tensor,
    prob_threshold: float = 0.9,
) -> Dict[str, float]:
    """Analyze per-sample step counts and low-probability steps.

    Returns a dict with:
      - avg_steps_per_sample: mean of max step index across samples
      - avg_low_prob_steps:   mean count of steps whose avg token prob < threshold
    """
    B = step_map.shape[0]
    mask = completion_mask.bool()

    max_steps_per_sample = step_map.max(dim=1).values.float() + 1  # [B], 0-indexed → count
    avg_steps = max_steps_per_sample.mean().item()

    low_prob_step_counts = []
    for b in range(B):
        valid = mask[b]
        if not valid.any():
            low_prob_step_counts.append(0.0)
            continue
        steps_b = step_map[b][valid]
        probs_b = token_probs[b][valid]
        unique_steps = steps_b.unique()
        low_count = 0
        for s in unique_steps:
            step_mask = steps_b == s
            if step_mask.any():
                avg_prob = probs_b[step_mask].mean().item()
                if avg_prob < prob_threshold:
                    low_count += 1
        low_prob_step_counts.append(float(low_count))

    avg_low_prob_steps = sum(low_prob_step_counts) / max(B, 1)

    return {
        "avg_steps_per_sample": avg_steps,
        "avg_low_prob_steps": avg_low_prob_steps,
        "max_steps_per_sample": max_steps_per_sample.max().item(),
        "min_steps_per_sample": max_steps_per_sample.min().item(),
    }