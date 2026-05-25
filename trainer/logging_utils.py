import json
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")

GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"


# Cross-rank metric gathering & JSONL logging

def log_step_to_jsonl(
    log_path: Path,
    global_step: int,
    epoch: int,
    lr: float,
    grad_norm: float,
    result: dict,
    strategy_key: str,
    reward_funcs: list,
    accelerator: Accelerator,
    rollout_metrics: Optional[Dict[str, float]] = None,
    timing: Optional[Dict[str, float]] = None,
    step_prob_metrics: Optional[Dict[str, float]] = None,
):
    rewards = result.get("rewards", None)
    rewards_per_func = result.get("rewards_per_func", None)

    reward_mean = 0.0
    reward_std = 0.0
    per_func_means = {}

    if rewards is not None:
        all_rewards = accelerator.gather(rewards.to(accelerator.device))
        reward_mean = all_rewards.float().mean().item()
        reward_std = all_rewards.float().std().item()

    if rewards_per_func is not None:
        all_rpf = accelerator.gather(rewards_per_func.to(accelerator.device))
        func_names = list(reward_funcs) if reward_funcs else [f"func_{i}" for i in range(all_rpf.shape[1])]
        for fi, fname in enumerate(func_names):
            per_func_means[fname] = all_rpf[:, fi].float().mean().item()

    # Pack all scalars into a single tensor
    metric_keys = [
        "loss", "clip_ratio", "mean_ratio",
        "effective_positions_ratio",
        "mean_kl", "mean_entropy",
    ]

    loss_val = result["loss"].item() if isinstance(result["loss"], torch.Tensor) else result["loss"]

    local_scalars = torch.tensor(
        [loss_val] + [result[k] for k in metric_keys[1:]] + [grad_norm],
        device=accelerator.device,
        dtype=torch.float32,
    )
    gathered_scalars = accelerator.gather(local_scalars.unsqueeze(0)).mean(dim=0)

    gathered_metrics = dict(zip(metric_keys, gathered_scalars[:len(metric_keys)].tolist()))
    grad_norm_gathered = gathered_scalars[-1].item()

    if accelerator.is_main_process:
        log_entry = {
            "global_step": global_step,
            "epoch": epoch,
            "lr": lr,
            "grad_norm": grad_norm_gathered,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "reward_per_func": per_func_means,
            "strategy": strategy_key,
            **gathered_metrics,
        }
        if rollout_metrics is not None:
            log_entry["rollout"] = rollout_metrics
        if timing is not None:
            log_entry["timing"] = timing
        if step_prob_metrics is not None:
            log_entry["step_prob"] = step_prob_metrics
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return gathered_metrics