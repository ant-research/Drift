import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import gc
import json
import logging
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, DeepSpeedPlugin
from jinja2 import Template
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from reward.rl_reward import compute_reward_per_func
from rollout.dream_rollout import dream_rollout
from rollout.llada_rollout import llada_rollout, RolloutOutput
from trainer.logging_utils import log_step_to_jsonl, GREEN, YELLOW, MAGENTA, CYAN, RESET
from trainer.lr_schedulers import get_scheduler
from trainer.masking_strategy import MaskingSchedule, build_masking_schedule
from trainer.model_utils import (
    get_model_type, load_train_model, load_ref_model, load_tokenizer,
    resolve_model_dtype, prepare_deepspeed, load_rollout_model,
)
from trainer.prompts import get_default_system_prompt
from trainer.rollout_utils import filter_rollout_samples, analyze_rollout_step_probs
from trainer.utils import get_config, save_checkpoint_top_k, log_gpu_memory

logger = get_logger(__name__, log_level="INFO")


# KL divergence estimators
def compute_kl_divergence(
    cur_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    kl_type: str = "k3",
) -> torch.Tensor:

    log_r = ref_logps - cur_logps
    if kl_type == "k1":
        per_token_kl = -log_r
    elif kl_type == "k2":
        per_token_kl = 0.5 * log_r ** 2
    elif kl_type == "k3":
        per_token_kl = (torch.exp(log_r) - 1.0) - log_r
    else:
        raise ValueError(f"Unknown kl_type='{kl_type}'. Supported: 'k1', 'k2', 'k3'.")
    return per_token_kl


# Dataset
class TrainDataset(Dataset):
    def __init__(self, questions: List[str], answers: List[str]):
        assert len(questions) == len(answers)
        self.questions = questions
        self.answers = answers

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return {"question": self.questions[idx], "answer": self.answers[idx]}


def simple_collate(batch):
    return {
        "questions": [item["question"] for item in batch],
        "answers": [item["answer"] for item in batch],
    }



# Rollout entry point
@torch.no_grad()
def run_rollout(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    ground_truths: List[str],
    config,
    model_type: str,
    num_generations: int = 4,
    batch_size: int = 4,
) -> RolloutOutput:
    """Dispatch rollout based on model type."""
    if model_type == "llada":
        return llada_rollout(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            ground_truths=ground_truths,
            config=config,
            num_generations=num_generations,
            batch_size=batch_size,
        )
    else:
        return dream_rollout(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            ground_truths=ground_truths,
            config=config,
            num_generations=num_generations,
            batch_size=batch_size,
        )


# Per-token log probability
@torch.no_grad()
def compute_per_token_logps(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    schedule: MaskingSchedule,
    mask_id: int,
    tokenizer=None,
    model_type: str = "llada",
) -> torch.Tensor:

    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    completion_ids = completion_ids.to(device)
    completion_mask = completion_mask.to(device)
    sched = schedule.to(device)

    full_seq = torch.cat([prompt_ids, completion_ids], dim=1)
    prompt_len = prompt_ids.shape[1]
    B, gen_len = completion_ids.shape

    total_steps = len(sched.masking_inputs[0])
    # [total_steps, B, gen_len]: each step stores logps under its own masking context
    per_step_logps = torch.zeros(total_steps, B, gen_len, device=device, dtype=torch.float32)

    for i in range(total_steps):
        should_masks = torch.stack(
            [sched.masking_inputs[b][i][1].to(device) for b in range(B)]
        )

        if should_masks.sum() == 0:
            logger.info(f"Step {i + 1} Skipped (dummy) / Total Step {total_steps}")
            del should_masks
            continue

        masked_input = full_seq.clone()
        masked_input[:, prompt_len:][should_masks] = mask_id

        logits = model(masked_input, use_cache=False).logits[:, prompt_len:, :]
        # Dream logits shift: next-token → current-token alignment
        if model_type == "dream":
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        token_logps = torch.gather(
            log_probs, dim=-1, index=completion_ids.unsqueeze(-1)
        ).squeeze(-1)  # [B, gen_len]

        per_step_logps[i] = token_logps * completion_mask.float()

        del logits, log_probs, token_logps, masked_input, should_masks

    return per_step_logps  # [total_steps, B, gen_len]


# Trainer (backward/step handled by ds_engine)
class DriftTrainer:
    def __init__(self, config, mask_id: int, tokenizer=None, model_type: str = "llada"):
        self.epsilon_low = getattr(config.training, "epsilon_low", 0.2)
        self.epsilon_high = getattr(config.training, "epsilon_high", 0.2)
        self.beta = getattr(config.training, "beta", 0.0)
        self.num_generations = getattr(config.rollout, "num_generations", 4)
        self.num_iterations = getattr(config.training, "num_iterations", 1)
        self.kl_type = getattr(config.training, "kl_type", "k3")
        self.entropy_coeff = getattr(config.training, "entropy_coeff", 0.0)
        self.low_prob_threshold = getattr(config.training, "low_prob_threshold", 0.2)
        self.loss_norm = getattr(config.training, "loss_norm", "sample")  # "token" or "sample"
        self.mask_id = mask_id
        self.config = config
        self.tokenizer = tokenizer
        self.model_type = model_type
        

    def _compute_per_token_adv_loss(
        self,
        cur_logps: torch.Tensor,
        old_logps: torch.Tensor,
        advantages: torch.Tensor,
        log_probs: Optional[torch.Tensor] = None,
        ref_entropy: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.num_iterations == 1:
            per_token_loss = -cur_logps * advantages.unsqueeze(1)
            ratio = torch.ones_like(cur_logps)
            is_clipped = torch.zeros_like(cur_logps)
        else:
            ratio = torch.exp(cur_logps - old_logps)
            clamped_ratio = torch.clamp(
                ratio, 1.0 - self.epsilon_low, 1.0 + self.epsilon_high
            )
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clamped_ratio * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
            is_clipped = (per_token_loss1 < per_token_loss2).float()

        if log_probs is not None:
            per_token_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
            if self.entropy_coeff != 0 and ref_entropy is not None:
                # entropy penalty: penalize entropy deviation from ref model (both directions)
                # |H_ref - H_cur|: prevent both collapse and excessive randomness
                entropy_delta = (ref_entropy - per_token_entropy).abs()
                per_token_loss = per_token_loss + self.entropy_coeff * entropy_delta
            elif self.entropy_coeff != 0:
                # fallback: no ref model, use absolute entropy (legacy behavior)
                per_token_loss = per_token_loss + self.entropy_coeff * per_token_entropy
            else:
                per_token_entropy = per_token_entropy.detach()
        else:
            per_token_entropy = torch.zeros_like(per_token_loss)

        return per_token_loss, ratio, is_clipped, per_token_entropy

    def _compute_per_token_kl_loss(
        self,
        cur_logps: torch.Tensor,
        ref_logps: torch.Tensor,
    ) -> torch.Tensor:
    
        per_token_kl = compute_kl_divergence(
            cur_logps=cur_logps,
            ref_logps=ref_logps,
            kl_type=self.kl_type,
        )

        return per_token_kl


    def _forward_and_gather(
        self,
        model: torch.nn.Module,
        full_seq: torch.Tensor,
        prompt_len: int,
        completion_ids: torch.Tensor,
        should_masks: torch.Tensor,
    ):
        masked_input = full_seq.clone().detach()
        masked_input[:, prompt_len:][should_masks] = self.mask_id
        
        logits = model(masked_input, use_cache=False).logits[:, prompt_len:, :]
        # Dream logits shift: next-token → current-token alignment
        if self.model_type == "dream":
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        cur_logps = torch.gather(
            log_probs, dim=-1, index=completion_ids.unsqueeze(-1)
        ).squeeze(-1)

        return cur_logps, log_probs, logits


    @torch.no_grad()
    def _forward_and_gather_ref(
        self,
        ref_model: torch.nn.Module,
        full_seq: torch.Tensor,
        prompt_len: int,
        completion_ids: torch.Tensor,
        should_masks: torch.Tensor,
        return_entropy: bool = False,
    ):
        masked_input = full_seq.clone().detach()
        masked_input[:, prompt_len:][should_masks] = self.mask_id

        logits = ref_model(masked_input, use_cache=False).logits[:, prompt_len:, :]
        # Dream logits shift: next-token → current-token alignment
        if self.model_type == "dream":
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        ref_logps = torch.gather(
            log_probs, dim=-1, index=completion_ids.unsqueeze(-1)
        ).squeeze(-1)

        if return_entropy:
            ref_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
            del logits, log_probs
            return ref_logps, ref_entropy

        del logits, log_probs
        return ref_logps


    def _compute_loss(
        self,
        model: torch.nn.Module,
        ref_model: Optional[torch.nn.Module],
        sched,
        full_seq: torch.Tensor,
        prompt_len: int,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        old_logps: torch.Tensor,
        advantages: torch.Tensor,
        ds_engine,
        B: int,
        device: torch.device,
    ) -> dict:
        mean_loss_sum = 0.0
        mean_adv_loss_sum = 0.0
        mean_clip_sum = 0.0
        mean_ratio_sum = 0.0
        mean_entropy_sum = 0.0
        mean_kl_loss_sum = 0.0
        total_mask_sum = 0.0

        total_steps = len(sched.masking_inputs[0])

        for i in range(total_steps):

            should_masks = torch.stack(
                [sched.masking_inputs[b][i][1].to(device) for b in range(B)]
            )
            
            # Decoded position at current timestep
            cur_pos = torch.stack(
                [sched.masking_inputs[b][i][2].to(device) for b in range(B)]
            )

            # old_logps for this step: [B, gen_len], same masking context as cur_logps
            old_logps_step = old_logps[i]  # [B, gen_len]

            step_effective_mask = should_masks.float() * completion_mask.float()

            cur_logps_step, log_probs, logits = self._forward_and_gather(
                model, full_seq, prompt_len, completion_ids, should_masks,
            )

            # Low-prob filtering
            if self.low_prob_threshold > 0:
                with torch.no_grad():
                    log_threshold = math.log(self.low_prob_threshold)
                    # PPO-clip: filter on old_logps_step (rollout policy, same context)
                    filter_logps = old_logps_step if self.num_iterations > 1 else cur_logps_step
                    low_prob_token = filter_logps < log_threshold
                    prob_valid_mask = step_effective_mask * (~low_prob_token).float()

                    # Ensure fully-filtered samples still contribute via cur_pos
                    step_effective_mask = (prob_valid_mask.bool() | (cur_pos.bool() & completion_mask.bool())).float()

            # --- Ref model forward (for KL and/or entropy penalty) ---
            need_ref_kl = self.beta > 0 and ref_model is not None
            need_ref_entropy = self.entropy_coeff != 0 and ref_model is not None
            ref_logps_step = None
            ref_entropy_step = None

            if need_ref_kl or need_ref_entropy:
                if need_ref_entropy:
                    ref_logps_step, ref_entropy_step = self._forward_and_gather_ref(
                        ref_model, full_seq, prompt_len, completion_ids, should_masks,
                        return_entropy=True,
                    )
                else:
                    ref_logps_step = self._forward_and_gather_ref(
                        ref_model, full_seq, prompt_len, completion_ids, should_masks,
                    )

            # --- Advantage loss ---
            per_token_adv_loss, ratio_step, is_clipped_step, per_token_entropy = \
                self._compute_per_token_adv_loss(
                    cur_logps_step, old_logps_step, advantages, log_probs=log_probs,
                    ref_entropy=ref_entropy_step,
                )

            # --- KL loss---
            if need_ref_kl:
                per_token_kl = self._compute_per_token_kl_loss(
                    cur_logps_step, ref_logps_step,
                )
                per_token_loss_step = per_token_adv_loss + per_token_kl * self.beta
                with torch.no_grad():
                    if self.loss_norm == "sample":
                        per_sample_kl_mask = step_effective_mask.sum(dim=1).clamp(min=1)
                        mean_kl_loss_sum += ((per_token_kl * step_effective_mask).sum(dim=1) / per_sample_kl_mask).mean().item()
                    else:
                        step_mask_count = step_effective_mask.sum().clamp(min=1)
                        mean_kl_loss_sum += ((per_token_kl * step_effective_mask).sum() / step_mask_count).item()
            else:
                per_token_loss_step = per_token_adv_loss
            del ref_logps_step, ref_entropy_step

            del log_probs

            # --- Logging accumulators ---

            with torch.no_grad():
                if self.loss_norm == "sample":
                    per_sample_mask = step_effective_mask.sum(dim=1).clamp(min=1)  # [B]
                    mean_loss_sum        += ((per_token_loss_step  * step_effective_mask).sum(dim=1) / per_sample_mask).mean().item()
                    mean_adv_loss_sum    += ((per_token_adv_loss   * step_effective_mask).sum(dim=1) / per_sample_mask).mean().item()
                    mean_clip_sum        += ((is_clipped_step      * step_effective_mask).sum(dim=1) / per_sample_mask).mean().item()
                    mean_ratio_sum       += ((ratio_step           * step_effective_mask).sum(dim=1) / per_sample_mask).mean().item()
                    mean_entropy_sum += ((per_token_entropy * step_effective_mask).sum(dim=1) / per_sample_mask).mean().item()
                else:
                    step_mask_count = step_effective_mask.sum().clamp(min=1)
                    mean_loss_sum        += ((per_token_loss_step   * step_effective_mask).sum() / step_mask_count).item()
                    mean_adv_loss_sum    += ((per_token_adv_loss    * step_effective_mask).sum() / step_mask_count).item()
                    mean_clip_sum        += ((is_clipped_step       * step_effective_mask).sum() / step_mask_count).item()
                    mean_ratio_sum       += ((ratio_step            * step_effective_mask).sum() / step_mask_count).item()
                    mean_entropy_sum += ((per_token_entropy * step_effective_mask).sum() / step_mask_count).item()

                total_mask_sum += step_effective_mask.sum().item()

            # --- Backward ---
            if self.loss_norm == "sample":
                per_sample_mask_sum = step_effective_mask.sum(dim=1).clamp(min=1)  # [B]
                per_sample_loss = (per_token_loss_step * step_effective_mask).sum(dim=1) / per_sample_mask_sum
                step_loss_scaled = per_sample_loss.mean() / total_steps
            else:
                step_mask_sum = step_effective_mask.sum().clamp(min=1)
                step_loss_scaled = (per_token_loss_step * step_effective_mask).sum() / step_mask_sum / total_steps

            # Only set boundary=True on last step to trigger allreduce
            is_last = (i == total_steps - 1)
            ds_engine.set_gradient_accumulation_boundary(is_last)
            ds_engine.backward(step_loss_scaled)

            del logits, cur_logps_step, per_token_loss_step
            del per_token_adv_loss, ratio_step, is_clipped_step, per_token_entropy
            torch.cuda.empty_cache()

        # After loop, step once to update parameters
        ds_engine.step()

        return {
            "mean_loss": mean_loss_sum / total_steps,
            "mean_adv_loss": mean_adv_loss_sum / total_steps,
            "mean_clip": mean_clip_sum / total_steps,
            "mean_ratio": mean_ratio_sum / total_steps,
            "mean_entropy": mean_entropy_sum / total_steps,
            "mean_kl_loss": mean_kl_loss_sum / total_steps,
            "total_mask_sum": total_mask_sum,
        }


    def compute_loss(
        self,
        model: torch.nn.Module,
        ref_model: Optional[torch.nn.Module],
        rollout: RolloutOutput,
        main_schedule: MaskingSchedule,
        old_logps: torch.Tensor = None,
        accelerator: Optional[Accelerator] = None,
    ) -> dict:
        device = next(model.parameters()).device
        completion_mask = rollout.completion_mask.to(device)

        # Get ds_engine
        ds_engine = accelerator.deepspeed_engine_wrapped.engine

        # Reward & Advantage (timed)
        t_reward_start = time.time()
        rewards_per_func = compute_reward_per_func(
            completions=rollout.completions,
            ground_truths=rollout.ground_truths,
            reward_funcs=self.config.training.reward_funcs,
        )
        rewards = rewards_per_func.sum(dim=1).to(device)
        torch.cuda.synchronize()
        reward_time = time.time() - t_reward_start

        grouped = rewards.view(-1, self.num_generations)
        mean_r = grouped.mean(dim=1).repeat_interleave(self.num_generations)
        std_r = grouped.std(dim=1).repeat_interleave(self.num_generations)
        advantages = (rewards - mean_r) / (std_r + 1e-8).clamp(min=1e-2)

        # Common tensors
        prompt_ids = rollout.prompt_ids.to(device)
        completion_ids = rollout.completion_ids.to(device)
        full_seq = torch.cat([prompt_ids, completion_ids], dim=1)
        prompt_len = prompt_ids.shape[1]
        B, gen_len = completion_ids.shape

        if old_logps is not None:
            old_logps = old_logps.to(device)  # [total_steps, B, gen_len]
        else:
            total_steps = len(main_schedule.masking_inputs[0])
            old_logps = torch.zeros(total_steps, B, gen_len, device=device, dtype=torch.float32)

        # Schedule -> device
        sched = main_schedule.to(device)

        loss_metrics = self._compute_loss(
            model=model,
            ref_model=ref_model,
            sched=sched,
            full_seq=full_seq,
            prompt_len=prompt_len,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            old_logps=old_logps,
            advantages=advantages,
            ds_engine=ds_engine,
            B=B,
            device=device,
        )

        total_mask_sum = loss_metrics["total_mask_sum"]
        total_steps = len(sched.masking_inputs[0])

        loss_value = loss_metrics["mean_adv_loss"] + self.beta * loss_metrics["mean_kl_loss"]
        clip_ratio = loss_metrics["mean_clip"]
        mean_ratio = loss_metrics["mean_ratio"]
        mean_kl = loss_metrics["mean_kl_loss"] if ref_model is not None else 0.0
        mean_entropy = loss_metrics["mean_entropy"]
        effective_positions_ratio = total_mask_sum / completion_mask.sum().clamp(min=1).item() / total_steps

        result = {
            "loss": loss_value,
            "adv_loss": loss_metrics["mean_adv_loss"],
            "clip_ratio": clip_ratio,
            "mean_ratio": mean_ratio,
            "mean_kl": mean_kl,
            "mean_entropy": mean_entropy,
            "effective_positions_ratio": effective_positions_ratio,
            "rewards": rewards.detach(),
            "rewards_per_func": rewards_per_func.detach(),
            "advantages": advantages.detach(),
            "grad_norm": getattr(ds_engine, '_global_grad_norm', 0.0),
            "reward_time": reward_time,
        }

        loss_mode = "REINFORCE" if self.num_iterations == 1 else "PPO-clip"
        logger.info(f"Mean Reward: {rewards.mean().item():.4f} (loss_mode={loss_mode})")
        logger.info(
            f"  loss={result['loss']:.4f}  "
            f"adv_loss={result['adv_loss']:.4f}  "
            f"clip={result['clip_ratio']:.4f}  ratio={result['mean_ratio']:.4f}  "
            f"effective_pos={result['effective_positions_ratio']:.1%}  "
            f"kl={result['mean_kl']:.4f}  "
            f"entropy={result['mean_entropy']:.4f}  "
        )
        return result


# Evaluation
@torch.no_grad()
def run_evaluation(
    model: torch.nn.Module,
    accelerator: Accelerator,
    config,
    tokenizer,
    pretrained_model_path: str,
    global_step: int,
    epoch: int,
    jsonl_log_path: Path,
    mask_id: int,
    model_type: str = "llada",
    model_dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, float]:
    eval_cfg = config.evaluation
    raw_datasets = getattr(eval_cfg, "eval_dataset", "MATH500")
    if isinstance(raw_datasets, str):
        dataset_names = [raw_datasets]
    else:
        dataset_names = list(raw_datasets)

    raw_data_types = getattr(eval_cfg, "data_type", getattr(config.dataset, "data_type", "math"))
    if isinstance(raw_data_types, str):
        data_type_list = [raw_data_types] * len(dataset_names)
    else:
        data_type_list = list(raw_data_types)
        if len(data_type_list) != len(dataset_names):
            raise ValueError(
                f"evaluation.data_type length ({len(data_type_list)}) "
                f"must match evaluation.eval_dataset length ({len(dataset_names)})"
            )

    eval_batch_size = getattr(eval_cfg, "batch_size", 4)
    num_generations = 1

    eval_rollout_config = deepcopy(config)
    if hasattr(eval_rollout_config, "rollout"):
        eval_rollout_config.rollout.temperature = getattr(eval_cfg, "temperature", 0)
        eval_rollout_config.rollout.steps = getattr(eval_cfg, "steps", 512)
        eval_rollout_config.rollout.max_gen_length = getattr(eval_cfg, "max_gen_length", 512)
        eval_rollout_config.rollout.num_generations = num_generations
        for param_name in [
            "remasking_strategy", "target", "dynamic_threshold",
            "block_size", "further_horizon", "use_cache",
        ]:
            if hasattr(eval_cfg, param_name):
                setattr(eval_rollout_config.rollout, param_name, getattr(eval_cfg, param_name))

    rollout_model = load_rollout_model(model, accelerator, pretrained_model_path, model_type=model_type, model_dtype=model_dtype)
    torch.cuda.empty_cache()

    world_size = accelerator.num_processes
    rank = accelerator.process_index

    per_dataset_metrics = {}

    for ds_idx, eval_dataset_name in enumerate(dataset_names):
        eval_start_time = time.time()
        ds_data_type = data_type_list[ds_idx]

        eval_dataset_path = f"./data/{eval_dataset_name}.json"
        if not Path(eval_dataset_path).exists():
            logger.warning(f"Eval dataset not found: {eval_dataset_path}, skipping.")
            continue

        with open(eval_dataset_path, "r") as f:
            eval_data = json.load(f)

        eval_questions = [x["question"] for x in eval_data]
        if ds_data_type in ("code_stdio", "code_function"):
            default_method = "function" if ds_data_type == "code_function" else "stdio"
            eval_answers = []
            for x in eval_data:
                method = x.get("test_method", default_method)
                tc = {"test_method": method}
                if method == "stdio":
                    tc["test_input"] = x["test_input"]
                    tc["test_output"] = x["test_output"]
                elif method == "function":
                    tc["test_list"] = x["test_list"]
                    if "prefix" in x:
                        tc["prefix"] = x["prefix"]
                tc["test_time_limit"] = x.get("test_time_limit", 1)
                eval_answers.append(tc)
        else:
            eval_answers = [x["ground_truth_answer"] for x in eval_data]

        ds_prompt_template = Template(get_default_system_prompt(model_type, ds_data_type))
        eval_prompts = [ds_prompt_template.render(problem=q) for q in eval_questions]
        if ds_data_type == "code_function":
            for i, ans in enumerate(eval_answers):
                prefix = ans.get("prefix", "") if isinstance(ans, dict) else ""
                if prefix:
                    eval_prompts[i] = eval_prompts[i] + prefix

        logger.info(f"{MAGENTA}***** Running Evaluation (step={global_step}, epoch={epoch}) *****{RESET}")
        logger.info(f"  Eval dataset: {eval_dataset_name} ({len(eval_questions)} samples)")
        logger.info(f"  Batch size: {eval_batch_size}, Num generations: {num_generations}")
        logger.info(f"  Model type: {model_type}")
        log_gpu_memory("before eval rollout")

        total_samples = len(eval_prompts)
        per_rank = math.ceil(total_samples / world_size)
        shard_start = rank * per_rank
        shard_end = min(shard_start + per_rank, total_samples)
        local_prompts = eval_prompts[shard_start:shard_end]
        local_answers = eval_answers[shard_start:shard_end]
        local_total = len(local_prompts)

        logger.info(
            f"  Rank {rank}/{world_size}: eval shard [{shard_start}:{shard_end}] "
            f"({local_total} samples)"
        )

        all_rewards_per_func = []

        running_correct = 0
        running_total = 0

        num_eval_batches = math.ceil(local_total / eval_batch_size)
        max_eval_batches_tensor = torch.tensor([num_eval_batches], dtype=torch.long, device=accelerator.device)
        dist.all_reduce(max_eval_batches_tensor, op=dist.ReduceOp.MAX)
        max_eval_batches = max_eval_batches_tensor.item()

        eval_progress = tqdm(
            range(max_eval_batches),
            desc=f"Eval {eval_dataset_name} (step {global_step})",
            total=max_eval_batches,
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
            leave=True,
        )

        for batch_idx in eval_progress:
            start_idx = batch_idx * eval_batch_size
            end_idx = min(start_idx + eval_batch_size, local_total)

            if start_idx < local_total:
                batch_prompts = local_prompts[start_idx:end_idx]
                batch_answers = local_answers[start_idx:end_idx]

                rollout_output = run_rollout(
                    model=rollout_model,
                    tokenizer=tokenizer,
                    prompts=batch_prompts,
                    ground_truths=batch_answers,
                    config=eval_rollout_config,
                    model_type=model_type,
                    num_generations=num_generations,
                    batch_size=eval_batch_size,
                )

                batch_rewards_per_func = compute_reward_per_func(
                    completions=rollout_output.completions,
                    ground_truths=rollout_output.ground_truths,
                    reward_funcs=config.training.reward_funcs,
                )
                all_rewards_per_func.append(batch_rewards_per_func)

                batch_rewards = batch_rewards_per_func.sum(dim=1)
                running_correct += (batch_rewards == 1).sum().item()
                running_total += len(batch_rewards)

                del rollout_output
                torch.cuda.empty_cache()

            local_correct_tensor = torch.tensor([running_correct], dtype=torch.float32, device=accelerator.device)
            local_total_tensor = torch.tensor([running_total], dtype=torch.float32, device=accelerator.device)
            global_correct_running = local_correct_tensor.clone()
            global_total_running = local_total_tensor.clone()
            dist.all_reduce(global_correct_running, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_total_running, op=dist.ReduceOp.SUM)
            global_acc = (global_correct_running / global_total_running.clamp(min=1)).item()

            if accelerator.is_local_main_process:
                print(f"  Eval {eval_dataset_name} (step {global_step}): {batch_idx+1}/{max_eval_batches}, acc={global_acc:.1%}, samples={int(global_total_running.item())}/{total_samples}")

        eval_progress.close()
        dist.barrier()

        if len(all_rewards_per_func) == 0:
            # This rank has no samples; create an empty tensor
            num_funcs = len(config.training.reward_funcs) if config.training.reward_funcs else 1
            all_rewards_per_func = torch.zeros(0, num_funcs, dtype=torch.float32)
        else:
            all_rewards_per_func = torch.cat(all_rewards_per_func, dim=0)

        all_rewards = all_rewards_per_func.sum(dim=1)

        local_reward_sum = all_rewards.sum().to(accelerator.device)
        local_reward_sq_sum = (all_rewards ** 2).sum().to(accelerator.device)
        local_correct = (all_rewards == 1).float().sum().to(accelerator.device)
        local_count = torch.tensor(float(len(all_rewards)), device=accelerator.device)
        local_per_func_sum = all_rewards_per_func.sum(dim=0).to(accelerator.device)

        global_reward_sum = local_reward_sum.clone()
        global_reward_sq_sum = local_reward_sq_sum.clone()
        global_correct = local_correct.clone()
        global_count = local_count.clone()
        global_per_func_sum = local_per_func_sum.clone()

        dist.all_reduce(global_reward_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_reward_sq_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_per_func_sum, op=dist.ReduceOp.SUM)

        eval_reward_mean = (global_reward_sum / global_count.clamp(min=1)).item()
        eval_reward_std = (
            (global_reward_sq_sum / global_count.clamp(min=1) - eval_reward_mean ** 2).clamp(min=0).sqrt()
        ).item()
        eval_accuracy = (global_correct / global_count.clamp(min=1)).item()

        eval_per_func_means = {}
        reward_funcs = list(config.training.reward_funcs) if config.training.reward_funcs else []
        func_names = reward_funcs if reward_funcs else [f"func_{i}" for i in range(all_rewards_per_func.shape[1])]
        for fi, fname in enumerate(func_names):
            eval_per_func_means[fname] = (global_per_func_sum[fi] / global_count.clamp(min=1)).item()

        torch.cuda.synchronize()
        eval_total_time = time.time() - eval_start_time

        ds_metrics = {
            "eval_reward_mean": eval_reward_mean,
            "eval_reward_std": eval_reward_std,
            "eval_accuracy": eval_accuracy,
            "eval_num_samples": int(global_count.item()),
            "eval_reward_per_func": eval_per_func_means,
            "eval_total_time": eval_total_time,
        }
        per_dataset_metrics[eval_dataset_name] = ds_metrics

        # jsonl record is written once after the loop, not per-dataset
        logger.info(
            f"{MAGENTA}***** Evaluation Results (step={global_step}) *****{RESET}\n"
            f"  Dataset: {eval_dataset_name} ({int(global_count.item())} samples)\n"
            f"  Reward mean: {eval_reward_mean:.4f} ± {eval_reward_std:.4f}\n"
            f"  Accuracy: {eval_accuracy:.2%}\n"
            f"  Per-func: {eval_per_func_means}\n"
            f"  Eval total time: {eval_total_time:.2f}s"
        )

    # --- end of per-dataset loop ---

    del rollout_model
    gc.collect()
    torch.cuda.empty_cache()
    log_gpu_memory("after eval")

    if len(per_dataset_metrics) == 0:
        return {}

    # Compute aggregate metrics
    avg_accuracy = sum(m["eval_accuracy"] for m in per_dataset_metrics.values()) / len(per_dataset_metrics)
    avg_reward_mean = sum(m["eval_reward_mean"] for m in per_dataset_metrics.values()) / len(per_dataset_metrics)
    total_eval_time = sum(m["eval_total_time"] for m in per_dataset_metrics.values())
    total_num_samples = sum(m["eval_num_samples"] for m in per_dataset_metrics.values())

    # Write a single jsonl record
    if accelerator.is_main_process:
        log_entry = {
            "type": "evaluation",
            "global_step": global_step,
            "epoch": epoch,
            "datasets": list(per_dataset_metrics.keys()),
            "eval_accuracy": avg_accuracy,
            "eval_reward_mean": avg_reward_mean,
            "eval_num_samples": total_num_samples,
            "eval_total_time": total_eval_time,
            "per_dataset": per_dataset_metrics,
        }
        with open(jsonl_log_path, "a") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    # Log output
    if len(per_dataset_metrics) == 1:
        ds_name = list(per_dataset_metrics.keys())[0]
        m = per_dataset_metrics[ds_name]
        logger.info(
            f"{MAGENTA}***** Evaluation Summary (step={global_step}) *****{RESET}\n"
            f"  Dataset: {ds_name}\n"
            f"  Accuracy: {m['eval_accuracy']:.2%}\n"
            f"  Reward mean: {m['eval_reward_mean']:.4f}"
        )
    else:
        logger.info(
            f"{MAGENTA}***** Multi-dataset Evaluation Summary (step={global_step}) *****{RESET}\n"
            f"  Datasets: {list(per_dataset_metrics.keys())}\n"
            f"  Per-dataset accuracy: "
            + ", ".join(f"{ds}={m['eval_accuracy']:.2%}" for ds, m in per_dataset_metrics.items())
            + f"\n  Average accuracy: {avg_accuracy:.2%}"
        )

    # Return: single dataset returns its metrics; multi returns aggregate
    if len(per_dataset_metrics) == 1:
        return list(per_dataset_metrics.values())[0]

    combined_metrics = {
        "eval_accuracy": avg_accuracy,
        "eval_reward_mean": avg_reward_mean,
        "per_dataset": {
            ds: {"eval_accuracy": m["eval_accuracy"], "eval_reward_mean": m["eval_reward_mean"]}
            for ds, m in per_dataset_metrics.items()
        },
    }
    return combined_metrics


# Main
def main():
    config = get_config()

    project_name = config.experiment.project
    pretrained_model = config.model.pretrained_model

    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.project) / "logs")

    accelerate_configs_file = f'accelerate_configs/{config.experiment.deepspeed_file}.json'
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=accelerate_configs_file)

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=config.training.mixed_precision,
        deepspeed_plugin=deepspeed_plugin,
        log_with=None,
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    # if accelerator.is_local_main_process:
    #     set_verbosity_info()
    # else:
    #     set_verbosity_error()

    if accelerator.is_main_process:
        os.makedirs(config.experiment.project, exist_ok=True)
        config_path = Path(config.experiment.project) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    if config.training.seed is not None:
        set_seed(config.training.seed)

    model_dtype = resolve_model_dtype(config)
    logger.info(f"Model loading dtype: {model_dtype} (resolved from config)")

    # Get model type and load tokenizer
    model_type = get_model_type(config)
    logger.info(f"Loading tokenizer for model_type={model_type}")
    tokenizer, mask_id = load_tokenizer(pretrained_model, model_type)
    logger.info(f"mask_id = {mask_id}")

    # Load train and reference models
    logger.info(f"Loading {model_type} model from {pretrained_model}")
    model = load_train_model(pretrained_model, model_type, model_dtype)
    ref_model = load_ref_model(pretrained_model, model_type, model_dtype)

    kl_beta = getattr(config.training, "beta", 0.0)
    use_kl = kl_beta > 0
    kl_type = getattr(config.training, "kl_type", "k3")
    entropy_coeff = getattr(config.training, "entropy_coeff", 0.0)

    data_type = getattr(config.dataset, "data_type", "math")
    system_prompt_template = getattr(
        config, "system_prompt_template",
        get_default_system_prompt(model_type, data_type)
    )

    strategy_name = getattr(config.training, "mask_strategy", "random_masking")
    strategy_k = getattr(config.training, "mask_num", 1)
    logps_mask_ratio = getattr(config.training, "logps_mask_ratio", 0.3)
    sample_temperature = getattr(config.training, "sample_temperature", float("inf"))
    low_prob_threshold = getattr(config.training, "low_prob_threshold", 0.2)

    logger.info(f"Strategy: name={strategy_name}, k={strategy_k}")
    logger.info(f"KL type: {kl_type} (beta={kl_beta})")
    logger.info(f"Entropy coeff: {entropy_coeff}")
    logger.info(f"Low probability threshold: {low_prob_threshold}")

    optimizer_config = config.optimizer.params
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    logger.info("Creating dataset and dataloader")
    dataset_path = "./data/" + f"{config.dataset.train_dataset}.json"
    with open(dataset_path, "r") as f:
        dataset_load = json.load(f)

    questions = [x["question"] for x in dataset_load]
    # answers = [x["ground_truth_answer"] for x in dataset_load]
    
    # Support for code task data
    if config.dataset.data_type in ("code_stdio", "code_function"):
        # Infer default test_method from data_type
        default_method = "function" if config.dataset.data_type == "code_function" else "stdio"
        answers = []
        for x in dataset_load:
            method = x.get("test_method", default_method)
            tc = {"test_method": method}
            if method == "stdio":
                tc["test_input"] = x["test_input"]
                tc["test_output"] = x["test_output"]
            elif method == "function":
                tc["test_list"] = x["test_list"]
                if "prefix" in x:
                    tc["prefix"] = x["prefix"]
            tc["test_time_limit"] = x.get("test_time_limit", 1)
            answers.append(tc)
    else:
        answers = [x["ground_truth_answer"] for x in dataset_load]

    dataset = TrainDataset(questions, answers)

    num_generations = config.rollout.get("num_generations", 4)
    num_iterations = getattr(config.training, "num_iterations", 1)
    use_reinforce = (num_iterations == 1)

    num_train_epochs = config.training.num_train_epochs
    num_batches_per_epoch = math.ceil(len(dataset) / config.rollout.batch_size / accelerator.num_processes)
    max_train_steps = num_batches_per_epoch * num_train_epochs * num_iterations

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale,
    )

    train_dataloader = DataLoader(
        dataset, batch_size=config.rollout.batch_size,
        shuffle=True, collate_fn=simple_collate, num_workers=0, drop_last=True,
    )

    model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader
    )
    ref_model = prepare_deepspeed(ref_model, accelerator)

    drift_trainer = DriftTrainer(config, mask_id=mask_id, tokenizer=tokenizer, model_type=model_type)

    jsonl_log_path = Path(config.experiment.project) / "logging.jsonl"
    if accelerator.is_main_process:
        jsonl_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(jsonl_log_path, "a") as f:
            f.write(json.dumps({
                "_meta": "training_log",
                "project": project_name,
                "model_type": model_type,
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "num_processes": accelerator.num_processes,
                "strategy_name": strategy_name,
                "strategy_k": strategy_k,
                "kl_type": kl_type,
                "entropy_coeff": entropy_coeff,
                "loss_mode": "REINFORCE" if use_reinforce else "PPO-clip",
                "num_iterations": num_iterations,
                "model_dtype": str(model_dtype),
            }) + "\n")

    do_eval = hasattr(config, "evaluation") and config.evaluation is not None
    eval_every = None
    eval_at_start = False
    if do_eval:
        eval_every = getattr(config.evaluation, "eval_every_steps", None)
        eval_at_start = getattr(config.evaluation, "eval_at_start", False)
        save_after_eval = getattr(config.evaluation, "save_after_eval", False)
        logger.info(f"  Evaluation enabled: dataset={config.evaluation.eval_dataset}")
        if eval_every:
            logger.info(f"  Eval every {eval_every} steps")
        if eval_at_start:
            logger.info(f"  Eval at start (before training)")
        if save_after_eval:
            logger.info(f"  Save model after evaluation")

    logger.info(f"  Model type = {model_type}")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num epochs = {num_train_epochs}")
    logger.info(f"  Batch size per device = {config.training.batch_size_lm}")
    logger.info(f"  Rollout size per device = {config.rollout.batch_size}")
    logger.info(f"  Num generations per prompt = {num_generations}")
    logger.info(f"  Num iterations per rollout = {num_iterations}")
    logger.info(f"  Loss mode = {'REINFORCE' if use_reinforce else 'PPO-clip'}")
    logger.info(f"  Model dtype = {model_dtype}")
    logger.info(f"  Max train steps = {max_train_steps}")
    logger.info(f"  Strategy: {strategy_name}")
    logger.info(f"  KL: {kl_type} (beta={kl_beta}), Entropy coeff: {entropy_coeff}")
    logger.info(f" {GREEN}***** Start training *****{RESET}")

    prompt_template = Template(system_prompt_template)

    if do_eval and eval_at_start:
        model.eval()
        run_evaluation(
            model=model, accelerator=accelerator, config=config,
            tokenizer=tokenizer, pretrained_model_path=pretrained_model,
            global_step=0, epoch=0,
            jsonl_log_path=jsonl_log_path, mask_id=mask_id,
            model_type=model_type, model_dtype=model_dtype,
        )
        model.train()

    model.train()
    global_step = 1

    for epoch in range(num_train_epochs):
        progress_bar = tqdm(
            total=num_batches_per_epoch * num_iterations,
            desc=f"Epoch {epoch + 1}/{num_train_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True, leave=True,
        )

        for batch_idx, batch in enumerate(train_dataloader):
            logger.info(f"Batch {batch_idx}/{len(train_dataloader)}")
            step_total_start = time.time()
            questions_batch = batch["questions"]
            answers_batch = batch["answers"]
            prompts = [prompt_template.render(problem=q) for q in questions_batch]
            # code_function: append prefix to prompt for direct code output
            if config.dataset.data_type == "code_function":
                for i, ans in enumerate(answers_batch):
                    prefix = ans.get("prefix", "") if isinstance(ans, dict) else ""
                    if prefix:
                        prompts[i] = prompts[i] + prefix

            # Rollout phase (timed)
            logger.info(f"{GREEN}***** Rollout at global_step {global_step} *****{RESET}")
            log_gpu_memory("before rollout phase")

            rollout_start = time.time()

            rollout_model = load_rollout_model(
                model, accelerator, pretrained_model,
                model_type=model_type, model_dtype=model_dtype,
            )
            torch.cuda.empty_cache()

            with torch.no_grad():
                rollout_output = run_rollout(
                    model=rollout_model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    ground_truths=answers_batch,
                    config=config,
                    model_type=model_type,
                    num_generations=num_generations,
                    batch_size=getattr(config.rollout, "batch_size", 4),
                )

            local_num_samples = torch.tensor(
                [rollout_output.completion_ids.shape[0]], device=accelerator.device, dtype=torch.long
            )
            all_num_samples = accelerator.gather(local_num_samples)
            if not (all_num_samples == all_num_samples[0]).all():
                raise RuntimeError(
                    f"Rollout sample count mismatch across ranks: {all_num_samples.tolist()}. "
                    "All ranks must produce the same number of rollout samples."
                )
            logger.info(
                f"Rollout consistency check passed: all {accelerator.num_processes} ranks "
                f"produced {all_num_samples[0].item()} samples each."
            )

            del rollout_model
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()

            torch.cuda.synchronize()
            rollout_time = time.time() - rollout_start

            logger.info(f"{GREEN}***** Rollout Finished! ({rollout_time:.2f}s) *****{RESET}")
            log_gpu_memory("after rollout")

            max_groups = getattr(config.training, "batch_size_lm", 2)
            rollout_output, rollout_metrics = filter_rollout_samples(
                rollout_output=rollout_output,
                num_generations=num_generations,
                max_groups=max_groups,
                reward_funcs=config.training.reward_funcs,
            )
            logger.info(
                f"  After filtering: {rollout_output.completion_ids.shape[0]} samples "
                f"({rollout_output.completion_ids.shape[0] // num_generations} groups)"
            )

            main_schedule = build_masking_schedule(
                step_map=rollout_output.step_map,
                completion_mask=rollout_output.completion_mask,
                token_probs=rollout_output.token_probs,
                strategy=strategy_name,
                k=strategy_k,
                mask_ratio=logps_mask_ratio,
                num_generations=num_generations,
                sample_temperature=sample_temperature,
            )
            if dist.is_initialized():
                rank = dist.get_rank()
                local_num_steps = len(main_schedule.masking_inputs[0])
                steps_tensor = torch.tensor([local_num_steps], dtype=torch.long, device="cuda")
                all_steps = [torch.zeros(1, dtype=torch.long, device="cuda") for _ in range(dist.get_world_size())]
                dist.all_gather(all_steps, steps_tensor)
                all_steps_list = [t.item() for t in all_steps]
                logger.info(f"[Rank {rank}] schedule num_steps per rank: {all_steps_list}")
                if len(set(all_steps_list)) > 1:
                    logger.error(
                        f"[Rank {rank}] MISMATCH! num_steps across ranks: {all_steps_list}. "
                        f"This will cause NCCL deadlock."
                    )

            logger.info(f"Rollout steps: {(rollout_output.step_map.max(dim=1).values + 1).tolist()}")
            logger.info(f"Built masking schedule: {strategy_name}")

            # Analyze step counts and low-prob steps
            step_prob_metrics = analyze_rollout_step_probs(
                step_map=rollout_output.step_map,
                token_probs=rollout_output.token_probs,
                completion_mask=rollout_output.completion_mask,
                prob_threshold=getattr(config.rollout, "dynamic_threshold", 0.9),
            )
            logger.info(
                f"  {CYAN}[Rollout] avg_steps={step_prob_metrics['avg_steps_per_sample']:.1f}, "
                f"avg_low_prob_steps={step_prob_metrics['avg_low_prob_steps']:.2f}, "
                f"max_steps={step_prob_metrics['max_steps_per_sample']:.0f}, "
                f"min_steps={step_prob_metrics['min_steps_per_sample']:.0f}{RESET}"
            )

            # Compute old logps (PPO-clip mode)
            cached_old_logps = None
            if not use_reinforce:
                logger.info(f"{GREEN}***** Computing old logps (PPO-clip mode) *****{RESET}")
                cached_old_logps = compute_per_token_logps(
                    model=model,
                    prompt_ids=rollout_output.prompt_ids,
                    completion_ids=rollout_output.completion_ids,
                    completion_mask=rollout_output.completion_mask,
                    schedule=main_schedule,
                    mask_id=mask_id,
                    tokenizer=tokenizer,
                    model_type=model_type,
                ).cpu()
                torch.cuda.empty_cache()
                log_gpu_memory("after old logps")
            else:
                logger.info(
                    f"{CYAN}***** Skipping old logps computation (REINFORCE mode) *****{RESET}"
                )

            cached_rollout = rollout_output

            # backward + step already handled inside compute_loss via ds_engine
            # No need for accumulate / optimizer.step / scheduler.step / zero_grad here
            for step in range(num_iterations):
                loss_mode_str = "REINFORCE" if use_reinforce else "PPO-clip"
                logger.info(
                    f"{GREEN}***** Computing loss [{loss_mode_str}] "
                    f"(iter {step+1}/{num_iterations}) *****{RESET}"
                )
                log_gpu_memory("before compute loss")

                train_start = time.time()

                result = drift_trainer.compute_loss(
                    model=model,
                    ref_model=ref_model if use_kl else None,
                    rollout=cached_rollout,
                    main_schedule=main_schedule,
                    old_logps=cached_old_logps,
                    accelerator=accelerator,
                )
                # Update learning rate
                lr_scheduler.step()

                torch.cuda.synchronize()
                train_time = time.time() - train_start

                loss_val = (
                    result["loss"].item()
                    if isinstance(result["loss"], torch.Tensor)
                    else result["loss"]
                )

                # grad_norm from compute_loss result (via ds_engine._global_grad_norm)
                grad_norm = result.get("grad_norm", 0.0)
                if isinstance(grad_norm, torch.Tensor):
                    grad_norm = grad_norm.item()

                # reward_time from compute_loss result
                reward_time = result.get("reward_time", 0.0)

                torch.cuda.synchronize()
                step_total_time = time.time() - step_total_start

                # Aggregate timing info
                timing_info = {
                    "step_total_time": round(step_total_time, 2),
                    "rollout_time": round(rollout_time, 2),
                    "train_time": round(train_time, 2),
                    "reward_time": round(reward_time, 2),
                }

                logger.info(
                    f"{CYAN}[TIMING] Step {global_step}: "
                    f"total={step_total_time:.2f}s, "
                    f"rollout={rollout_time:.2f}s, "
                    f"train={train_time:.2f}s, "
                    f"reward={reward_time:.2f}s{RESET}"
                )

                # Log and evaluate
                logger.info(f"{GREEN} ***** Optimizer Step Finished *****{RESET}")
                logger.info(f"{YELLOW}***** Global Step {global_step} Finished *****{RESET}")

                log_step_to_jsonl(
                    log_path=jsonl_log_path,
                    global_step=global_step,
                    epoch=epoch + 1,
                    lr=optimizer.param_groups[0]["lr"],
                    grad_norm=grad_norm,
                    result=result,
                    strategy_key=strategy_name,
                    reward_funcs=config.training.reward_funcs,
                    accelerator=accelerator,
                    rollout_metrics=rollout_metrics,
                    timing=timing_info,
                    step_prob_metrics=step_prob_metrics,
                )

                if do_eval and eval_every and global_step % eval_every == 0:
                    model.eval()
                    eval_metrics = run_evaluation(
                        model=model, accelerator=accelerator, config=config,
                        tokenizer=tokenizer, pretrained_model_path=pretrained_model,
                        global_step=global_step,
                        epoch=epoch + 1, jsonl_log_path=jsonl_log_path, mask_id=mask_id,
                        model_type=model_type, model_dtype=model_dtype,
                    )
                    model.train()
                    if save_after_eval:
                        save_checkpoint_top_k(
                            model=model,
                            tokenizer=tokenizer,
                            config=config,
                            accelerator=accelerator,
                            global_step=global_step,
                            eval_metrics=eval_metrics,
                            top_k=getattr(config.evaluation, "top_k_checkpoints", 2),
                            metric_key=getattr(config.evaluation, "top_k_metric", "eval_accuracy"),
                        )

                progress_bar.update(1)
                progress_bar.set_postfix(
                    loss=f"{loss_val:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    step=global_step,
                )
                torch.cuda.empty_cache()

                global_step += 1

    accelerator.wait_for_everyone()
    accelerator.end_training()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()