import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from jinja2 import Template
from transformers import AutoTokenizer

from rollout.llada_generate import generate_with_prefix_cache, denoise_step_map

rank = dist.get_rank() if dist.is_initialized() else 0


@dataclass
class RolloutOutput:
    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    completions: List[str]
    ground_truths: List[str]
    step_map: torch.Tensor
    token_probs: torch.Tensor          # (N, gen_length)

    def to(self, device: torch.device) -> "RolloutOutput":
        return RolloutOutput(
            prompt_ids=self.prompt_ids.to(device),
            prompt_mask=self.prompt_mask.to(device),
            completion_ids=self.completion_ids.to(device),
            completion_mask=self.completion_mask.to(device),
            completions=self.completions,
            ground_truths=self.ground_truths,
            step_map=self.step_map.to(device),
            token_probs=self.token_probs.to(device),
        )


# Core generation function

def remap_step_map_continuous(step_map: torch.Tensor, completion_mask: torch.Tensor) -> torch.Tensor:
    """Remap step_map to continuous 0, 1, 2, ... per sample.

    Invalid positions (completion_mask == 0) stay 0.
    """
    completion_mask = completion_mask.to(step_map.device)
    remapped = torch.zeros_like(step_map)
    for i in range(step_map.shape[0]):
        mask = completion_mask[i].bool()
        if not mask.any():
            continue
        vals = step_map[i][mask]
        unique_sorted, inverse = vals.unique(sorted=True, return_inverse=True)

        remapped[i][mask] = inverse
    return remapped


@torch.no_grad()
def _generate_batch(
    model,
    tokenizer,
    prompt_texts: List[str],
    ground_truths: List[str],
    config,
    mask_id: int,
    pad_id: int,
    device: torch.device,
) -> RolloutOutput:
    """Generate a batch on a single device and return RolloutOutput.

    Model should already be on device in eval mode.
    """
    # Ensure eval mode
    was_training = model.training
    model.eval()

    # Ensure model config allows use_cache
    want_cache = getattr(config.rollout, "use_cache", False)
    old_model_use_cache = getattr(model.config, "use_cache", None)
    if want_cache and old_model_use_cache is not None:
        model.config.use_cache = True

    # ---- Tokenize ----
    enc = tokenizer(
        prompt_texts,
        padding=True,
        return_tensors="pt",
        padding_side="left",
    )
    input_ids = enc["input_ids"].to(device)          # (B, L_prompt_padded)
    attention_mask = enc["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    # ---- Rollout params ----
    further_horizon = None
    if hasattr(config.rollout, "use_cache") and config.rollout.use_cache:
        further_horizon = getattr(config.rollout, "further_horizon", None)

    unmask_threshold = None
    if getattr(config.rollout, "remasking_strategy", "") != "low_confidence_static":
        unmask_threshold = getattr(config.rollout, "dynamic_threshold", None)

    # ---- Generate ----
    out = generate_with_prefix_cache(
        model,
        input_ids,
        steps=config.rollout.steps,
        gen_length=config.rollout.max_gen_length,
        block_length=config.rollout.block_size,
        temperature=config.rollout.temperature,
        target=config.rollout.target,
        mask_id=mask_id,
        further_horizon=further_horizon,
        use_cache=getattr(config.rollout, "use_cache", False),
        unmask_threshold=unmask_threshold,
    )

    generated_seqs = out.sequences  # (B, L_prompt + gen_length)
    gen_length = config.rollout.max_gen_length
    remaining_masks = (generated_seqs[:, prompt_len:] == mask_id).sum(dim=1)
    token_probs = out.token_probs[:, prompt_len:]

    # ---- Split prompt / completion ----
    completion_ids = generated_seqs[:, prompt_len:]  # (B, gen_length)
    completion_mask = (
        (completion_ids != pad_id) & (completion_ids != mask_id)
    ).long()

    # ---- Compute step_map ----
    B = input_ids.shape[0]
    step_map_list = []
    for i in range(B):
        smap = denoise_step_map(out.history, mask_id=mask_id, sample_idx=i)
        smap = smap[prompt_len:]  # (gen_length,)
        step_map_list.append(smap)
    step_map = torch.stack(step_map_list, dim=0)  # (B, gen_length)
    # ---- Remap step_map to continuous indices ----
    step_map = remap_step_map_continuous(step_map, completion_mask)

    # ---- Decode text ----
    completion_texts = tokenizer.batch_decode(
        completion_ids.cpu().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    # Restore original training state and config
    if want_cache and old_model_use_cache is not None:
        model.config.use_cache = old_model_use_cache
    if was_training:
        model.train()

    return RolloutOutput(
        prompt_ids=input_ids.cpu(),
        prompt_mask=attention_mask.cpu(),
        completion_ids=completion_ids.cpu(),
        completion_mask=completion_mask.cpu(),
        completions=completion_texts,
        ground_truths=ground_truths,
        step_map=step_map.cpu(),
        token_probs=token_probs.float().cpu(),
    )


# Entry point: llada_rollout

@torch.no_grad()
def llada_rollout(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    ground_truths: List[str],
    config,
    num_generations: int = 4,
    batch_size: int = 4,
) -> RolloutOutput:
    """Sample rollout completions for a list of prompts.

    Uses the passed-in model (already prepared by accelerate).
    Distribution is managed by accelerate, not here.

    Returns RolloutOutput with all generated results and metadata.
    """
    device = next(model.parameters()).device
    mask_id = tokenizer.encode("<|mdm_mask|>")[0]
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    
    # ---- 1. Expand: repeat each prompt num_generations times ----
    expanded_prompts: List[str] = []
    expanded_gts: List[str] = []
    for prompt, gt in zip(prompts, ground_truths):
        expanded_prompts.extend([prompt] * num_generations)
        expanded_gts.extend([gt] * num_generations)

    N = len(expanded_prompts)
    num_batches = math.ceil(N / batch_size)

    # ---- 2. Generate in batches ----
    model.eval()
    all_outputs: List[RolloutOutput] = []

    from tqdm.auto import tqdm
    pbar = tqdm(
        range(0, N, batch_size),
        desc="Rollout",
        total=num_batches,
        dynamic_ncols=True,
        leave=True,
    )

    for start in pbar:
        end = min(start + batch_size, N)
        batch_prompts = expanded_prompts[start:end]
        batch_gts = expanded_gts[start:end]

        batch_output = _generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=batch_prompts,
            ground_truths=batch_gts,
            config=config,
            mask_id=mask_id,
            pad_id=pad_id,
            device=device,
        )

        all_outputs.append(batch_output)
        pbar.set_postfix(samples=f"{end}/{N}")

    pbar.close()

    # Restore original mode
    model.train()

    # ---- 3. Merge all batches ----
    max_prompt_len = max(output.prompt_ids.shape[1] for output in all_outputs)
    max_gen_len = max(output.completion_ids.shape[1] for output in all_outputs)
    # print("Max Prompt Length: ", max_prompt_len)
    # print("Max Completion Length: ", max_gen_len)


    all_prompt_ids = []
    all_prompt_mask = []
    all_completion_ids = []
    all_completion_mask = []
    all_completions = []
    all_ground_truths = []
    all_step_map = []
    all_token_probs = []

    for output in all_outputs:
        B, L = output.prompt_ids.shape
        if L < max_prompt_len:
            # left-pad prompt
            pad_len = max_prompt_len - L
            all_prompt_ids.append(
                F.pad(output.prompt_ids, (pad_len, 0), value=pad_id)
            )
            all_prompt_mask.append(
                F.pad(output.prompt_mask, (pad_len, 0), value=0)
            )
        else:
            all_prompt_ids.append(output.prompt_ids)
            all_prompt_mask.append(output.prompt_mask)

        all_completion_ids.append(output.completion_ids)
        all_completion_mask.append(output.completion_mask)
        all_completions.extend(output.completions)
        all_ground_truths.extend(output.ground_truths)
        all_step_map.append(output.step_map)
        all_token_probs.append(output.token_probs)
    
    return RolloutOutput(
        prompt_ids=torch.cat(all_prompt_ids, dim=0),          # (N, max_prompt_len)
        prompt_mask=torch.cat(all_prompt_mask, dim=0),        # (N, max_prompt_len)
        completion_ids=torch.cat(all_completion_ids, dim=0),  # (N, gen_length)
        completion_mask=torch.cat(all_completion_mask, dim=0),# (N, gen_length)
        completions=all_completions,                           # list[str], len=N
        ground_truths=all_ground_truths,                       # list[str], len=N
        step_map=torch.cat(all_step_map, dim=0),              # (N, gen_length)
        token_probs=torch.cat(all_token_probs, dim=0),
    )