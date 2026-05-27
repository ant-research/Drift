import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

from rollout.dream import DreamTokenizer
from rollout.dream.generation_utils_block import DreamGenerationConfig
from rollout.dream_generate import dream_generate, denoise_step_map


# RolloutOutput data contract

@dataclass
class RolloutOutput:
    prompt_ids: torch.Tensor       # (N, max_prompt_len)
    prompt_mask: torch.Tensor      # (N, max_prompt_len)
    completion_ids: torch.Tensor   # (N, gen_length)
    completion_mask: torch.Tensor  # (N, gen_length)
    completions: List[str]         # len = N
    ground_truths: List[str]       # len = N
    step_map: torch.Tensor         # (N, gen_length)
    token_probs: torch.Tensor      # (N, gen_length)

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


# Step map utilities

def remap_step_map_continuous(
    step_map: torch.Tensor, completion_mask: torch.Tensor
) -> torch.Tensor:
    """
    Remap each sample's step_map to contiguous values 0, 1, 2, ...
    Invalid positions (completion_mask == 0) remain 0.
    """
    completion_mask = completion_mask.to(step_map.device)
    remapped = torch.zeros_like(step_map)
    for i in range(step_map.shape[0]):
        mask = completion_mask[i].bool()
        if not mask.any():
            continue
        vals = step_map[i][mask]
        _, inverse = vals.unique(sorted=True, return_inverse=True)
        remapped[i][mask] = inverse
    return remapped




# Single-device batch generation

@torch.no_grad()
def _generate_batch(
    model,
    tokenizer,
    prompt_texts: List[str],
    ground_truths: List[str],
    config,
    mask_id: int,
    pad_id: int,
    eos_id: int,
    device: torch.device,
) -> RolloutOutput:
    """
    Generate completions for a batch of prompts on a single device.
    Model should already be on device and in eval mode.
    """
    was_training = model.training
    model.eval()

    # Tokenize
    enc = tokenizer(
        prompt_texts,
        padding=True,
        return_tensors="pt",
        padding_side="left",
    )
    input_ids = enc["input_ids"].to(device)
    prompt_len = input_ids.shape[1]
    attn_mask = input_ids.ne(pad_id).to(device)

    # Rollout params
    use_cache = getattr(config.rollout, "use_cache", False)

    # further_horizon: only enabled when use_cache=True
    further_horizon = None
    if use_cache:
        further_horizon = getattr(config.rollout, "further_horizon", None)

    # unmask_threshold: matches source logic
    unmask_threshold = None
    remasking_strategy = getattr(config.rollout, "remasking_strategy", "")
    if isinstance(remasking_strategy, (list, tuple)):
        remasking_strategy = remasking_strategy[0] if remasking_strategy else ""

    if remasking_strategy == "low_confidence_static":
        unmask_threshold = None
    else:
        unmask_threshold = getattr(config.rollout, "dynamic_threshold", None)

    # Target scoring strategy
    target = config.rollout.target

    # Build generation config
    generation_config = DreamGenerationConfig(
        output_history=True,
        return_dict_in_generate=True,
        max_length=config.rollout.max_gen_length + prompt_len,
        steps=config.rollout.steps,
        temperature=config.rollout.temperature,
        top_p=getattr(config.rollout, "top_p", None),
        top_k=getattr(config.rollout, "top_k", None),
        tar=target,
        alg_temp=getattr(config.rollout, "alg_temp", None),
    )

    # Run sampling
    out = dream_generate(
        model,
        input_ids,
        attention_mask=attn_mask,
        generation_config=generation_config,
        block_length=config.rollout.block_size,
        use_cache=use_cache,
        further_horizon=further_horizon,
        mask_token_id=mask_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        pad_target_penalty=getattr(config.rollout, "pad_target_penalty", 1.0),
        unmask_threshold=unmask_threshold,
    )

    generated_seqs = out.sequences       # (B, prompt_len + gen_length)
    token_probs = out.token_probs[:, prompt_len:]  # (B, gen_length)

    # Split prompt / completion
    completion_ids = generated_seqs[:, prompt_len:]
    completion_mask = (
        (completion_ids != pad_id) & (completion_ids != mask_id)
    ).long()

    # Compute step_map
    B = input_ids.shape[0]
    step_map_list = []
    for i in range(B):
        smap = denoise_step_map(out.history, mask_id=mask_id, sample_idx=i)
        smap = smap[prompt_len:]
        step_map_list.append(smap)
    step_map = torch.stack(step_map_list, dim=0)
    step_map = remap_step_map_continuous(step_map, completion_mask)

    # Decode text
    completion_texts = tokenizer.batch_decode(
        completion_ids.cpu().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    if was_training:
        model.train()

    return RolloutOutput(
        prompt_ids=input_ids.cpu(),
        prompt_mask=enc["attention_mask"].cpu(),
        completion_ids=completion_ids.cpu(),
        completion_mask=completion_mask.cpu(),
        completions=completion_texts,
        ground_truths=ground_truths,
        step_map=step_map.cpu(),
        token_probs=token_probs.float().cpu(),
    )


# Entry point: dream_rollout

@torch.no_grad()
def dream_rollout(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    ground_truths: List[str],
    config,
    num_generations: int = 4,
    batch_size: int = 4,
) -> RolloutOutput:
    """RL rollout entry point for Dream models."""
    device = next(model.parameters()).device
    mask_id = model.config.mask_token_id
    pad_id = model.config.pad_token_id
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # Expand prompts for num_generations
    expanded_prompts: List[str] = []
    expanded_gts: List[str] = []
    for prompt, gt in zip(prompts, ground_truths):
        expanded_prompts.extend([prompt] * num_generations)
        expanded_gts.extend([gt] * num_generations)

    N = len(expanded_prompts)
    num_batches = math.ceil(N / batch_size)

    # Generate in batches
    model.eval()
    all_outputs: List[RolloutOutput] = []

    from tqdm.auto import tqdm
    pbar = tqdm(
        range(0, N, batch_size),
        desc="Dream Rollout",
        total=num_batches,
        dynamic_ncols=True,
        leave=True,
    )

    for start in pbar:
        end = min(start + batch_size, N)
        batch_output = _generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=expanded_prompts[start:end],
            ground_truths=expanded_gts[start:end],
            config=config,
            mask_id=mask_id,
            pad_id=pad_id,
            eos_id=eos_id,
            device=device,
        )
        all_outputs.append(batch_output)
        pbar.set_postfix(samples=f"{end}/{N}")

    pbar.close()
    model.train()

    # Merge all batches
    max_prompt_len = max(o.prompt_ids.shape[1] for o in all_outputs)

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
            pad_len = max_prompt_len - L
            all_prompt_ids.append(F.pad(output.prompt_ids, (pad_len, 0), value=pad_id))
            all_prompt_mask.append(F.pad(output.prompt_mask, (pad_len, 0), value=0))
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
        prompt_ids=torch.cat(all_prompt_ids, dim=0),
        prompt_mask=torch.cat(all_prompt_mask, dim=0),
        completion_ids=torch.cat(all_completion_ids, dim=0),
        completion_mask=torch.cat(all_completion_mask, dim=0),
        completions=all_completions,
        ground_truths=all_ground_truths,
        step_map=torch.cat(all_step_map, dim=0),
        token_probs=torch.cat(all_token_probs, dim=0),
    )
