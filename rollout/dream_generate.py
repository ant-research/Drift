from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.distributions as dists
import torch.nn.functional as F
from transformers.utils import ModelOutput

from rollout.dream.generation_utils_block import DreamGenerationConfig


# Output data structure

@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[List[torch.Tensor]] = None
    token_probs: Optional[torch.Tensor] = None  # (B, max_length)


# Logits filtering

def top_p_logits(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus sampling: keep tokens with cumulative probability <= top_p."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Keep only the top_k highest-probability tokens."""
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


# Sampling

def sample_tokens(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    tar: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample tokens from logits. Returns (target_score, x0).
    Aligned with source sample_tokens.
    """
    logits = logits.float()

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)

    dist = dists.Categorical(logits=logits)
    x0 = dist.sample()
    probs = dist.probs

    if temperature > 0:
        target = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    else:
        target, x0 = probs.max(dim=-1)

    if tar == "confidence":
        return target, x0

    if tar == "margin_confidence":
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top1_probs = sorted_probs[..., 0]
        top2_probs = sorted_probs[..., 1]
        target = top1_probs - top2_probs

    if tar == "neg_entropy":
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        target = torch.sum(probs * log_probs, dim=-1)

    return target, x0


# Diffusion decoding loop

@torch.no_grad()
def dream_generate(
    model,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor],
    generation_config: DreamGenerationConfig,
    block_length: Optional[int] = 32,
    use_cache: bool = False,
    further_horizon: int = 128,
    mask_token_id: int = 151666,
    eos_token_id: int = 151645,
    pad_token_id: int = 151643,
    pad_target_penalty: float = 1.0,
    unmask_threshold: Optional[float] = None,
) -> DreamModelOutput:
    """
    Dream diffusion decoding (aligned with source).

    Key alignment points: logits shifting, first-step assignment,
    static timestep schedule, dynamic threshold, and 4D attention mask slicing.
    """
    # Parse config
    output_history = generation_config.output_history
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    steps = generation_config.steps
    temperature = generation_config.temperature
    top_p = generation_config.top_p
    top_k = generation_config.top_k
    tar = generation_config.tar   # e.g. 'confidence'
    alg_temp = generation_config.alg_temp
    cgws = further_horizon

    histories = [] if (return_dict_in_generate and output_history) else None

    B, L0 = input_ids.shape
    gen_length = max_length - L0

    # Initialize sequence: prompt + mask
    x = torch.full((B, max_length), mask_token_id, dtype=torch.long, device=input_ids.device)
    x[:, :L0] = input_ids

    # Per-position probability when unmasked (reserved for RL extension)
    token_probs = torch.zeros(B, max_length, dtype=torch.float64, device=x.device)

    # Block configuration
    if block_length is None:
        block_length = gen_length

    assert gen_length % block_length == 0, (
        f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
    )
    num_blocks = gen_length // block_length

    # Steps per block allocation
    base, rem = divmod(steps, num_blocks)
    steps_per_block = [base + (1 if i < rem else 0) for i in range(num_blocks)]

    # Linear timestep schedule per block
    timesteps = [
        torch.linspace(1, generation_config.eps, spb + 1, device=x.device)
        for spb in steps_per_block
    ]

    # 4D attention mask
    if attention_mask is not None and torch.any(attention_mask == 0.0):
        attention_mask = F.pad(
            attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0
        )
        tok_idx = attention_mask.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attention_mask == 0, 1)
        attention_mask = torch.logical_and(
            attention_mask.unsqueeze(1).unsqueeze(-2),
            attention_mask.unsqueeze(1).unsqueeze(-1),
        )
        attention_mask = torch.where(
            attention_mask,
            torch.tensor(0.0, device=attention_mask.device),
            torch.tensor(float("-inf"), device=attention_mask.device),
        )
    else:
        tok_idx = None
        attention_mask = "full"

    past_key_values = None

    # Per-block decoding
    for blk in range(num_blocks):
        s = L0 + blk * block_length        # block start
        e = L0 + (blk + 1) * block_length  # block end

        if cgws is not None:
            window_end = min(e + cgws, max_length)
            window_slice = slice(s, window_end)

        spb = steps_per_block[blk]

        # First forward pass
        # Full-sequence forward, then assign only the block's first token.
        if use_cache:
            model_output = model(x, attention_mask, tok_idx, use_cache=True)
            past_key_values = model_output.past_key_values
            # Truncate KV cache to block start
            new_pkv = []
            for layer_i in range(len(past_key_values)):
                new_pkv.append(())
                for tensor_j in range(len(past_key_values[layer_i])):
                    kv_tensor = past_key_values[layer_i][tensor_j]
                    if kv_tensor.dim() == 4:
                        # (batch, heads, seq_len, head_dim)
                        new_pkv[layer_i] += (kv_tensor[:, :, :s, :],)
                    elif kv_tensor.dim() == 3:
                        # (batch, seq_len, hidden_dim) - Dream format
                        new_pkv[layer_i] += (kv_tensor[:, :s, :],)
                    else:
                        raise ValueError(
                            f"Unexpected KV cache tensor dim={kv_tensor.dim()}, "
                            f"shape={kv_tensor.shape}. Expected 3D or 4D."
                        )
            past_key_values = new_pkv
        else:
            model_output = model(x, attention_mask, tok_idx, use_cache=False)

        logits = model_output.logits
        # Logits shift (next-token -> current-token alignment)
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        # Mask illegal tokens: mask token and ghost IDs (from lm_head padding)
        logits[:, :, mask_token_id] = float('-inf')
        if logits.size(-1) > mask_token_id + 1:
            logits[:, :, mask_token_id + 1:] = float('-inf')

        # First step: sample and assign only the block's first token
        _, x0_first = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
        x[:, s] = x0_first[:, s]

        if histories is not None:
            histories.append(x.clone().cpu())

        # Iterative unmask loop
        i = 1
        while True:
            # Build mask_index
            if cgws is not None:
                mask_index = (x[:, window_slice] == mask_token_id)
            else:
                mask_index = (x[:, s:] == mask_token_id)

            # Attention mask slicing
            if attention_mask != "full":
                if cgws is not None:
                    current_attention_mask = attention_mask[:, :, window_slice, :window_end]
                else:
                    current_attention_mask = attention_mask[:, :, s:, :]
            else:
                current_attention_mask = attention_mask

            # Forward pass
            if use_cache:
                if cgws is not None:
                    model_output = model(
                        x[:, window_slice], current_attention_mask,
                        tok_idx[:, window_slice] if tok_idx is not None else None,
                        past_key_values=past_key_values, use_cache=True,
                    )
                else:
                    model_output = model(
                        x[:, s:], current_attention_mask,
                        tok_idx[:, s:] if tok_idx is not None else None,
                        past_key_values=past_key_values, use_cache=True,
                    )
                logits = model_output.logits
                # Logits shift
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            else:
                model_output = model(x, attention_mask, tok_idx, use_cache=False)
                logits = model_output.logits
                logits = logits[:, s:]
                # Logits shift
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            # Check if block has no remaining masks before sampling
            if (x[:, s:e] == mask_token_id).sum() == 0:
                break

            # Sample only masked positions within the block
            mask_index[:, block_length:] = False
            mask_logits = logits[mask_index]
            # Mask illegal tokens: mask token and ghost IDs
            mask_logits[:, mask_token_id] = float('-inf')
            if mask_logits.size(-1) > mask_token_id + 1:
                mask_logits[:, mask_token_id + 1:] = float('-inf')
            target_scores, x0 = sample_tokens(
                mask_logits, temperature, top_p=top_p, top_k=top_k, tar=tar
            )

            # Pad token penalty (Dream-specific)
            if pad_target_penalty != 1.0:
                _pad_mask_flat = (x0 == pad_token_id)
                if _pad_mask_flat.any():
                    target_scores = target_scores.clone()
                    target_scores[_pad_mask_flat] = target_scores[_pad_mask_flat] / pad_target_penalty

            # Build full_target scores
            if cgws is not None:
                full_target = torch.full_like(
                    x[:, window_slice], -torch.inf,
                    device=x.device, dtype=logits.dtype
                )
            else:
                full_target = torch.full_like(
                    x[:, s:], -torch.inf,
                    device=x.device, dtype=logits.dtype
                )
            full_target = full_target.float()
            full_target[mask_index] = target_scores
            full_target[:, block_length:] = -torch.inf

            # Static schedule
            # Uses linear timestep schedule when unmask_threshold is None.
            if unmask_threshold is None:
                num_mask_token = mask_index.sum() / mask_index.shape[0]
                t = timesteps[blk][i]
                s_time = timesteps[blk][i + 1]
                number_transfer_tokens = (
                    int(num_mask_token * (1 - s_time / t))
                    if i < spb - 1
                    else int(num_mask_token)
                )

                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(full_target, number_transfer_tokens)
                    else:
                        ft = full_target / alg_temp
                        ft = F.softmax(ft, dim=-1)
                        transfer_index = torch.multinomial(ft, num_samples=number_transfer_tokens)

                    # Build candidate token sequence
                    if cgws is not None:
                        x_ = torch.zeros_like(
                            x[:, window_slice], device=x.device, dtype=torch.long
                        ) + mask_token_id
                    else:
                        x_ = torch.zeros_like(
                            x[:, s:], device=x.device, dtype=torch.long
                        ) + mask_token_id
                    x_[mask_index] = x0.clone()

                    row_indices = torch.arange(
                        x.size(0), device=x.device
                    ).unsqueeze(1).expand_as(transfer_index)

                    if cgws is not None:
                        x[:, window_slice][row_indices, transfer_index] = x_[row_indices, transfer_index]
                    else:
                        x[:, s:][row_indices, transfer_index] = x_[row_indices, transfer_index]

            # Dynamic threshold
            else:
                if cgws is not None:
                    xwin = x[:, window_slice]
                else:
                    xwin = x[:, s:]

                selected_map = torch.zeros_like(xwin, dtype=torch.bool)
                selected_map[mask_index] = (target_scores >= unmask_threshold)
                no_sel = ~selected_map.any(dim=-1)
                no_sel = no_sel & mask_index.any(dim=-1)

                if no_sel.any():
                    masked_scores = full_target.masked_fill(~mask_index, float("-inf"))
                    best_idx = torch.argmax(masked_scores, dim=-1)
                    selected_rows = torch.nonzero(no_sel, as_tuple=False).squeeze(-1)
                    selected_map[selected_rows, best_idx[selected_rows]] = True

                selected_map &= mask_index
                x_candidates = torch.full_like(xwin, mask_token_id, dtype=torch.long)
                x_candidates[mask_index] = x0

                xwin[selected_map] = x_candidates[selected_map]

            if histories is not None:
                histories.append(x.clone().cpu())

            i += 1

            if (x[:, s:e] == mask_token_id).sum() == 0:
                break

        # Early termination: block is all pad
        block_all_pad = torch.all(x[:, s:e] == pad_token_id)
        if block_all_pad:
            if e < x.size(1):
                x[:, e:] = pad_token_id
            if histories is not None:
                histories.append(x.clone().cpu())
            break

    return DreamModelOutput(
        sequences=x,
        history=histories,
        token_probs=token_probs,
    )


# Extract step_map from history

def denoise_step_map(history, mask_id: int, sample_idx: int = 0):
    """
    Walk history snapshots and record the step at which each position
    first transitions from MASK to non-MASK.
    """
    L = history[0].shape[1]
    step_map = torch.zeros(L, dtype=torch.long)
    prev = torch.full((L,), mask_id, dtype=torch.long)

    for t, snap in enumerate(history, start=0):
        cur = snap[sample_idx]
        changed = (prev == mask_id) & (cur != mask_id)
        step_map[changed] = t
        prev = cur
        if (step_map == 0).sum() == 0:
            break
    return step_map
