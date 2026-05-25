import json
import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from jinja2 import Template
from omegaconf import DictConfig, ListConfig, OmegaConf
from termcolor import cprint
from transformers import AutoModel, AutoTokenizer

from rollout.llada.modeling_llada import LLaDAModelLM


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    noise = (- torch.log(noise)) ** temperature
    return logits.exp() / noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


# Return type
@dataclass
class DiffusionOutput:
    sequences:   torch.Tensor
    history:     List[torch.Tensor]
    nfe:         int
    token_probs: torch.Tensor


@torch.no_grad()
def generate_with_prefix_cache(
        model, prompt,
        steps, gen_length, block_length, temperature,
        target, mask_id, further_horizon, use_cache, unmask_threshold
    ) -> DiffusionOutput:

    cgws = further_horizon
    B, L0 = prompt.shape
    x = torch.full((B, L0 + gen_length), mask_id, dtype=torch.long, device=prompt.device)
    max_length = L0 + gen_length
    x[:, :L0] = prompt
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    base, rem = divmod(steps, num_blocks)
    steps_per_block = [base + (i < rem) for i in range(num_blocks)]

    nfe = 0
    hist: List[torch.Tensor] = []

    # Track probability of each token when unmasked
    token_probs = torch.zeros(B, L0 + gen_length, dtype=torch.float64, device=prompt.device)

    for blk in range(num_blocks):
        s, e = L0 + blk * block_length, L0 + (blk + 1) * block_length

        if cgws is not None:
            window_end = max_length if cgws is None else min(e + cgws, max_length)
            window_slice = slice(s, window_end)

        cur_steps = steps_per_block[blk]
        num_transfer = get_num_transfer_tokens((x[:, s:e] == mask_id), cur_steps)

        # first full forward
        if use_cache:
            out = model(x, use_cache=True)
            pkv = out.past_key_values
            new_pkv = tuple(
                tuple(t[:, :, :s] for t in layer) for layer in pkv
            )
            pkv = new_pkv
        else:
            out = model(x, use_cache=False)

        mask_all = (x == mask_id)
        mask_all[:, e:] = 0

        x0, tr_idx, tp = get_transfer_index(
            out.logits, temperature, target, mask_all,
            x, num_transfer[:, 0], unmask_threshold)
        # Record unmasked token probabilities
        token_probs[tr_idx] = tp[tr_idx]
        x[tr_idx] = x0[tr_idx]
        hist.append(x.clone().cpu())
        nfe += 1

        i = 1
        while True:
            nfe += 1
            if cgws is not None:
                mask_blk = (x[:, window_slice] == mask_id)
            else:
                mask_blk = (x[:, s:] == mask_id)
            mask_blk[:, block_length:] = 0

            if use_cache:
                if cgws is not None:
                    logits = model(x[:, window_slice], past_key_values=pkv, use_cache=True).logits
                    x0, tr_idx, tp = get_transfer_index(
                        logits, temperature, target,
                        mask_blk, x[:, window_slice], num_transfer[:, i], unmask_threshold)
                    # Map window-relative tr_idx back to global indices
                    token_probs[:, window_slice][tr_idx] = tp[tr_idx]
                    x[:, window_slice][tr_idx] = x0[tr_idx]
                else:
                    logits = model(x[:, s:], past_key_values=pkv, use_cache=True).logits
                    x0, tr_idx, tp = get_transfer_index(
                        logits, temperature, target,
                        mask_blk, x[:, s:], num_transfer[:, i], unmask_threshold)
                    token_probs[:, s:][tr_idx] = tp[tr_idx]
                    x[:, s:][tr_idx] = x0[tr_idx]
            else:
                logits = model(x, use_cache=False).logits
                logits = logits[:, s:]
                x0, tr_idx, tp = get_transfer_index(
                    logits, temperature, target,
                    mask_blk, x[:, s:], num_transfer[:, i], unmask_threshold)
                token_probs[:, s:][tr_idx] = tp[tr_idx]
                x[:, s:][tr_idx] = x0[tr_idx]

            hist.append(x.clone().cpu())

            if (x[:, s:e] == mask_id).sum() == 0:
                break
            i += 1

    return DiffusionOutput(sequences=x, history=hist, nfe=nfe, token_probs=token_probs)


def get_transfer_index(logits, temperature, target, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    # Compute softmax probabilities for token prob logging
    p = F.softmax(logits.to(torch.float64), dim=-1)
    # True probability of selected token (without noise)
    token_prob = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (B, L)

    if target == 'confidence':
        x0_p = token_prob
    elif target == 'margin_confidence':
        top2 = torch.topk(p, 2, dim=-1).values
        x0_p = top2[..., 0] - top2[..., 1]
    elif target == 'neg_entropy':
        x0_p = -torch.sum(p * torch.log(p + 1e-10), dim=-1)
    elif target == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(target)

    x0 = torch.where(mask_index, x0, x)

    if threshold is not None:
        selected = mask_index & (x0_p >= threshold)
        has_mask = mask_index.any(dim=-1)
        none_sel = (~selected.any(dim=-1)) & has_mask
        if none_sel.any():
            masked_scores = x0_p.masked_fill(~mask_index, float("-inf"))
            best_idx = masked_scores.argmax(dim=-1)
            rows = torch.nonzero(none_sel, as_tuple=False).squeeze(-1)
            selected[rows, best_idx[rows]] = True
        return x0, selected, token_prob

    confidence = x0_p.masked_fill(~mask_index, float("-inf"))
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    for j in range(confidence.shape[0]):
        k = int(num_transfer_tokens[j].item() if torch.is_tensor(num_transfer_tokens[j]) else num_transfer_tokens[j])
        if k <= 0:
            continue
        _, sel = torch.topk(confidence[j], k=k)
        transfer_index[j, sel] = True
    return x0, transfer_index, token_prob  


import random 
def random_select(data_list, random_k):
    data_list = random.sample(data_list, random_k)
    return data_list


# obtain prompt
def get_prompt(data_i):
    return Template(system_prompts).render(problem = data_i["question"])


# def extract_final_boxed_answer(s: str):
#     tag = r'\boxed{'
#     start = s.rfind(tag)          # last \boxed{
#     if start == -1:
#         return "Can not extract the answer!"

#     i = start + len(tag)
#     depth = 1                    # we are already inside one '{'
#     buf = []

#     while i < len(s) and depth:
#         ch = s[i]
#         if ch == '{':
#             depth += 1
#         elif ch == '}':
#             depth -= 1
#             if depth == 0:       # matching '}' for the opening \boxed{
#                 break
#         buf.append(ch)
#         i += 1

#     return ''.join(buf) if depth == 0 else "Can not extract the answer!"


def denoise_step_map(history, mask_id: int, sample_idx: int = 0):
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


from tqdm import tqdm


def worker(pretrained_model, rank, prompts, orig_idx, seq_dict, step_dict, batch_size, config):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # load model once
    model_gpu = (LLaDAModelLM
                 .from_pretrained(pretrained_model,
                                  trust_remote_code=True,
                                  torch_dtype=torch.bfloat16)
                 .to(device)
                 .eval())

    tokenizer_gpu = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)

    # process in chunks of `batch_size`
    for start in tqdm(range(0, len(prompts), batch_size),
                      desc=f"GPU {rank}", position=rank, leave=True):
        batch_prompts = prompts[start:start+batch_size]
        batch_idxs    = orig_idx[start:start+batch_size]

        # tokenize & move to GPU
        enc = tokenizer_gpu(batch_prompts,
                            padding=True, #truncation=True,
                            return_tensors="pt", padding_side="left")
        input_ids = enc["input_ids"].to(device)

        mask_id = tokenizer_gpu.encode('<|mdm_mask|>')[0]

        if config.rollout.use_cache == False:
            config.rollout.further_horizon = None
        
        if config.rollout.remasking_strategy == "low_confidence_static":
            unmask_threshold = None
        else:
            unmask_threshold = config.rollout.dynamic_threshold

        # generate_with_prefix_cache
        out = generate_with_prefix_cache(
            model_gpu, input_ids,
            steps=config.rollout.steps, gen_length=config.rollout.max_gen_length,
            block_length=config.rollout.block_size, temperature=config.rollout.temperature,
            target=config.rollout.target, mask_id=mask_id, further_horizon=config.rollout.further_horizon,
            use_cache=config.rollout.use_cache, unmask_threshold = unmask_threshold
        )

        out.sequences = out.sequences.cpu()
        torch.cuda.empty_cache()

        # decode
        seq_ids = out.sequences[:, input_ids.shape[1]:].tolist()
        texts  = tokenizer_gpu.batch_decode(
            seq_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        
        # compute and store step maps
        for i, idx in enumerate(batch_idxs):
            # extract step map for sample i in this batch
            m = denoise_step_map(out.history, mask_id=mask_id, sample_idx=i)
            step_map = m[input_ids.shape[1]:].tolist()
            seq_dict[idx]  = texts[i]
            step_dict[idx] = step_map

        # free unused GPU cache
        torch.cuda.empty_cache()


def get_data_chunk(data, num_node, node_idx):
    total = len(data)
    chunk_size = (total + num_node - 1) // num_node 
    start_idx = node_idx * chunk_size
    end_idx = min((node_idx + 1) * chunk_size, total)
    return data[start_idx:end_idx]


def extract_code(full_output):
    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    if matches:
        code_output = matches[-1].strip()
    else:
        code_output = "We can not extract the code in the output. "
    return code_output