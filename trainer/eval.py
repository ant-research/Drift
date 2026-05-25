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

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import ListConfig, OmegaConf
from tqdm.auto import tqdm

from reward.rl_reward import compute_reward_per_func
from rollout.dream import DreamModel as RolloutDreamModel
from rollout.llada import LLaDAModelLM as RolloutLLaDAModelLM
from trainer.main_rl import (
    get_model_type,
    load_tokenizer,
    run_rollout,
    get_default_system_prompt,
    resolve_model_dtype,
)
from trainer.utils import get_config, log_gpu_memory

logger = get_logger(__name__, log_level="INFO")

MAGENTA = "\033[95m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


# Helpers

def ensure_list(val):
    """Wrap a scalar or OmegaConf ListConfig into a plain Python list."""
    if val is None:
        return []
    if isinstance(val, (list, ListConfig)):
        return list(val)
    return [val]


def load_eval_data(dataset_name, data_type, prompt_template):
    """Load a dataset and return (prompts, answers)."""
    dataset_path = f"./data/{dataset_name}.json"
    if not Path(dataset_path).exists():
        logger.warning(f"Dataset not found: {dataset_path}, skipping.")
        return None, None

    with open(dataset_path, "r") as f:
        eval_data = json.load(f)

    questions = [x["question"] for x in eval_data]

    if data_type in ("code_stdio", "code_function"):
        default_method = "function" if data_type == "code_function" else "stdio"
        answers = []
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
            answers.append(tc)
    else:
        answers = [x["ground_truth_answer"] for x in eval_data]

    prompts = [prompt_template.render(problem=q) for q in questions]

    # code_function: append prefix to prompt
    if data_type == "code_function":
        for i, ans in enumerate(answers):
            prefix = ans.get("prefix", "") if isinstance(ans, dict) else ""
            if prefix:
                prompts[i] = prompts[i] + prefix

    return prompts, answers


def load_rollout_model(pretrained_model_path, model_type, model_dtype):
    """Directly load rollout model from disk (no DeepSpeed needed for eval)."""
    if model_type == "llada":
        model = RolloutLLaDAModelLM.from_pretrained(
            pretrained_model_path, torch_dtype=model_dtype
        )
    else:
        model = RolloutDreamModel.from_pretrained(
            pretrained_model_path, torch_dtype=model_dtype
        )
    return model.to("cuda").eval()


def run_single_eval(
    accelerator,
    config,
    tokenizer,
    pretrained_model_path,
    prompt_template,
    model_type,
    model_dtype,
    mask_id,
    dataset_name,
    data_type,
    eval_steps,
    eval_max_gen_length,
    eval_batch_size,
    reward_funcs,
):
    """
    Run evaluation for one (model, dataset, gen_length) combo.
    Returns a dict with accuracy, reward stats, etc.
    """
    eval_cfg = config.evaluation

    prompts, answers = load_eval_data(dataset_name, data_type, prompt_template)
    if prompts is None:
        return None

    total_samples = len(prompts)
    num_generations = 1

    logger.info(f"{MAGENTA}{'='*60}{RESET}")
    logger.info(
        f"{MAGENTA}Eval: model={Path(pretrained_model_path).name}, "
        f"dataset={dataset_name}, steps={eval_steps}, "
        f"max_gen_length={eval_max_gen_length}{RESET}"
    )
    logger.info(f"  Samples: {total_samples}, batch_size: {eval_batch_size}")

    # Shard across ranks
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    per_rank = math.ceil(total_samples / world_size)
    shard_start = rank * per_rank
    shard_end = min(shard_start + per_rank, total_samples)
    local_prompts = prompts[shard_start:shard_end]
    local_answers = answers[shard_start:shard_end]
    local_total = len(local_prompts)

    logger.info(f"  Rank {rank}/{world_size}: shard [{shard_start}:{shard_end}] ({local_total} samples)")

    # Build rollout config from evaluation section
    # (eval.yaml may not have a "rollout" key, so we always create it)
    rollout_dict = {
        "temperature": getattr(eval_cfg, "temperature", 0),
        "steps": eval_steps,
        "max_gen_length": eval_max_gen_length,
        "num_generations": num_generations,
    }
    for param_name in [
        "remasking_strategy", "target", "dynamic_threshold",
        "block_size", "further_horizon", "use_cache",
    ]:
        if hasattr(eval_cfg, param_name):
            val = getattr(eval_cfg, param_name)
            if isinstance(val, (list, ListConfig)):
                val = val[0]
            rollout_dict[param_name] = val

    eval_rollout_config = deepcopy(config)
    eval_rollout_config.rollout = OmegaConf.create(rollout_dict)

    # Load rollout model directly from disk (no DeepSpeed needed for eval)
    rollout_model = load_rollout_model(
        pretrained_model_path, model_type, model_dtype,
    )
    torch.cuda.empty_cache()

    # Batched inference
    running_correct = 0
    running_total = 0
    all_rewards_per_func = []

    num_eval_batches = math.ceil(local_total / eval_batch_size)
    max_eval_batches_tensor = torch.tensor([num_eval_batches], dtype=torch.long, device=accelerator.device)
    dist.all_reduce(max_eval_batches_tensor, op=dist.ReduceOp.MAX)
    max_eval_batches = max_eval_batches_tensor.item()

    eval_start_time = time.time()

    for batch_idx in range(max_eval_batches):
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
                reward_funcs=reward_funcs,
            )
            all_rewards_per_func.append(batch_rewards_per_func)

            batch_rewards = batch_rewards_per_func.sum(dim=1)
            running_correct += (batch_rewards == 1).sum().item()
            running_total += len(batch_rewards)

            del rollout_output
            torch.cuda.empty_cache()

        # Running global accuracy
        local_correct_t = torch.tensor([running_correct], dtype=torch.float32, device=accelerator.device)
        local_total_t = torch.tensor([running_total], dtype=torch.float32, device=accelerator.device)
        global_correct_t = local_correct_t.clone()
        global_total_t = local_total_t.clone()
        dist.all_reduce(global_correct_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_total_t, op=dist.ReduceOp.SUM)
        global_acc = (global_correct_t / global_total_t.clamp(min=1)).item()

        if accelerator.is_local_main_process:
            print(
                f"  [{dataset_name} steps={eval_steps}] "
                f"batch {batch_idx+1}/{max_eval_batches}, "
                f"acc={global_acc:.1%}, "
                f"samples={int(global_total_t.item())}/{total_samples}"
            )

    dist.barrier()

    # Cleanup rollout model
    del rollout_model
    gc.collect()
    torch.cuda.empty_cache()

    # Aggregate metrics
    all_rewards_per_func = torch.cat(all_rewards_per_func, dim=0)
    all_rewards = all_rewards_per_func.sum(dim=1)

    local_reward_sum = all_rewards.sum().to(accelerator.device)
    local_correct = (all_rewards == 1).float().sum().to(accelerator.device)
    local_count = torch.tensor(float(len(all_rewards)), device=accelerator.device)
    local_per_func_sum = all_rewards_per_func.sum(dim=0).to(accelerator.device)

    global_reward_sum = local_reward_sum.clone()
    global_correct = local_correct.clone()
    global_count = local_count.clone()
    global_per_func_sum = local_per_func_sum.clone()

    dist.all_reduce(global_reward_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(global_correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(global_per_func_sum, op=dist.ReduceOp.SUM)

    eval_reward_mean = (global_reward_sum / global_count.clamp(min=1)).item()
    eval_accuracy = (global_correct / global_count.clamp(min=1)).item()
    eval_time = time.time() - eval_start_time

    func_names = list(reward_funcs) if reward_funcs else [
        f"func_{i}" for i in range(all_rewards_per_func.shape[1])
    ]
    per_func_means = {}
    for fi, fname in enumerate(func_names):
        per_func_means[fname] = (global_per_func_sum[fi] / global_count.clamp(min=1)).item()

    result = {
        "model": pretrained_model_path,
        "dataset": dataset_name,
        "steps": eval_steps,
        "max_gen_length": eval_max_gen_length,
        "accuracy": eval_accuracy,
        "reward_mean": eval_reward_mean,
        "num_samples": int(global_count.item()),
        "reward_per_func": per_func_means,
        "eval_time": round(eval_time, 2),
    }

    logger.info(
        f"{MAGENTA}Result: dataset={dataset_name}, steps={eval_steps}, "
        f"acc={eval_accuracy:.2%}, reward={eval_reward_mean:.4f}, "
        f"time={eval_time:.1f}s{RESET}"
    )

    return result


# Main

def main():
    config = get_config()

    project_name = config.experiment.project

    if getattr(config.training, "enable_tf32", False) if hasattr(config, "training") else False:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(project_name) / "logs")

    # No DeepSpeed needed for eval-only: each GPU loads a full copy of the
    # rollout model directly from disk (same as main_rl.py's run_evaluation
    # which loads rollout_model onto each GPU independently).
    accelerator = Accelerator(
        mixed_precision=getattr(config.training, "mixed_precision", "bf16") if hasattr(config, "training") else "bf16",
        log_with=None,
        project_dir=config.experiment.logging_dir,
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
        os.makedirs(project_name, exist_ok=True)

    seed = None
    if hasattr(config, "training") and hasattr(config.training, "seed"):
        seed = config.training.seed
    if seed is not None:
        set_seed(seed)

    model_dtype = resolve_model_dtype(config) if hasattr(config, "training") else torch.bfloat16
    model_type = get_model_type(config)

    eval_cfg = config.evaluation
    raw_data_types = getattr(eval_cfg, "data_type", "math")
    if isinstance(raw_data_types, str):
        data_type_list = [raw_data_types]
    else:
        data_type_list = list(raw_data_types)
    eval_batch_size = getattr(eval_cfg, "batch_size", 4)
    reward_funcs = getattr(config.training, "reward_funcs", ["accuracy"]) if hasattr(config, "training") else ["accuracy"]

    # Normalize list configs
    model_paths = ensure_list(config.model.pretrained_model)
    datasets = ensure_list(eval_cfg.eval_dataset)

    # Validate data_type_list length matches datasets
    if len(data_type_list) != len(datasets):
        logger.warning(
            f"data_type length ({len(data_type_list)}) != datasets length ({len(datasets)}). "
            f"Using first data_type for all datasets."
        )
        data_type_list = [data_type_list[0]] * len(datasets) if data_type_list else ["math"]
    steps_list = ensure_list(getattr(eval_cfg, "steps", 512))
    gen_len_list = ensure_list(getattr(eval_cfg, "max_gen_length", 512))

    assert len(steps_list) == len(gen_len_list), (
        f"steps ({len(steps_list)}) and max_gen_length ({len(gen_len_list)}) "
        f"must have the same length (paired 1:1)"
    )

    # Build system prompt template
    system_prompt_template = getattr(
        config, "system_prompt_template",
        get_default_system_prompt(model_type, data_type_list[0]),
    )
    from jinja2 import Template
    prompt_template = Template(system_prompt_template)

    total_combos = len(model_paths) * len(datasets) * len(steps_list)
    logger.info(f"{BOLD}Batch Evaluation: {total_combos} combos{RESET}")
    logger.info(f"  Models ({len(model_paths)}): {model_paths}")
    logger.info(f"  Datasets ({len(datasets)}): {datasets}")
    logger.info(f"  Steps/GenLen ({len(steps_list)}): {list(zip(steps_list, gen_len_list))}")

    all_results = []
    jsonl_path = Path(project_name) / "batch_eval_results.jsonl"

    combo_idx = 0

    for model_path in model_paths:
        logger.info(f"\n{GREEN}{'='*60}{RESET}")
        logger.info(f"{GREEN}Loading model: {model_path}{RESET}")
        logger.info(f"{GREEN}{'='*60}{RESET}")

        tokenizer, mask_id = load_tokenizer(model_path, model_type)

        for ds_idx, dataset_name in enumerate(datasets):
            ds_data_type = data_type_list[ds_idx]
            for step_val, gen_len_val in zip(steps_list, gen_len_list):
                combo_idx += 1
                logger.info(
                    f"\n{CYAN}[Combo {combo_idx}/{total_combos}] "
                    f"model={Path(model_path).name}, dataset={dataset_name}, "
                    f"steps={step_val}, max_gen_length={gen_len_val}{RESET}"
                )

                result = run_single_eval(
                    accelerator=accelerator,
                    config=config,
                    tokenizer=tokenizer,
                    pretrained_model_path=model_path,
                    prompt_template=prompt_template,
                    model_type=model_type,
                    model_dtype=model_dtype,
                    mask_id=mask_id,
                    dataset_name=dataset_name,
                    data_type=ds_data_type,
                    eval_steps=step_val,
                    eval_max_gen_length=gen_len_val,
                    eval_batch_size=eval_batch_size,
                    reward_funcs=reward_funcs,
                )

                if result is not None:
                    all_results.append(result)
                    if accelerator.is_main_process:
                        with open(jsonl_path, "a") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")

                dist.barrier()

    # Print summary table
    if accelerator.is_main_process and all_results:
        print(f"\n{'='*100}")
        print(f"{'BATCH EVALUATION SUMMARY':^100}")
        print(f"{'='*100}")
        header = f"{'Model':<40} {'Dataset':<15} {'Steps':>6} {'GenLen':>6} {'Accuracy':>10} {'Reward':>8} {'Time':>8}"
        print(header)
        print("-" * 100)
        for r in all_results:
            model_short = Path(r["model"]).name
            if len(model_short) > 38:
                model_short = "..." + model_short[-35:]
            print(
                f"{model_short:<40} {r['dataset']:<15} {r['steps']:>6} "
                f"{r['max_gen_length']:>6} {r['accuracy']:>9.2%} "
                f"{r['reward_mean']:>8.4f} {r['eval_time']:>7.1f}s"
            )
        print(f"{'='*100}")

        # Save full summary JSON
        summary_path = Path(project_name) / "batch_eval_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {jsonl_path}")
        print(f"Summary saved to: {summary_path}")

    accelerator.end_training()
    logger.info("Batch evaluation complete!")


if __name__ == "__main__":
    main()
