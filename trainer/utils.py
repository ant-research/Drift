import json
import shutil
import time
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist
from accelerate.logging import get_logger
from omegaconf import OmegaConf

logger = get_logger(__name__, log_level="INFO")

# config utils
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


# checkpoint utils
def _copy_model_python_files(save_path: Path, model_type: str, pretrained_model: str = None):
    if model_type not in ("dream", "llada"):
        return

    source_dir = None

    if pretrained_model:
        model_path = Path(pretrained_model)
        if model_path.exists():
            source_dir = model_path
        else:
            logger.warning(f"Source model directory not found: {source_dir}")
            return

    files_to_copy = []
    if model_type == "dream":
        files_to_copy = [
            "modeling_dream.py",
            "configuration_dream.py",
            "tokenization_dream.py",
            "generation_utils.py",
        ]
    elif model_type == "llada":
        files_to_copy = [
            "modeling_llada.py",
            "configuration_llada.py",
        ]

    for filename in files_to_copy:
        src = source_dir / filename
        dst = save_path / filename
        if src.exists():
            shutil.copy2(src, dst)
            logger.debug(f"Copied {filename} to {save_path}")
        else:
            logger.warning(f"Source file not found: {src}")


def save_checkpoint(model, tokenizer, config, accelerator, name):
    output_dir = Path(config.experiment.project)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = getattr(config.experiment, "checkpoint_dir", None)
    if checkpoint_dir:
        save_base = Path(checkpoint_dir)
    else:
        save_base = output_dir / "ckpt"
    save_base.mkdir(parents=True, exist_ok=True)

    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    if accelerator.is_main_process and checkpoints_total_limit is not None:
        ckpts = sorted(
            [d for d in save_base.iterdir() if d.name.startswith("checkpoint")],
            key=lambda p: int(p.name.split("-")[1]),
        )
        if len(ckpts) >= checkpoints_total_limit:
            to_remove = ckpts[: len(ckpts) - checkpoints_total_limit + 1]
            logger.info(f"removing checkpoints: {', '.join(p.name for p in to_remove)}")
            for p in to_remove:
                shutil.rmtree(p, ignore_errors=True)

    state_dict = accelerator.get_state_dict(model)
    model_to_save = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        model_to_save.save_pretrained(
            save_base / name,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(str(save_base / name))

        model_type = getattr(config.model, "model_type", "llada").lower()
        pretrained_model = getattr(config.model, "pretrained_model", None)
        _copy_model_python_files(save_base / name, model_type, pretrained_model)

        metadata = {
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with (save_base / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model + tokenizer to {save_base / name}")


def save_checkpoint_top_k(
    model,
    tokenizer,
    config,
    accelerator,
    global_step: int,
    eval_metrics: Dict[str, float],
    top_k: int = 3,
    metric_key: str = "eval_accuracy",
):
    """Save checkpoint if current eval result is in top-k, with rolling deletion."""
    checkpoint_dir = getattr(config.experiment, "checkpoint_dir", None)
    if checkpoint_dir:
        output_dir = Path(checkpoint_dir)
    else:
        output_dir = Path(config.experiment.project) / "ckpt"
    output_dir.mkdir(parents=True, exist_ok=True)

    topk_meta_path = output_dir / "topk_meta.json"

    if accelerator.is_main_process:
        if topk_meta_path.exists():
            try:
                with open(topk_meta_path, "r") as f:
                    topk_entries = json.load(f)
            except (json.JSONDecodeError, ValueError):
                logger.warning("topk_meta.json is corrupted, resetting to empty list.")
                topk_entries = []
        else:
            topk_entries = []

        current_score = eval_metrics.get(metric_key, 0.0)

        should_save = (
            len(topk_entries) < top_k
            or current_score > min(e["score"] for e in topk_entries)
        )
    else:
        should_save = False

    should_save_tensor = torch.tensor(
        [1 if should_save else 0], dtype=torch.long, device=accelerator.device
    )
    dist.broadcast(should_save_tensor, src=0)
    should_save = should_save_tensor.item() == 1

    if not should_save:
        if accelerator.is_main_process:
            logger.info(
                f"Step {global_step}: {metric_key}={eval_metrics.get(metric_key, 0.0):.4f} "
                f"not in top-{top_k}, skipping save."
            )
        return

    ckpt_name = f"checkpoint-{global_step}"
    state_dict = accelerator.get_state_dict(model)
    model_to_save = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        save_path = output_dir / ckpt_name
        model_to_save.save_pretrained(
            save_path,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(str(save_path))

        model_type = getattr(config.model, "model_type", "llada").lower()
        pretrained_model = getattr(config.model, "pretrained_model", None)
        _copy_model_python_files(save_path, model_type, pretrained_model)

        topk_entries.append({
            "step": global_step,
            "score": current_score,
            "ckpt_name": ckpt_name,
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": eval_metrics,
        })

        topk_entries.sort(key=lambda e: e["score"], reverse=True)

        if len(topk_entries) > top_k:
            to_remove = topk_entries[top_k:]
            topk_entries = topk_entries[:top_k]
            for entry in to_remove:
                rm_path = output_dir / entry["ckpt_name"]
                if rm_path.exists():
                    shutil.rmtree(rm_path, ignore_errors=True)
                    logger.info(
                        f"Removed checkpoint {entry['ckpt_name']} "
                        f"({metric_key}={entry['score']:.4f})"
                    )

        with open(topk_meta_path, "w") as f:
            json.dump(topk_entries, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Saved top-k checkpoint: {ckpt_name} ({metric_key}={current_score:.4f}). "
            f"Current top-{top_k}: "
            + ", ".join(f"{e['ckpt_name']}({e['score']:.4f})" for e in topk_entries)
        )

    dist.barrier()


# misc
def log_gpu_memory(tag: str = ""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(
            f"[GPU Memory{' - ' + tag if tag else ''}] "
            f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
        )
