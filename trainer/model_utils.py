import gc
import os
from copy import deepcopy

import torch
from accelerate.logging import get_logger
from transformers import AutoConfig, AutoTokenizer

from models.dream import DreamModel
from models.llada import LLaDAModelLM
from rollout.dream import DreamModel as RolloutDreamModel
from rollout.llada import LLaDAModelLM as RolloutLLaDAModelLM

logger = get_logger(__name__, log_level="INFO")


# Model type dispatch

def get_model_type(config) -> str:
    """Get model type from config, defaults to 'llada'."""
    model_type = getattr(config.model, "model_type", "llada").lower()
    assert model_type in ("llada", "dream"), \
        f"Unsupported model_type='{model_type}'. Must be 'llada' or 'dream'."
    return model_type


def load_train_model(pretrained_model_path: str, model_type: str, model_dtype: torch.dtype):
    """Load training model (will be wrapped by accelerate.prepare)."""
    if model_type == "llada":
        return LLaDAModelLM.from_pretrained(pretrained_model_path, torch_dtype=model_dtype)
    else:
        return DreamModel.from_pretrained(pretrained_model_path, torch_dtype=model_dtype)


def load_ref_model(pretrained_model_path: str, model_type: str, model_dtype: torch.dtype):
    """Load reference model (for KL constraint)."""
    if model_type == "llada":
        return LLaDAModelLM.from_pretrained(pretrained_model_path, torch_dtype=model_dtype)
    else:
        return DreamModel.from_pretrained(pretrained_model_path, torch_dtype=model_dtype)


def load_tokenizer(pretrained_model_path: str, model_type: str):
    """Load tokenizer."""
    if model_type == "llada":
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        mask_id = tokenizer.encode("<|mdm_mask|>")[0]
        return tokenizer, mask_id
    else:
        from rollout.dream import DreamTokenizer
        tokenizer = DreamTokenizer.from_pretrained(pretrained_model_path, trust_remote_code=True)
        # Dream mask_id is read from model config
        from transformers import AutoConfig
        dream_config = AutoConfig.from_pretrained(pretrained_model_path, trust_remote_code=True)
        mask_id = dream_config.mask_token_id
        return tokenizer, mask_id

# Precision helper

def resolve_model_dtype(config) -> torch.dtype:
    _DTYPE_MAP = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "no": torch.float32,
    }
    explicit = getattr(config.training, "model_dtype", None)
    if explicit is not None:
        dtype = _DTYPE_MAP.get(str(explicit).lower())
        if dtype is None:
            raise ValueError(
                f"Unknown config.training.model_dtype='{explicit}'. "
                f"Supported: {list(_DTYPE_MAP.keys())}"
            )
        return dtype
    mp = getattr(config.training, "mixed_precision", None)
    if mp is not None:
        dtype = _DTYPE_MAP.get(str(mp).lower())
        if dtype is not None:
            return dtype
    return torch.bfloat16

# ZeRO-3 helpers

def prepare_deepspeed(model, accelerator, deepspeed_config=None, deepspeed_plugin=None, training_args=None):
    try:
        import deepspeed
        from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig
        from accelerate.utils import DeepSpeedPlugin
    except ImportError:
        pass

    if deepspeed_config is not None:
        if isinstance(deepspeed_config, dict):
            hf_ds_config = HfTrainerDeepSpeedConfig(deepspeed_config)
            if training_args is not None:
                hf_ds_config.trainer_config_process(training_args)
            temp_plugin = DeepSpeedPlugin(hf_ds_config=hf_ds_config)
            config_kwargs = deepcopy(temp_plugin.deepspeed_config)
        else:
            raise ValueError(f'deepspeed_config should be a dict, got {type(deepspeed_config)}')
    elif deepspeed_plugin is not None:
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
    else:
        deepspeed_plugin = accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

    stage = config_kwargs['zero_optimization']['stage']

    if model is not None:
        hidden_size = (
            max(model.config.hidden_sizes) if getattr(model.config, 'hidden_sizes', None) else getattr(
                model.config, 'hidden_size', None))
        if hidden_size is not None and stage == 3:
            config_kwargs.update({
                'zero_optimization.reduce_bucket_size': hidden_size * hidden_size,
                'zero_optimization.stage3_param_persistence_threshold': 10 * hidden_size,
                'zero_optimization.stage3_prefetch_bucket_size': 0.9 * hidden_size * hidden_size,
            })

    if stage != 3:
        config_kwargs['zero_optimization']['stage'] = 0

    env_vars_to_clear = [
        'DEEPSPEED_ZERO_STAGE',
        'DEEPSPEED_CONFIG',
        'DEEPSPEED_CONFIG_FILE',
    ]
    saved_env = {}
    for env_var in env_vars_to_clear:
        if env_var in os.environ:
            saved_env[env_var] = os.environ[env_var]
            del os.environ[env_var]

    try:
        model, *_ = deepspeed.initialize(args=None, model=model, config=config_kwargs)
        model.eval()
    finally:
        for env_var, value in saved_env.items():
            os.environ[env_var] = value

    return model


def load_rollout_model_llada(model, accelerator, pretrained_model_path, model_dtype=torch.bfloat16):
    """LLaDA: load a standalone rollout model and copy current weights."""
    import deepspeed

    unwrapped_model = accelerator.unwrap_model(model)

    import transformers.integrations.deepspeed as ds_integration
    saved_config = getattr(ds_integration, '_hf_deepspeed_config_weak_ref', None)
    ds_integration._hf_deepspeed_config_weak_ref = None

    try:
        rollout_model = RolloutLLaDAModelLM.from_pretrained(
            pretrained_model_path, torch_dtype=model_dtype
        )
    finally:
        ds_integration._hf_deepspeed_config_weak_ref = saved_config

    state_dict = {}
    for name, param in unwrapped_model.named_parameters():
        with deepspeed.zero.GatheredParameters([param]):
            state_dict[name] = param.data.detach().cpu().clone()
    for name, buf in unwrapped_model.named_buffers():
        state_dict[name] = buf.detach().cpu().clone()

    rollout_model.load_state_dict(state_dict, strict=False)
    del state_dict
    gc.collect()

    return rollout_model.to("cuda").eval()


def load_rollout_model_dream(model, accelerator, pretrained_model_path, model_dtype=torch.bfloat16):
    """Dream: load a standalone rollout model and copy current weights.

    Uses the same model class as training but instantiates separately
    to avoid side effects from the DeepSpeed wrapper.
    """
    import deepspeed

    unwrapped_model = accelerator.unwrap_model(model)

    import transformers.integrations.deepspeed as ds_integration
    saved_config = getattr(ds_integration, '_hf_deepspeed_config_weak_ref', None)
    ds_integration._hf_deepspeed_config_weak_ref = None

    try:
        rollout_model = RolloutDreamModel.from_pretrained(
            pretrained_model_path, torch_dtype=model_dtype
        )
    finally:
        ds_integration._hf_deepspeed_config_weak_ref = saved_config

    # Gather full parameters from DeepSpeed ZeRO-3 shards
    state_dict = {}
    for name, param in unwrapped_model.named_parameters():
        with deepspeed.zero.GatheredParameters([param]):
            state_dict[name] = param.data.detach().cpu().clone()
    for name, buf in unwrapped_model.named_buffers():
        state_dict[name] = buf.detach().cpu().clone()

    rollout_model.load_state_dict(state_dict, strict=False)
    del state_dict
    gc.collect()

    return rollout_model.to("cuda").eval()


def load_rollout_model(model, accelerator, pretrained_model_path, model_type: str = "llada", model_dtype=torch.bfloat16):
    """Unified entry point for loading a rollout model."""
    if model_type == "llada":
        return load_rollout_model_llada(model, accelerator, pretrained_model_path, model_dtype)
    else:
        return load_rollout_model_dream(model, accelerator, pretrained_model_path, model_dtype)