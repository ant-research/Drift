# System prompt templates

DEFAULT_SYSTEM_PROMPT_LLADA = (
    """<|startoftext|><|start_header_id|>user<|end_header_id|>"""
    """This is the problem:\n"""
    """{{problem}}<|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n"""
)

DEFAULT_SYSTEM_PROMPT_DREAM = (
    """<|im_start|>user\n"""
    """This is the problem:\n"""
    """{{problem}}<|im_end|>\n"""
    """<|im_start|>assistant\n"""
)

DEFAULT_SYSTEM_PROMPT_LLADA_MATH = (
    """<|startoftext|><|start_header_id|>user<|end_header_id|>"""
    """You need to put your final answer in \\boxed{}. This is the problem:\n"""
    """{{problem}}<|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n"""
)


DEFAULT_SYSTEM_PROMPT_LLADA_CODE = (
    """<|startoftext|><|start_header_id|>user<|end_header_id|>"""
    """{{problem}}\nPlace your code within a single Python code block ```python ```. Do not include more than one code block. """
    """<|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n"""
)

DEFAULT_SYSTEM_PROMPT_DREAM_MATH = (
    """<|im_start|>user\n"""
    """You need to put your final answer in \\boxed{}. This is the problem:\n"""
    """{{problem}}<|im_end|>\n"""
    """<|im_start|>assistant\n"""
)

DEFAULT_SYSTEM_PROMPT_DREAM_CODE = (
    """<|im_start|>user\n"""
    """{{problem}}\nPlace your code within a single Python code block ```python ```. Do not include more than one code block. """
    """<|im_end|>\n"""
    """<|im_start|>assistant\n"""
)


def get_default_system_prompt(model_type: str, data_type: str = "math") -> str:
    """Return the default system prompt template for the given model and data type."""
    if model_type == "dream":
        if data_type in ("code_stdio", "code_function"):
            return DEFAULT_SYSTEM_PROMPT_DREAM_CODE
        elif data_type == "math":
            return DEFAULT_SYSTEM_PROMPT_DREAM_MATH
        return DEFAULT_SYSTEM_PROMPT_DREAM
    else:
        if data_type in ("code_stdio", "code_function"):
            return DEFAULT_SYSTEM_PROMPT_LLADA_CODE
        elif data_type == "math":
            return DEFAULT_SYSTEM_PROMPT_LLADA_MATH
        return DEFAULT_SYSTEM_PROMPT_LLADA