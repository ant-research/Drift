import re
from typing import Callable, Dict, List, Union

import torch

from reward.code_executor import execute_code_tests, extract_code
from reward.math_utils import is_math_equal as _is_math_equal_sync


def extract_final_boxed_answer(s: str) -> str:
    r"""Extract the content of the last \boxed{...} in model output."""
    tag = r'\boxed{'
    start = s.rfind(tag)
    if start == -1:
        return "Can not extract the answer!"

    i = start + len(tag)
    depth = 1
    buf = []

    while i < len(s) and depth:
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                break
        buf.append(ch)
        i += 1

    return ''.join(buf) if depth == 0 else "Can not extract the answer!"


def is_math_equal(prediction: str, ground_truth: str) -> bool:
    """Synchronous math equality check using math_utils.is_math_equal."""
    return _is_math_equal_sync(prediction, ground_truth)


def math_reward(
    completions: List[str],
    ground_truths: List[str],
    **kwargs,
) -> List[float]:
    r"""
    Math correctness reward: extract \boxed{} answer and compare with
    ground truth. Correct → +1.0, incorrect → 0.0.
    """
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        prediction = extract_final_boxed_answer(completion)
        if prediction == "Can not extract the answer!":
            rewards.append(0.0)
        elif is_math_equal(prediction, gt):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def extract_answer_from_tags(text: str) -> str:
    """Extract answer from <answer>...</answer> tags.

    - Takes the last <answer> block (in case model writes one during reasoning)
    - Strips whitespace, newlines, separators
    - Returns a special marker on failure
    """
    if not isinstance(text, str):
        return "Can not extract the answer!"
    
    # re.DOTALL so . matches newlines; non-greedy
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if not matches:
        return "Can not extract the answer!"
    
    # Take the last match, strip whitespace and common separators
    raw = matches[-1].strip()
    cleaned = re.sub(r"[\s,\-_|]", "", raw)
    
    if not cleaned:
        return "Can not extract the answer!"
    return cleaned


def _normalize_sudoku(s: str) -> str:
    """Normalize: strip all non-digit characters."""
    return re.sub(r"\D", "", s or "")


def _is_valid_sudoku_format(s: str) -> bool:
    """Check if string is exactly 16 digits, each 1-4."""
    return len(s) == 16 and all(c in "1234" for c in s)


def sudoku_reward(completions, ground_truths, **kwargs):
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        prediction = extract_answer_from_tags(completion)
        
        # Cannot extract <answer> content
        if prediction == "Can not extract the answer!":
            rewards.append(0.0)
            continue
        
        norm_pred = _normalize_sudoku(prediction)
        norm_gt = _normalize_sudoku(gt)
        
        # Fully correct
        if norm_pred == norm_gt and _is_valid_sudoku_format(norm_pred):
            rewards.append(1.0)
        # # Valid format (16 digits of 1-4) but wrong answer
        # elif _is_valid_sudoku_format(norm_pred):
        #     rewards.append(0.5)
        # Other cases
        else:
            rewards.append(0.0)
    
    return rewards


def _safe_eval_arith(expr: str):
    """Safe eval: only digits, operators, parens, whitespace allowed."""
    if not re.fullmatch(r'[\d+\-*/().\s]+', expr):
        return None
    try:
        result = eval(expr, {"__builtins__": {}}, {})
        if not isinstance(result, (int, float)):
            return None
        return result
    except Exception:
        return None


def countdown_reward(completions, ground_truths, **kwargs):
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        prediction = extract_final_boxed_answer(completion)
        if prediction == "Can not extract the answer!":
            rewards.append(0.0)
            continue

        # Parse ground truth
        if isinstance(gt, dict):
            target = float(gt["target"])
            allowed_numbers = [int(n) for n in gt.get("numbers", [])]
        else:
            target = float(gt)
            allowed_numbers = None

        # Step 1: evaluate the expression
        value = _safe_eval_arith(prediction)
        if value is None or abs(value - target) > 1e-6:
            rewards.append(0.0)
            continue

        # Step 2: number usage check
        if allowed_numbers is not None:
            used_numbers = [int(n) for n in re.findall(r'\d+', prediction)]
            remaining = list(allowed_numbers)
            valid = True
            for n in used_numbers:
                if n in remaining:
                    remaining.remove(n)
                else:
                    valid = False
                    break
            if not valid:
                rewards.append(0.0)
                continue

        rewards.append(1.0)
    return rewards


def format_reward(
    completions: List[str],
    ground_truths: List[str],
    **kwargs,
) -> List[float]:
    r"""Format reward: +0.5 if output contains \boxed{}, else 0.0."""
    return [0.5 if r'\boxed{' in c else 0.0 for c in completions]


def code_reward(
    completions: List[str],
    ground_truths: Union[List[str], List[dict]],
    n_workers: int = 64,
    num_chunks: int = 64,
    **kwargs,
) -> List[float]:
    """
    Code generation correctness reward (continuous).

    Extracts Python code from each completion, executes it against the
    test cases carried in ``ground_truths``, and returns the fraction of
    tests passed per sample.
    """
    # ground_truths carries the test-case dicts for code tasks
    test_cases = ground_truths

    if test_cases is None or len(test_cases) == 0:
        return [0.0] * len(completions)

    # Safety check: if ground_truths is still plain strings (misconfigured),
    # return zeros rather than crashing
    if isinstance(test_cases[0], str):
        return [0.0] * len(completions)

    # Delegate execution to code_executor
    pass_fail = execute_code_tests(
        completions=completions,
        test_cases=test_cases,
        n_workers=n_workers,
        num_chunks=num_chunks,
    )

    # Convert boolean matrix → fractional pass rate
    rewards = []
    for pf in pass_fail:
        n_tests = len(pf)
        rewards.append(sum(pf) / n_tests if n_tests > 0 else 0.0)
    return rewards



REWARD_FUNCS_REGISTRY: Dict[str, Callable] = {
    # Math / reasoning
    "math":          math_reward,
    "sudoku":        sudoku_reward,
    "countdown":     countdown_reward,
    "format":        format_reward,
    "code":          code_reward,
}


def compute_reward_per_func(
    completions: List[str],
    ground_truths: Union[List[str], List[dict]],
    reward_funcs: List[str],
    n_workers: int = 64,
    num_chunks: int = 64,
) -> torch.Tensor:
    """
    Compute rewards for each reward function, returning (N, num_funcs) tensor.

    For code tasks, ``ground_truths`` should be a list of test-case dicts.
    For math tasks, ``ground_truths`` should be a list of answer strings.
    The two task types are never mixed in a single call.
    """
    N = len(completions)
    num_funcs = len(reward_funcs)
    rewards_per_func = torch.zeros(N, num_funcs, dtype=torch.float32)

    shared_kwargs = dict(
        n_workers=n_workers,
        num_chunks=num_chunks,
    )

    for func_idx, func_name in enumerate(reward_funcs):
        if func_name not in REWARD_FUNCS_REGISTRY:
            raise ValueError(
                f"Unknown reward function '{func_name}'. "
                f"Available: {list(REWARD_FUNCS_REGISTRY.keys())}"
            )
        reward_fn = REWARD_FUNCS_REGISTRY[func_name]

        func_rewards = reward_fn(
            completions=completions,
            ground_truths=ground_truths,
            **shared_kwargs,
        )

        rewards_per_func[:, func_idx] = torch.tensor(
            func_rewards, dtype=torch.float32
        )

    return rewards_per_func