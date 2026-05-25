from typing import Optional
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Try to import sympy (for LaTeX equality checking)
try:
    from sympy.parsing.latex import parse_latex
    _HAS_SYMPY = True
except ImportError:
    _HAS_SYMPY = False

# Try to import math_verify (optional, stronger equality checking)
try:
    from math_verify import parse as mv_parse, verify as mv_verify
    _HAS_MATH_VERIFY = True
except ImportError:
    _HAS_MATH_VERIFY = False


# Answer extraction

def extract_final_boxed_answer(s: str) -> str:
    r"""Extract the content of the last \boxed{...} from model output."""
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


# Math equivalence

# Module-level thread pool (for sympy timeout control, reused)
_THREAD_POOL: Optional[ThreadPoolExecutor] = None


def _get_thread_pool() -> ThreadPoolExecutor:
    global _THREAD_POOL
    if _THREAD_POOL is None:
        _THREAD_POOL = ThreadPoolExecutor(max_workers=4)
    return _THREAD_POOL


# Repetition detection

def _repeatness(s: str) -> bool:
    """Detect high substring repetition (suffix array + LCP)."""
    from itertools import islice, zip_longest

    def ranks(l):
        index = {v: i for i, v in enumerate(sorted(set(l)))}
        return [index[v] for v in l]

    def suffix_array(s_arr):
        line = ranks(s_arr)
        n, k, ans, sa = len(s_arr), 1, line, [0] * len(s_arr)
        while k < n - 1:
            line = ranks(list(zip_longest(line, islice(line, k, None), fillvalue=-1)))
            ans, k = line, k << 1
        for i, k_val in enumerate(ans):
            sa[k_val] = i
        return ans, sa

    def lcp(arr, sa, inv_suff):
        n, ans, k = len(arr), [0] * len(arr), 0
        for i in range(n):
            if inv_suff[i] == n - 1:
                k = 0
                continue
            j = sa[inv_suff[i] + 1]
            while i + k < n and j + k < n and arr[i + k] == arr[j + k]:
                k += 1
            ans[inv_suff[i]] = k
            if k > 0:
                k -= 1
        return ans

    arr = [ord(c) for c in s]
    n = len(arr)
    if n <= 1:
        return False
    c, sa = suffix_array(arr)
    cnt = sum(lcp(arr, sa, c))
    return (cnt * 2 / (n * (n + 1))) > 0.2


# String normalization

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if not substr or substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    new_str += "{" + a + "}{" + b + "}" + substr[2:]
                else:
                    new_str += "{" + a + "}" + b + substr[2:]
    return new_str


def _fix_a_slash_b(string):
    parts = string.split("/")
    if len(parts) != 2:
        return string
    try:
        a, b = int(parts[0]), int(parts[1])
        if string == f"{a}/{b}":
            return f"\\frac{{{a}}}{{{b}}}"
    except (ValueError, AssertionError):
        pass
    return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def _strip_string(string):
    """Comprehensive LaTeX string normalization."""
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace(",", "")

    # Strip units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            string = splits[0]

    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # Add leading zeros
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if string and string[0] == ".":
        string = "0" + string

    # Remove 'k = ' style prefixes
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)

    if string == "0.5":
        string = "\\frac{1}{2}"

    string = _fix_a_slash_b(string)
    return string


SUBSTITUTIONS = [
    ("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""),
    (r"\ ", ""), (" ", ""), ("mbox", "text"),
    (",\\text{and}", ","), ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "ft",
    "hours", "km", "units", "\\ldots", "sue", "points", "feet",
    "minutes", "digits", "cents", "degrees", "cm", "gm", "pounds",
    "meters", "meals", "edges", "students", "childrentickets",
    "multiples", "\\text{s}", "\\text{.}", "\\text{\ns}",
    "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}",
    r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}",
    '"', "\\dots",
]


def _normalize_final_answer(final_answer: str) -> str:
    """Normalization from Minerva (arxiv 2206.14858)."""
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer


# Equality levels

def _is_equiv(str1: str, str2: str) -> bool:
    """Fast path: string normalization + float comparison, no sympy."""
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        ss1, ss2 = _strip_string(str1), _strip_string(str2)
        try:
            return float(ss1) == float(ss2)
        except (ValueError, TypeError):
            return ss1 == ss2
    except Exception:
        return str1 == str2


def _is_latex_equal_impl(str1: str, str2: str) -> bool:
    """
    Slow path: sympy LaTeX parsing for symbolic/numeric comparison.
    May be slow; caller should enforce a timeout.
    """
    if not _HAS_SYMPY:
        # No sympy available, fall back to normalized string comparison
        return _normalize_final_answer(str1) == _normalize_final_answer(str2)

    try:
        sym1, val1 = parse_latex(str1), parse_latex(str1).evalf()
        sym2, val2 = parse_latex(str2), parse_latex(str2).evalf()
        if sym1 == sym2 or val1 == val2:
            return True
        raise ValueError
    except Exception:
        try:
            n1 = _normalize_final_answer(str1)
            n2 = _normalize_final_answer(str2)
            sym1, val1 = parse_latex(n1), parse_latex(n1).evalf()
            sym2, val2 = parse_latex(n2), parse_latex(n2).evalf()
            if sym1 == sym2 or val1 == val2:
                return True
        except Exception:
            return _normalize_final_answer(str1) == _normalize_final_answer(str2)
    return False


def _math_verify_equal_impl(str1: str, str2: str) -> bool:
    """Check equality using the math_verify library."""
    if not _HAS_MATH_VERIFY:
        return False
    return mv_verify(mv_parse(str1), mv_parse(str2))


def is_math_equal(
    prediction: str,
    ground_truth: str,
    timeout: float = 2.0,
    math_mode: str = "legacy",
) -> bool:
    """
    Synchronous math answer equality check with three-tier logic and timeout.

    Pipeline:
      1. _is_equiv: fast string normalization + float comparison (<1ms)
      2. Repetition detection: skip garbage long outputs
      3. _is_latex_equal_impl / _math_verify_equal_impl:
         sympy or math_verify equality, run in thread pool with timeout

    Parameters
    ----------
    prediction, ground_truth : str
    timeout : float
        Timeout in seconds for sympy/math_verify
    math_mode : str
        "legacy" for sympy, "math_verify" for math_verify library
    """
    if prediction is None or ground_truth is None:
        return prediction is None and ground_truth is None

    # Fast path
    if _is_equiv(prediction, ground_truth):
        return True

    # Repetition detection
    if (len(prediction) > 128 and _repeatness(prediction)):
        return False
    if (len(ground_truth) > 128 and _repeatness(ground_truth)):
        return False

    # Slow path (with timeout)
    pool = _get_thread_pool()

    if math_mode == "legacy":
        future = pool.submit(_is_latex_equal_impl, prediction, ground_truth)
    elif math_mode == "math_verify":
        future = pool.submit(_math_verify_equal_impl, prediction, ground_truth)
    else:
        raise ValueError(f"Unknown math_mode: {math_mode}")

    try:
        return bool(future.result(timeout=timeout))
    except FuturesTimeoutError:
        future.cancel()
        return False
    except Exception:
        return False