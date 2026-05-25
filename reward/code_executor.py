from typing import List, Optional
import io
import os
import sys
import re
import time
import math
import textwrap
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed


# Test splitting

def _split_check_function(test_list: List[str]) -> List[str]:
    """Split test_list entries containing multiple asserts into individual tests.

    Parses the `def check(candidate): ... check(func_name)` pattern and
    produces one test entry per assert statement, so code_reward can return
    a continuous pass rate instead of binary 0/1.

    Non-assert lines (imports, variable assignments) are kept as setup code
    in each split test entry.
    """
    if not test_list:
        return test_list

    split_results = []
    for test_str in test_list:
        # Try to parse def check(candidate): ... check(func_name) structure
        # Match the check function definition
        check_match = re.search(
            r'^(.*?)'                           # preamble (METADATA etc.)
            r'(def\s+check\s*\([^)]*\)\s*:\s*\n)'  # def check(...):
            r'(.*?)'                            # body (indented lines)
            r'(\ncheck\s*\([^)]*\)\s*)$',       # check(func_name)
            test_str, re.DOTALL
        )
        if not check_match:
            # Cannot parse, keep as-is
            split_results.append(test_str)
            continue

        preamble = check_match.group(1)
        check_def = check_match.group(2)
        body = check_match.group(3)
        check_call = check_match.group(4)

        # Split body into setup lines and assert lines
        lines = body.split('\n')
        setup_lines = []
        assert_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('assert ') or stripped.startswith('assert('):
                assert_lines.append(line)
            else:
                # Setup code: imports, variable assignments, loops, etc.
                setup_lines.append(line)

        if len(assert_lines) <= 1:
            # Only 0 or 1 assert, no splitting needed
            split_results.append(test_str)
            continue

        # Build an independent check function for each assert
        setup_block = '\n'.join(setup_lines) + '\n' if setup_lines else ''
        for assert_line in assert_lines:
            single_test = (
                f"{preamble}{check_def}"
                f"{setup_block}{assert_line}\n"
                f"{check_call}"
            )
            split_results.append(single_test)

    return split_results


# Code extraction

def extract_code(full_output: str) -> str:
    """Extract the last ```python ... ``` code block from model output."""
    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return "# Code extraction failed — no python code block found."


# Function-style testing (assert statements)

def _run_many_pipe(snippet: str, tests: List[str], conn):
    """Worker: exec snippet then run each assert-style test statement."""
    import io as _io
    _original_stdout = sys.stdout
    sys.stdout = _io.StringIO()
    results = []
    try:
        ns = {}
        exec(textwrap.dedent(snippet), ns, ns)
        for stmt in tests:
            try:
                exec(stmt, ns, ns)
                results.append(True)
            except AssertionError:
                # Relaxed retry: compare str(actual) == str(expected)
                try:
                    if "==" in stmt:
                        body = stmt.replace("assert", "", 1).strip()
                        lhs, rhs = body.split("==", 1)
                        actual = eval(lhs.strip(), ns, ns)
                        expected = eval(rhs.strip(), ns, ns)
                        results.append(str(actual) == str(expected))
                    else:
                        results.append(False)
                except Exception:
                    results.append(False)
            except SystemExit:
                results.append(True)
            except Exception:
                results.append(False)
        conn.send(results)
    except SystemExit:
        conn.send([True] * len(tests))
    except Exception:
        conn.send([False] * len(tests))
    finally:
        sys.stdout = _original_stdout
        try:
            conn.close()
        except Exception:
            pass


# Env vars to clean before subprocess to prevent CUDA/distributed init
_ENV_KEYS_TO_CLEAN = [
    "LOCAL_RANK", "RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
    "MASTER_ADDR", "MASTER_PORT",
    "CUDA_VISIBLE_DEVICES",
    "DEEPSPEED_ZERO_STAGE", "DEEPSPEED_CONFIG", "DEEPSPEED_CONFIG_FILE",
    "OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE",
    "NCCL_SOCKET_IFNAME", "NCCL_DEBUG",
    "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT",
    "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE",
]


def _clean_env_worker(snippet: str, tests: List[str], conn):
    """Wrapper: clean env vars, isolate CWD to tmpdir, then run tests."""
    import tempfile, shutil
    for key in _ENV_KEYS_TO_CLEAN:
        os.environ.pop(key, None)
    # Isolate: run code in a disposable temp directory so that any file
    # I/O performed by model-generated code does not pollute the main CWD.
    tmpdir = tempfile.mkdtemp(prefix="code_exec_")
    try:
        os.chdir(tmpdir)
        _run_many_pipe(snippet, tests, conn)
    finally:
        try:
            os.chdir("/")
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


def _check_snippet_many(
    snippet: str,
    tests: List[str],
    t_limit: int,
    spawn_slack: float = 5.0,
) -> List[bool]:
    """Run snippet + tests in a forked subprocess with timeout."""
    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(
        target=_clean_env_worker,
        args=(snippet, tests, child_conn),
        daemon=True,
    )
    p.start()
    child_conn.close()

    deadline = time.monotonic() + t_limit + spawn_slack
    res = None
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            wait = min(remaining, 0.05)
            if parent_conn.poll(wait):
                try:
                    res = parent_conn.recv()
                except EOFError:
                    res = None
                break
            if not p.is_alive():
                if parent_conn.poll(0.05):
                    try:
                        res = parent_conn.recv()
                    except EOFError:
                        res = None
                break

        if res is None and parent_conn.poll(0.05):
            try:
                res = parent_conn.recv()
            except EOFError:
                res = None

        if res is None:
            if p.is_alive():
                p.terminate()
            res = [False] * len(tests)
    finally:
        try:
            p.join(timeout=0.5)
        except Exception:
            pass
        try:
            parent_conn.close()
        except Exception:
            pass

    return [bool(x) for x in res]


def evaluate_function_tests(
    snippets: List[str],
    test_lists: List[List[str]],
    time_limits: List[int],
    n_workers: Optional[int] = None,
) -> List[List[bool]]:
    """
    Run function-style tests for a batch of code snippets.

    Each snippet is executed in an isolated spawn-subprocess, then each
    assert statement in the corresponding test_list is evaluated.

    Parameters
    ----------
    snippets : list[str]
        Extracted code strings.
    test_lists : list[list[str]]
        Per-snippet list of assert statements.
    time_limits : list[int]
        Per-snippet execution time limit (seconds).
    n_workers : int, optional
        Thread pool size (defaults to cpu count).

    Returns
    -------
    results : list[list[bool]]
        results[i][j] = whether snippet i passed test j.
    """
    n_cpu = os.cpu_count() or 4
    n_workers = max(1, int(n_workers)) if n_workers is not None else n_cpu

    results: List[List[bool]] = [
        [False] * len(tests) for tests in test_lists
    ]

    futures = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for i, (snippet, tests, tl) in enumerate(
            zip(snippets, test_lists, time_limits)
        ):
            fut = pool.submit(_check_snippet_many, snippet, tests, tl)
            futures[fut] = i

        for fut in as_completed(futures):
            i = futures[fut]
            try:
                ok_list = fut.result()
            except Exception:
                ok_list = [False] * len(test_lists[i])
            results[i] = [bool(x) for x in ok_list]

    return results


# STDIO-style testing (stdin/stdout matching)

def _worker_stdio(script: str, input_val: str, output_queue):
    """Worker: exec script with simulated stdin/stdout in isolated tmpdir."""
    import tempfile, shutil

    # Isolate CWD so model-generated file I/O won't pollute the main directory
    tmpdir = tempfile.mkdtemp(prefix="code_exec_stdio_")
    try:
        os.chdir(tmpdir)

        input_lines = iter(input_val.splitlines())

        def fake_input(prompt=""):
            try:
                return next(input_lines)
            except StopIteration:
                raise EOFError("No more input")

        stdout_capture = io.StringIO()
        original_stdout = sys.stdout
        original_stdin = sys.stdin
        sys.stdout = stdout_capture
        sys.stdin = io.StringIO(input_val)

        context = {"__name__": "__main__", "input": fake_input}

        try:
            exec(script, context)
            output_queue.put(stdout_capture.getvalue())
        except SystemExit:
            output_queue.put(stdout_capture.getvalue())
        except Exception as e:
            output_queue.put(f"error: {e}")
        finally:
            sys.stdout = original_stdout
            sys.stdin = original_stdin
    finally:
        try:
            os.chdir("/")
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


def _run_scripts_with_timeout(
    scripts: List[str],
    inputs: List[str],
    time_limits: List[float],
) -> List[str]:
    """Run scripts in parallel processes with per-script timeout."""
    results = [None] * len(scripts)
    processes = []
    queues = []
    deadlines = []

    for i in range(len(scripts)):
        q = mp.Queue()
        p = mp.Process(target=_worker_stdio, args=(scripts[i], inputs[i], q))
        processes.append(p)
        queues.append(q)
        p.start()
        deadlines.append(time.time() + time_limits[i])

    while any(p.is_alive() for p in processes):
        now = time.time()
        for i, p in enumerate(processes):
            if p.is_alive() and now >= deadlines[i]:
                p.terminate()
                results[i] = "Timeout Error"
        time.sleep(0.001)

    for i, p in enumerate(processes):
        if results[i] is None:
            try:
                results[i] = queues[i].get_nowait()
            except Exception as e:
                results[i] = f"Execution Error: {e}"

    return results


def _stdio_output_eq(actual: str, expected: str) -> bool:
    """Whitespace-normalized string comparison for stdio output."""
    return " ".join(actual.split()) == " ".join(expected.split())


def evaluate_stdio_tests(
    snippets: List[str],
    test_inputs: List[List[str]],
    test_outputs: List[List[str]],
    time_limits: List[int],
    num_chunks: int = 64,
) -> List[List[bool]]:
    """
    Run STDIO-style tests for a batch of code snippets.

    Each snippet is executed in a subprocess with simulated stdin;
    actual stdout is compared to expected output via whitespace-normalized
    string matching.

    Parameters
    ----------
    snippets : list[str]
        Extracted code strings.
    test_inputs : list[list[str]]
        Per-snippet list of stdin inputs.
    test_outputs : list[list[str]]
        Per-snippet list of expected stdout outputs.
    time_limits : list[int]
        Per-snippet execution time limit (seconds).
    num_chunks : int
        Chunk count for batched subprocess execution.

    Returns
    -------
    results : list[list[bool]]
        results[i][j] = whether snippet i produced correct output for case j.
    """
    # Flatten (snippet, test_case) pairs
    flat_code, flat_inp, flat_tl = [], [], []
    flat_idx = []  # (snippet_index, test_case_index)

    for i, (snippet, inputs, tl) in enumerate(
        zip(snippets, test_inputs, time_limits)
    ):
        for k, inp in enumerate(inputs):
            flat_code.append(snippet)
            flat_inp.append(inp)
            flat_tl.append(tl)
            flat_idx.append((i, k))

    # Run in chunks to avoid process explosion
    n = len(flat_code)
    chunk_size = max(1, math.ceil(n / max(1, num_chunks)))
    exe_results = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_results = _run_scripts_with_timeout(
            flat_code[start:end],
            flat_inp[start:end],
            flat_tl[start:end],
        )
        exe_results.extend(chunk_results)

    # Unflatten
    results: List[List[bool]] = [
        [False] * len(outs) for outs in test_outputs
    ]
    for flat_i, actual in enumerate(exe_results):
        i, k = flat_idx[flat_i]
        expected = test_outputs[i][k]
        results[i][k] = _stdio_output_eq(actual, expected)

    return results


# Unified entry point

def execute_code_tests(
    completions: List[str],
    test_cases: List[dict],
    n_workers: int = 64,
    num_chunks: int = 64,
) -> List[List[bool]]:
    """
    Execute generated code against test cases and return pass/fail results.

    Automatically dispatches to function-style or stdio-style execution
    based on each test_case's "test_method" field.

    Parameters
    ----------
    completions : list[str]
        Raw model outputs (containing ```python ... ``` blocks).
    test_cases : list[dict]
        Per-sample test specification. Each dict must contain:
          - "test_method": "function" or "stdio"
        For "function":
          - "test_list": list[str]  (assert statements)
          - "prefix": str (optional, function signature to prepend)
        For "stdio":
          - "test_input": list[str]
          - "test_output": list[str]
        Optional:
          - "test_time_limit": int (default 1 second)
    n_workers : int
        Thread pool size for function-style tests.
    num_chunks : int
        Chunk count for stdio-style tests.

    Returns
    -------
    results : list[list[bool]]
        results[i][j] = whether completion i passed test case j.
        Length of results[i] equals the number of test cases for that item.
    """
    N = len(completions)
    assert len(test_cases) == N, (
        f"completions ({N}) and test_cases ({len(test_cases)}) length mismatch"
    )

    all_results: List[List[bool]] = [[] for _ in range(N)]

    # Partition by test method
    func_indices, stdio_indices = [], []
    for i, tc in enumerate(test_cases):
        method = tc.get("test_method", "function")
        if method == "stdio":
            stdio_indices.append(i)
        else:
            func_indices.append(i)

    # Function-style execution
    if func_indices:
        snippets, test_lists, time_limits = [], [], []
        for i in func_indices:
            tc = test_cases[i]
            prefix = tc.get("prefix", "")
            # Prepend prefix to completion before extraction (matches RL framework)
            if prefix:
                code = extract_code(prefix + completions[i])
            else:
                code = extract_code(completions[i])
            snippets.append(code)
            test_lists.append(_split_check_function(tc["test_list"]))
            time_limits.append(tc.get("test_time_limit", 1))

        func_results = evaluate_function_tests(
            snippets, test_lists, time_limits, n_workers=n_workers
        )
        for local_i, global_i in enumerate(func_indices):
            all_results[global_i] = func_results[local_i]

    # STDIO-style execution
    if stdio_indices:
        snippets, test_inputs, test_outputs, time_limits = [], [], [], []
        for i in stdio_indices:
            tc = test_cases[i]
            code = extract_code(completions[i])
            snippets.append(code)
            test_inputs.append(tc["test_input"])
            test_outputs.append(tc["test_output"])
            time_limits.append(tc.get("test_time_limit", 1))

        stdio_results = evaluate_stdio_tests(
            snippets, test_inputs, test_outputs, time_limits,
            num_chunks=num_chunks,
        )
        for local_i, global_i in enumerate(stdio_indices):
            all_results[global_i] = stdio_results[local_i]

    return all_results