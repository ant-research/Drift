# Data Format Specification

All dataset files are placed in the `data/` directory as JSON files. Each file is a JSON array of objects. The required fields depend on the `data_type` configured in the YAML config.

## Data Types Overview

| `data_type` | Task | Reward Function | Description |
|---|---|---|---|
| `math` | Mathematical reasoning | `math` | Math problems with a ground truth answer |
| `sudoku` | Sudoku puzzle solving | `sudoku` | Sudoku puzzles with a ground truth answer |
| `countdown` | Countdown number game | `countdown` | Countdown problems with a ground truth answer |
| `code_function` | Code generation (function) | `code` | Code problems verified by function-call unit tests |
| `code_stdio` | Code generation (stdio) | `code` | Code problems verified by stdin/stdout tests |

---

## Math / Sudoku / Countdown

These data types share the same format. Each sample requires a question and a ground truth answer string.
`MATH_rldf.json` is derived by rolling out 16 generations per prompt with LLaDA and filtering for samples with non-zero advantage standard deviation, ensuring meaningful training signal.

### Required Fields

| Field | Type | Description |
|---|---|---|
| `question` | `string` | The problem statement |
| `ground_truth_answer` | `string` | The expected answer |

### Optional Fields

| Field | Type | Description |
|---|---|---|
| `level` | `int` | Difficulty level (for reference only, not used in training) |
| `subject` | `string` | Subject category (for reference only, not used in training) |

### Example

```json
[
  {
    "question": "What is the value of $2 + 3$?",
    "ground_truth_answer": "5"
  },
  {
    "question": "Solve for $x$: $2x + 1 = 7$.",
    "ground_truth_answer": "3"
  }
]
```

---

## Code — Function Mode (`code_function`)

Each sample contains a coding problem and a set of function-call test cases for verification.

### Required Fields

| Field | Type | Description |
|---|---|---|
| `question` | `string` | The coding problem description |
| `test_list` | `list[string]` | A list of test code strings, each defining a `check(candidate)` function that asserts correctness |
| `test_time_limit` | `number` | Time limit in seconds for running tests (typically `1`) |
| `test_method` | `string` | Must be `"function"` (can be omitted if `data_type` is `code_function`, defaults to `"function"`) |

### Optional Fields

| Field | Type | Description |
|---|---|---|
| `prefix` | `string` | Code prefix prepended to the model's output before running tests (e.g., import statements) |

### Example

```json
[
  {
    "question": "Write a python function to remove first and last occurrence of a given character from the string.\nYour code should pass the test:\nassert remove_Occ(\"hello\",\"l\") == \"heo\"\n",
    "test_list": [
      "\n\ndef check(candidate):\n    assert remove_Occ(\"hello\",\"l\") == \"heo\"\n    assert remove_Occ(\"abcda\",\"a\") == \"bcd\"\n\ncheck(remove_Occ)\n"
    ],
    "test_time_limit": 1,
    "test_method": "function"
  }
]
```

---

## Code — Stdio Mode (`code_stdio`)

Each sample contains a coding problem and stdin/stdout test pairs for verification.

### Required Fields

| Field | Type | Description |
|---|---|---|
| `question` | `string` | The coding problem description |
| `test_input` | `list[string]` | List of stdin inputs for test cases |
| `test_output` | `list[string]` | List of expected stdout outputs (must match `test_input` in length) |
| `test_time_limit` | `number` | Time limit in seconds for running tests (typically `1`) |
| `test_method` | `string` | Must be `"stdio"` (can be omitted if `data_type` is `code_stdio`, defaults to `"stdio"`) |

### Example

```json
[
  {
    "question": "Read two integers from stdin and print their sum.",
    "test_input": ["2 3\n", "10 20\n"],
    "test_output": ["5\n", "30\n"],
    "test_time_limit": 1,
    "test_method": "stdio"
  }
]
```

---

## Config Reference

In the YAML config file, the dataset section should specify:

```yaml
dataset:
    train_dataset: "MATH_train"      # filename without .json extension in data/
    data_type: "math"                # one of: math, sudoku, countdown, code_function, code_stdio

evaluation:
    eval_dataset: "MATH500"          # filename without .json extension in data/
    data_type: "math"                # should match the task type
```
You can also build your own dataset and specify the filename in the config.

The `reward_funcs` in the training section should correspond to the data type:

```yaml
training:
    reward_funcs: ["math"]   
```
