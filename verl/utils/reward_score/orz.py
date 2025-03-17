import re
from math_verify.metric import math_metric
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None
    
def remove_answer(s):
    left = "<answer>"
    right = "</answer>"
    try:
        assert s[: len(left)] == left
        assert s[-len(right) :] == right
        return s[len(left) : -len(right)]
    except Exception:
        return None


def get_answer_str(s: str) -> str:
    res = remove_boxed(last_boxed_only_string(s))
    if res is not None:
        return res
    else:
        res = remove_answer(s)
        if res is not None:
            return res
        else:
            return s

def extract_solution(solution_str):
    # First, find all <answer>...</answer> blocks
    answer_pattern = re.compile(r"<answer>.*?</answer>", re.DOTALL)
    answer_blocks = re.findall(answer_pattern, solution_str)
    # Second, find \boxed{...} in each block
    boxed_pattern = re.compile(r"\\boxed{.*?}")
    matches = []
    for block in answer_blocks:
        boxed_matches = re.findall(boxed_pattern, block)
        matches.extend(boxed_matches)
    # Return the last match if it exists, else an empty string
    result = matches[-1] if matches else answer_blocks[-1] if answer_blocks else ""
    result =  "\\boxed{" + get_answer_str(result) + "}"
    return result

def compute_format_score(text):
    answer_pattern = re.compile(r"<answer>.*?</answer>", re.DOTALL)
    boxed_pattern = re.compile(r"\\boxed{.*?}")
    answer_blocks = re.findall(answer_pattern, text)
    boxed_matches = []
    for block in answer_blocks:
        boxed_matches = re.findall(boxed_pattern, block)
    if not answer_blocks:
        return -2
    elif not boxed_matches:
        return -1
    else:
        return 0

def compute_score(solution_str, ground_truth, format_score=0., score=1.):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    format_score = compute_format_score(solution_str)

    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.

    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [answer])
    except Exception as e:
        print(e)

    return ret_score + format_score