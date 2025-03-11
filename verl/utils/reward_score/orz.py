import re


def extract_solution(solution_str):

    # pattern = re.compile(r"(\\boxed{.*})")
    pattern = re.compile(r"<answer>.*?(\\boxed{.*}).*?</answer>", re.DOTALL)
    matches = re.findall(pattern, solution_str)
    result = matches[-1] if matches else ""
    return result

def compute_format_score(text):
    pattern = r'<answer>.*?(\\boxed\{.*?\}).*?</answer>'
    return 1 if re.search(pattern, text, re.DOTALL) else 0

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

    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score + format_score
        else:
            return format_score