import re
from math_verify import parse, LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig
from math_verify import verify

EXTRACTION_CONFIG = [LatexExtractionConfig(), ExprExtractionConfig()]


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    CORRECT_REWARD = 1.0
    INCORRECT_REWARD = 0.0

    # Put the ground truth into latex environment so HF math_verify can parse it as a math expression
    ground_truth = "$" + str(ground_truth) + "$"

    # Trust the HF math_verify library to parse the solution and ground truth
    # Note: this will not parse plain text and multiple choice answers correctly, such as "A".
    # To parse plain text, add StringExtractionConfig.
    gold = parse(ground_truth, extraction_config=EXTRACTION_CONFIG)
    output = parse(solution_str, extraction_config=EXTRACTION_CONFIG)
    if verify(gold, output):
        return CORRECT_REWARD

    return INCORRECT_REWARD


if __name__ == "__main__":
    # Test the function with a sample input
    solution_1 = r"This is a test \\boxed{\\left( 3, \\frac{\\pi}{2}\\right)}"
    ground_truth_1 = r"\\left(3, \\frac{\\pi}{2} \\right)"
    score_1 = compute_score(None, solution_1, ground_truth_1)
    print(f"Score for solution 1: {score_1}, expected: 1.0")  # Expected output: 1.0

    solution_2 = r"This is a test \\boxed{2.5}"
    ground_truth_2 = r"5/4"
    score_2 = compute_score(None, solution_2, ground_truth_2)
    print(f"Score for solution 2: {score_2}, expected: 0.0")  # Expected output: 0.0