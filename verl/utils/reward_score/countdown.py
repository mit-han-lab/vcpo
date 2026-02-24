import random
import re

def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:" if present
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]

    # Prefer boxed answers if present.
    boxed_pattern = r"\\boxed\{([^}]*)\}"
    boxed_matches = list(re.finditer(boxed_pattern, solution_str))
    if boxed_matches:
        final_answer = boxed_matches[-1].group(1).strip()
        return final_answer

    # Fallback to <answer> tags if provided.
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str))
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]

        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)

        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except Exception:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception:
        return None


def _normalize_equation(equation_str: str) -> str:
    return re.sub(r"\s+", "", equation_str or "")


def _extract_target_numbers(ground_truth, extra_info):
    if isinstance(ground_truth, dict):
        target = ground_truth.get("target") or ground_truth.get("target_number")
        numbers = ground_truth.get("numbers") or ground_truth.get("nums")
        if target is not None and numbers is not None:
            return target, numbers

    if extra_info:
        target = extra_info.get("target")
        numbers = extra_info.get("numbers")
        if target is None or numbers is None:
            metadata = extra_info.get("metadata", {}) if isinstance(extra_info, dict) else {}
            target = target if target is not None else metadata.get("target")
            numbers = numbers if numbers is not None else metadata.get("numbers")
        if target is not None and numbers is not None:
            return target, numbers

    return None, None


def compute_score(solution_str, ground_truth, extra_info=None, method="strict", format_score=0.0, score=1.0):
    """The scoring function for countdown task.

    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers or a string equation
        extra_info: optional dict with target/numbers from preprocessing
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    target, numbers = _extract_target_numbers(ground_truth, extra_info)

    equation = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print("No equation found")
        return 0

    if target is None or numbers is None:
        normalized_pred = _normalize_equation(equation)
        normalized_gt = _normalize_equation(ground_truth) if isinstance(ground_truth, str) else ""
        return score if normalized_pred == normalized_gt and normalized_gt else 0.0

    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            print("Invalid equation")
        return format_score

    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print("Could not evaluate equation")
            return format_score

        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except Exception:
        if do_print:
            print("Error evaluating equation")
        return format_score
