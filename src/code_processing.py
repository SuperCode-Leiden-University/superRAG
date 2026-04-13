"""
# task_id:
HumanEval/0

# prompt:
from typing import List
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    # Check if in given list of numbers, are any two numbers closer to each other than
    # given threshold.
    # >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    # False
    # >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    # True

# canonical_solution:
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False
"""

# extract the code from the model's answer
def extract_code(response, task_id, func_def):

    def_index = response.find(func_def) # robust signature for finding the function
    code_start = response.rfind("```", def_index) # code blocks start and end with ```
    code_end = response.find("```", def_index)

    code = response[code_start:code_end].strip()

    sample = {
        "task_id": task_id,
        "completion": code
    }
    return sample
