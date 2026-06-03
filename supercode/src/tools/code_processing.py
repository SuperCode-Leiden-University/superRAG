""" sample structure for HumanEval:
'canonical_solution':  'for idx, elem in enumerate(numbers):'
                       '    for idx2, elem2 in enumerate(numbers):'
                       '        if idx != idx2:'
                       '            distance = abs(elem - elem2)'
                       '            if distance < threshold:'
                       '                return True'
                       'return False',
 'entry_point': 'has_close_elements',
 'prompt': 'from typing import List'
           'def has_close_elements(numbers: List[float], threshold: float) -> bool:'
           '    "Check if in given list of numbers, are any two numbers closer to each other than'
           '    given threshold.'
           '    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)'
           '    False'
           '    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)'
           '    True',
 'task_id': 'HumanEval/0',
 'test': 'METADATA = {'author': 'jt','dataset': 'test'}'
         'def check(candidate):'
         '    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True'
         '    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False'
         '    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True'
         '    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False'
         '    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True'
         '    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True'
         '    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False'
"""

# extract the code from the model's answer
def extract_code(response, entry_point=None):
    """
    if entry_point == None: code_def = response.rfind("def ") # if entry_point is unknown
    else: code_def = response.rfind("def "+entry_point) # robust signature for finding the function

    if code_def==-1:
        print("WARNING: function not found")
        return ""
    code_start = response.rfind("```", 0, code_def)+3
    if response.rfind("python", code_start, code_def)>-1:
        code_start += 6 # remove "python" as well if present
    code_end = response.find("```", code_start)  # code blocks start and end with ``` (exclude ```)
    """
    code_start = response.rfind("<code>")+6
    code_end = response.rfind("</code>")
    if code_start==-1 or code_end==-1:
        print("WARNING: code not found")
        return ""

    code = response[code_start:code_end].strip()
    return code


def convert_to_json(task_id, response, code, **kwargs):
    json_sample = {
        "task_id": task_id,
        "completion": response, # backup the model's answer
        "code": code,
        **kwargs
    }
    return json_sample