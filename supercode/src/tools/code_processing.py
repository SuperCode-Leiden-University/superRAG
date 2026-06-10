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
    # sanity check
    if entry_point is not None:
        entry_index = response.find(entry_point)
        if entry_index == -1:
            print("WARNING: entry_point not found")
            return ""

    # tags from instructions
    offset = len("<code>")
    code_start = response.rfind("<code>")+offset
    code_end = response.rfind("</code>")

    if code_start==-1+offset or code_end==-1:
        # failsafe in case the model uses the standard ``` ... ```
        offset = len("```")
        code_end = response.rfind("```")
        code_start = response.rfind("```", 0, code_end) + offset

        pl_offset = len("python")
        if response.rfind("python", code_start, code_end) :
            code_start+=pl_offset ; offset+=pl_offset

        if code_start==-1+offset or code_end==-1:
            print("WARNING: code not found")
            return ""
    print(">> code extracted successfully")
    code = response[code_start:code_end].strip()
    return code



def extract_test_code(prompt, test, entry_point):
    test_code = ""

    start = 0
    failsafe = 0

    keyword_start = "assert candidate("
    keyword_end = ") =="
    next_start = test.find(keyword_start, start)

    while next_start > -1:
        failsafe += 1
        print("\n--------\n>> looping", failsafe)

        start = test.find(keyword_start, start) + len(keyword_start)
        end = test.find(keyword_end, start)
        next_start = test.find("\n"+keyword_start, start)

        #print("start", start, "\nend", end, "\nnext_start", next_start)

        test_case = test[start: end]
        print(">> test_case:", test_case)

        test_sol = test[end + len(keyword_end): next_start-1] # the -1 is to account for the "\n" I would get otherwise
        print(">> test_sol:", test_sol)

        if test_case in prompt:
            assertion = f"\nassert " + entry_point + "(" + test_case + ") == " + test_sol + ", f'the correct result is {" + test_sol + "} but the function output is {" + entry_point + "(" + test_case + ")}' "
            #print("\nassertion:", assertion)
            test_code += assertion
        if failsafe > 50:
            print("WARNING: extract_test_code failsafe activated!")
            break

        print("test_code:\n", test_code)

        return test_code


def convert_to_json(task_id, response, code, **kwargs):
    json_sample = {
        "task_id": task_id,
        "completion": response, # backup the model's answer
        "code": code,
        **kwargs
    }
    return json_sample