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
    failsafe_assertion = ""

    start = 0
    failsafe = 0
    verbose = 1

    keyword_start = "assert"    ; len_start = len(keyword_start)
    keyword_func = "candidate(" ; len_func = len(keyword_func)
    keyword_end_line = "\n"     #; len_end = len(keyword_end_line)
    next_start = test.find(keyword_start, start+len_start)

    while next_start > -1:
        if verbose > 1: print("\n--------\n>> looping", failsafe)

        func = test.find(keyword_func, next_start) #+ len_func
        start = test.rfind(keyword_start, next_start, func) #+ len_start
        end_line = test.find(keyword_end_line, start)
        next_start = test.find(keyword_start, start+len_start)

        if verbose > 1: print(">> start =", start, "; func =", func, "; end_line =", end_line, "; next_start =", next_start)

        assert_line = test[start:end_line]
        if verbose > 0: print(">> assert_line:", assert_line)

        end_assert = max(
            assert_line.rfind("= ")+2,
            assert_line.rfind("> ")+2,
            assert_line.rfind("< ")+2,
        )

        test_case = assert_line[
            assert_line.find(keyword_func)+len_func :
            assert_line.rfind(")", 0, end_assert)
        ]
        if verbose > 1: print(">> test_case:", test_case)

        test_sol = assert_line[end_assert:]
        if verbose > 1: print(">> test_sol:", test_sol)

        assert_line = assert_line.replace(keyword_func, entry_point+"(")

        assertion = f"\n" + assert_line + ", f'the correct result is {" + test_sol + "} but the function output is {" + entry_point + "(" + test_case + ")}' "
        if verbose > 2: print("assertion:", assertion)
        if failsafe == 1 : failsafe_assertion = assertion
        # I'm using the second test case bc the first is often the null example

        if entry_point+"("+test_case+")" in prompt:
            if verbose > 0: print("\n>> FOUND test case, i =", failsafe,"\n")
            test_code += assertion
        #else: print(">> no test case, i =", failsafe)

        if failsafe >= 50:
            print("\nWARNING: exit loop failsafe activated!")
            break
        failsafe += 1

    # in some cases the examples in the prompt are not in the test
    # to avoid an empty test, I'll add the first assertion
    if test_code == "":
        print("\nWARNING: failsafe due to empty test_code")
        test_code = failsafe_assertion

    print("\n--------\n--------\n>>test_code:", test_code)

    return test_code


def convert_to_json(task_id, response, code, **kwargs):
    json_sample = {
        "task_id": task_id,
        "completion": response, # backup the model's answer
        "code": code,
        **kwargs
    }
    return json_sample