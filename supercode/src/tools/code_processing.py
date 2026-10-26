import re
import json
from pprint import pp as pprint

# importing my functions from other files
from supercode.src.tools.tools import *

# importing variables from the config file
from supercode.src.configs.parse_config import *


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


def extract_test_code(prompt, code):
    if verbose>0: print(">> extracting test code")

    test_code = ""
    failsafe_assertion = ""
    start = 0
    failsafe = 0
    keyword_start = "assert"    ; len_start = len(keyword_start)
    keyword_end_line = "\n"     #; len_end = len(keyword_end_line)

    test_start = code.find(keyword_start)
    func_code = code[:test_start]
    test = code[test_start:]

    func_name_start = code.find("def ")+4
    func_name_end = code.find("(", func_name_start)
    entry_point = code[func_name_start:func_name_end]
    keyword_func = f"{entry_point}(" ; len_func = len(keyword_func)

    next_start = test.find(keyword_start, start+len_start)


    while next_start>-1:
        if verbose>1: print("\n--------\n>> looping", failsafe)

        func = test.find(keyword_func, next_start) #+ len_func
        start = test.rfind(keyword_start, next_start, func) #+ len_start
        end_line = test.find(keyword_end_line, start)
        next_start = test.find(keyword_start, start+len_start)

        if verbose>1: print(f">> start = {start}; func = {func}; end_line = {end_line}; next_start = {next_start}")

        assert_line = test[start:end_line]
        if verbose>0: print(f">> assert_line:{assert_line}")

        end_assert = max(
            assert_line.rfind("= ")+2,
            assert_line.rfind("> ")+2,
            assert_line.rfind("< ")+2,
        )

        test_case = assert_line[
            assert_line.find(keyword_func)+len_func :
            assert_line.rfind(")", 0, end_assert)
        ]
        if verbose>1: print(f">> test_case: {test_case}")

        test_sol = assert_line[end_assert:]
        if verbose>1: print(f">> test_sol: {test_sol}")

        assert_line = assert_line.replace(keyword_func, entry_point+"(")

        assertion = f"\n"+assert_line+", f'the correct result is {"+test_sol+"} but the function output is {"+entry_point+"("+test_case+")}' "
        if verbose>2: print(f">> assertion: {assertion}")
        if failsafe == 1 : failsafe_assertion = assertion
        # I'm using the second test case bc the first is often the null example

        if f"{entry_point}({test_case})" in prompt:
            if verbose>0: print(f"\n>> FOUND test case, i = {failsafe}\n")
            test_code += assertion
        #else: print(">> no test case, i =", failsafe)

        if failsafe >= 50:
            print("\n>> WARNING: exit loop failsafe activated!")
            break
        failsafe += 1

    # in some cases the examples in the prompt are not in the test
    # to avoid an empty test, I'll add the first assertion
    if test_code == "":
        print("\n>> WARNING: failsafe due to empty test_code")
        test_code = failsafe_assertion

    test_code += '\nprint("end of the code")'
    print(f"\n--------\n--------\n>> test_code:\n{test_code}")

    return f"{func_code}\n{test_code}"


def extract_code(completion_text):
    """
    Extracts programming language and code from a code block in the completion text.
    Returns (language, code) or (None, None) if no code block found.
    """
    # Regex to match fenced code blocks: ```language\ncode\n```
    pattern = r'```(\w*)\s(.*?)(\s```|.$)' # .$ takes into account the case where the models doesn't finish
    match = re.search(pattern, completion_text, re.DOTALL)
    if match:
        language = match.group(1).strip()
        code = match.group(2).strip()
        return language, code
    else:
        print(">> WARNING: no code was found!")
        return None, None


def save_completion(task_id, completion, language, code, compiler_output, filepath):
    # "language": language, "code": code, "compiler_output": result
    if verbose>0: print(">> saving response from model")

    # extract language and code from 'completion'
    completion_task = {
        "task_id": task_id,
        "completion": completion,
        "language": language,
        "code": code,
        "compiler_output": compiler_output
    }
    if verbose>2: print("\n>> completion:") ; pprint(completion_task)
    # save to JSONL (one JSON object per line)
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(completion_task, ensure_ascii=False)+'\n')
        f.close()


def select_response(responses):
    # # if single answer
    # if len(responses)==1:
    #     language, code = extract_code(responses[0])
    #     result = sandboxed_compiler(code, language)
    #     return responses[0]
    # 
    # # if multi-sampling
    # else:
    if verbose>0: print(">> selecting the best out of", len(responses), "responses")

    # filter for correctness (unit tests from prompt only)
    default_idx = 0 # aka the first sample
    correct_idx = []
    correct_runtimes = []
    code_results = []
    
    for i, response in enumerate(responses):
        language, code = extract_code(response)
        if code==None and i==default_idx: default_idx+=1 # in case the code is empty
        result = sandboxed_compiler(code, language, k_times=k_runtimes)
        # returns (returncode, "output: ...", runtimes_dict)
        # where runtimes_dict = {"real": (mean, std), "user": ..., "sys": ..., "cpu": ..., "util": ...}

        # collect code and compiler output for later
        code_results.append({"language": language, "code": code, "compiler_output": result})

        if result[0] == 0 : # if the code is correct
            correct_idx.append(i)
            correct_runtimes.append(result[2]) # code output and errors

    if default_idx>=len(responses): default_idx=0 # failsafe in the case where all codes are empty

    # if none are correct, select the first non-empty code
    if not correct_idx:
        if verbose>0: print(">> no correct response was found, returning the first not empty code\n")
        return default_idx, code_results
    
    # if only one answer is correct
    elif len(correct_idx)==1:
        if verbose>0: print(f">> only one correct response was found: i={correct_idx[0]}\n")
        return correct_idx[0], code_results
    
    # if more then one answer is correct, select the fastest
    else:
        # runtime = {"real", "user", "sys", "cpu", "util"}, they are all in seconds except "util" which is cpu/real
        if verbose>0: print(f">> using {optim_metric} as runtime metric\n")
        metric_times = [ runtime[optim_metric][0] for runtime in correct_runtimes ] # extract runtime (mean, std) for the selected metric

        if optim_metric == "util": # max cpu utilization
            i = metric_times.index(max(metric_times)) # find the code with the min runtime or max utilization
        else: # min runtime
            i = metric_times.index(min(metric_times)) # find the code with the min runtime or max utilization

        if verbose>0: print(f">> i={correct_idx[i]} was the fastest out of {len(correct_idx)} correct responses\n")
        return correct_idx[i], code_results
