import logging, os
logfile = os.path.join(os.path.dirname(__file__), "lm_eval_import.log")
logging.basicConfig(filename=logfile, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger(__name__).info("START module import of code_processing.py")

from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model, model_registry

from supercode.model import Model
logging.getLogger(__name__).info("END module import")


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

""" To integrate a custom model with lm-eval-harness, 
I need a class that inherits from `lm_eval.api.model.LM` and it need 3 methods:
 - `generate_until(self, requests)` --> used for code‑generation tasks like HumanEval.
 - `loglikelihood(self, requests)` --> used for tasks that score text (multiple‑choice, QA, etc.).
 - `loglikelihood_rolling(self, requests)` --> used for perplexity‑style tasks.
I only care about code generation, so I can implement the first and raise `NotImplementedError` for the others.
"""

print(os.getcwd())
print("LOADING supercode.code_processing")


# Only register if not already present
if "supercode_model" not in model_registry._objs:
    @register_model("supercode_model")
    class SuperCodeLM(TemplateLM):
        def __init__(self, **kwargs):
            super().__init__()
            print("start __init__")
            self.model = Model()
            print("ended __init__")

        def generate_until(self, requests):
            print("start generate_until")

            outputs = []
            for req in requests:
                prompt = req["prompt"]
                max_new_tokens = req.get("max_gen_toks", 256) # limit number of tokens
                stop = req.get("until", None) # HumanEval expects the model to generate only the function body

                response = self.model.call(prompt, max_new_tokens=max_new_tokens, stop=stop)
                outputs.append(response)
            print("end generate_until")
            return outputs

        def loglikelihood(self, requests):
            raise NotImplementedError("Not needed for HumanEval")

        def loglikelihood_rolling(self, requests):
            raise NotImplementedError("Not needed for HumanEval")
else:
    # If already registered, import the existing class for local use (optional)
    SuperCodeLM = model_registry._objs["supercode_model"]
logging.getLogger(__name__).info("END class def (SuperCodeLM)")


def create_model(**kwargs):
    print("start create_model")
    return SuperCodeLM(**kwargs)

logging.getLogger(__name__).info("END func def (create_model)")


# extract the code from the model's answer
def extract_code(response, entry_point=None):
    if entry_point == None: code_def = response.find("def ") # if entry_point is unknown
    else: code_def = response.find("def "+entry_point) # robust signature for finding the function

    if code_def==-1:
        print("function not found")
        return ""
    code_start = response.rfind("```", 0, code_def)+3
    if response.rfind("python", code_start, code_def)>-1:
        code_start += 6 # remove "python" as well if present
    code_end = response.find("```", code_start)  # code blocks start and end with ``` (exclude ```)

    #code_start = response.rfind("def", 0, entry_index) # this works for functions, but not for classes
    #code_end = response.find("```", code_start) # code blocks start and end with ``` (exclude ```)

    code = response[code_start:code_end].strip()
    return code


def convert_to_json(task_id, response, code):
    json_sample = {
        "task_id": task_id,
        "completion": response, # backup the model's answer
        "code": code
    }
    return json_sample