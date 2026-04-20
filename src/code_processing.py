from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from src.model import Model


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

@register_model("supercode_model")
class SuperCodeLM(LM):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = Model()

    def generate_until(self, requests):
        outputs = []
        for req in requests:
            prompt = req["prompt"]
            max_new_tokens = req.get("max_gen_toks", 256) # limit number of tokens
            stop = req.get("until", None) # HumanEval expects the model to generate only the function body

            response = self.model.call(prompt, max_new_tokens=max_new_tokens, stop=stop)
            outputs.append(response)
        return outputs

    def loglikelihood(self, requests):
        raise NotImplementedError("Not needed for HumanEval")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("Not needed for HumanEval")


def create_model(**kwargs):
    return SuperCodeLM(**kwargs)


# extract the code from the model's answer
def extract_code(sample, response):

    def_index = response.find(sample["entry_point"]) # robust signature for finding the function
    code_start = response.rfind("```", def_index)+3 # code blocks start and end with ``` (exclude ```)
    code_end = response.find("```", def_index)

    code = response[code_start:code_end].strip()

    sample = {
        "task_id": sample["task_id"],
        "completion": code
    }
    return sample
