import re
from human_eval.data import write_jsonl, read_problems


# extract the code from the model's answer
def extract_code(response):
    # extract the first code block if present
    code_block = re.findall(r"```(.*?)```", response, re.DOTALL)
    if code_block:
        return code_block[0].strip()

    # otherwise, try to extract the first function definition
    func_match = re.search(r"def\s+\w+\(.*", response, re.DOTALL)
    if func_match:
        return response[func_match.start():].strip()

    # fallback: return raw text
    return response.strip()


def convert_to_json(response):
    problems = read_problems()
    samples = []

    for task_id, problem in problems.items():
        code = extract_code(response)

        samples.append({
            "task_id": task_id,
            "completion": code
        })

    write_jsonl("samples.jsonl", samples)
