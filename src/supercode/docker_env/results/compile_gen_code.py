import subprocess
import ast
import tempfile

from datasets import load_dataset # load datasets from Hugging Face
from supercode.configs.parse_config import verbose, gen_code_dir


success_count = 0
failure_count = 0
samples = []

benchmark_file = gen_code_dir+"/humaneval_patch_results.txt"

# importing the benchmark from hugging face
dataset = load_dataset("openai/openai_humaneval")

with open("humaneval_baseline.jsonl") as f:
    for i, line in enumerate(f, start=1):
        try:
            samples.append(ast.literal_eval(line))
        except Exception as e:
            print("Error on line", i)
            print("Line content:", repr(line))
            raise


for sample in samples:
    #print("### sample:", sample["task_id"])
    function = sample["completion"]

    # extract examples with solutions from the prompt
    #index_start = function.find("def ")+4

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        tmp.write(function)
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            ["python3", tmp_path],
            check=True,
            capture_output=True,
            text=True
        )
        #print(result.stdout)
        success_count += 1
    except Exception as e:
        failure_count += 1
        #print("Execution error:", e)

# Final summary
print("\n==============================")
print("Execution Summary")
print("==============================")
print(f"Total samples: {len(samples)}")
print(f"Successful executions: {success_count} ({success_count/len(samples)*100:.2f}%)")
print(f"Failed executions: {failure_count} ({failure_count/len(samples)*100:.2f}%)")

"""
import json

samples = []
with open("humaneval_baseline.jsonl") as f:
    for i, line in enumerate(f, start=1):
        try:
            obj = json.loads(line)
        except Exception as e:
            print("Error on line", i)
            print("Line content:", repr(line))
            raise

for sample in samples:
    print("\n\n### sample:", sample["task_id"])
    function = sample["completion"]
    try: subprocess.run(["python from typing import List \n"+function], shell=True)
    except Exception as e: print(e)
"""