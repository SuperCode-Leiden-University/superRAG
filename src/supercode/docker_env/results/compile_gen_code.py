import subprocess
import ast
import tempfile


success_count = 0
failure_count = 0
samples = []

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