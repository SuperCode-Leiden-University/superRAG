# libraries
import torch, subprocess, pprint # pretty-print for dict
from datetime import datetime
from datasets import load_dataset # load datasets from Hugging Face

from human_eval.data import write_jsonl, read_problems

# my packages
from supercode.model import Model
from supercode.code_processing import extract_code

# importing variables from the config file
from supercode.configs.parse_config import verbose, gen_code_dir

""" bash command for lm-eval (python_file=src/supercode/code_processing.py,)
python -m lm_eval \
  --model supercode_model \
  --model_args model_path=src/supercode/model.py \
  --tasks humaneval \
  --batch_size 1 \
  --num_fewshot 0 \
  --write_out \
  --output_path src/supercode/docker_env/results
"""
""" bash command to run docker and then evaluate the humaneval benchmark 
docker compose -f supercode/docker_env/sandbox_code.yml run --rm sandbox_code python -m human_eval.evaluate_functional_correctness samples_humaneval.jsonl

Note: this assumes that results are saved in supercode/docker_env/results/samples_humaneval.jsonl
"""




# ---------------------------------------------------------------------------------------------- #
# check if I have an Nvidia GPU on the machine
if verbose>1 :
    print("Is cuda available?", torch.cuda.is_available())
    #print(torch.cuda.device_count(), torch.cuda.get_device_name(0), sep="\n")


# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
##### IMPORTING THE MODEL & BENCHMARK

# "model" is for processing text and generating an answer
model = Model()

# importing the benchmark from hugging face
dataset = load_dataset("openai/openai_humaneval")
# dataset = load_dataset("bigcode/crosscodeeval")["test"]

print("Dataset structure:\n", dataset, sep="")
""" dataset looks like this:
    DatasetDict({
        test: Dataset({
            features: ['task_id', 'prompt', 'canonical_solution', 'test', 'entry_point'],
            num_rows: 164
        })
    })
"""
test_set = dataset["test"]
L_test = len(test_set)
sample = test_set[0]
#print("\nsample 0:"); pprint.pprint(sample)


# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
##### BENCHMARK THE MODEL
for i in range(L_test):
    print("exemple ", i, "/", L_test, sep ="" )
    """
    sample = test_set[i]
    prompt = sample["prompt"]
    sol = sample["canonical_solution"]

    def_start = prompt.find("def")
    def_end = prompt.find("\n", def_start)
    func_def = prompt[def_start:def_end]
    print("func_def:", func_def)
    #"""
    try:
        # standard HumanEval code
        problems = read_problems()
        num_samples_per_task = 2 #200

        samples = [
            dict(task_id=task_id, completion=model.call(problems[task_id]["prompt"]))
            for task_id in problems
            for _ in range(num_samples_per_task)
        ]
        write_jsonl(gen_code_dir+"/samples.jsonl", samples)
        """
        # my code
        response = model.call(prompt)

        # convert to json
        json_sample = extract_code(sample, response)
        print("\n\njson_sample", json_sample, sep="\n")

        # save as jsonl (json line: json objects separated by newline characters)
        with open(gen_code_dir+"/humaneval.jsonl", "a") as f:
            f.write(str(json_sample)+"\n\n")
            f.close()
        #"""
        print("\n\nevaluationg samples:")
        subprocess.run(["evaluate_functional_correctness "+gen_code_dir+"/samples.jsonl"], shell=True)
    except Exception as e:
        print(f"\nAn error occurred:\n{e}\n")
    break