# libraries
import torch  # pretty-print for dict
import ast
from datasets import load_dataset # load datasets from Hugging Face

from human_eval.data import read_problems

# my packages
from src.model import Model
from src.tools.code_processing import *

# importing variables from the config file
from src.configs.parse_config import verbose, gen_code_dir, model_id

""" bash command for lm-eval (python_file=supercode/src/code_processing.py,)
python -m lm_eval \
  --model supercode_model \
  --model_args model_path=supercode/src/model.py \
  --tasks humaneval \
  --batch_size 1 \
  --num_fewshot 0 \
  --write_out \
  --output_path supercode/src/docker_env/results
"""
""" bash command to run docker and then evaluate the humaneval benchmark 
docker compose -f src/docker_env/sandbox_code.yml run --rm sandbox_code python -m human_eval.evaluate_functional_correctness samples_humaneval.jsonl

Note: this assumes that results are saved in src/docker_env/results/samples_humaneval.jsonl
"""




# ---------------------------------------------------------------------------------------------- #
# check if I have an Nvidia GPU on the machine
if verbose>1 :
    print("Is cuda available?", torch.cuda.is_available())
    #print(torch.cuda.device_count(), torch.cuda.get_device_name(0), sep="\n")

# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
##### BENCHMARK SETTINGS
baseline = False
# if baseline is true then the model does not use any external info
# if it is false, it checks first if it can find the baseline solution and then ask the model to improve it

# importing the benchmark from hugging face
dataset = load_dataset("openai/openai_humaneval"); bench_name = "humaneval"
# dataset = load_dataset("bigcode/crosscodeeval")["test"]; bench_name = "crosscodeeval"

if baseline:
    print("benchmark baseline for "+bench_name)
    general_prompt = "write a function based on the following description and use assert to check that the function returns the expected results for the examples provided.\n"
    benchmark_file = gen_code_dir + "/"+bench_name+"_baseline-" + model_id[model_id.find("/") + 1:] + ".jsonl"
else:
    print("benchmark model for "+bench_name)
    general_prompt = "improve this function based on the following description and use assert to check that the function returns the expected results for the examples provided.\n"
    baseline_file = gen_code_dir + "/"+bench_name+"_baseline-" + model_id[model_id.find("/") + 1:] + ".jsonl"
    benchmark_file = gen_code_dir+"/"+bench_name+"_patch-"+model_id[model_id.find("/")+1:]+".jsonl"

# ----------------------------------------------------------------------------------------------
# "model" is for processing text and generating an answer
model = Model()

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
try:
    # standard HumanEval code
    problems = read_problems()
    n_tasks = len(problems)
    num_samples_per_task = 5 #200

    # create an empty file (or overwrite if the file exists)
    with open(benchmark_file, "w") as f:
        f.close()

    if not baseline: # load the baseline functions
        baseline_funcs = []
        print("reading jsonl for " + baseline_file + "...")
        with open(gen_code_dir + "/" + baseline_file + ".jsonl") as f:
            for i, line in enumerate(f, start=1):
                try:
                    baseline_funcs.append(ast.literal_eval(line))
                except Exception as e:
                    print("Error on line", i)
                    print("Line content:", repr(line))
                    raise Exception("jsonl file is not written correctly")
        if len(baseline_funcs) != num_samples_per_task*n_tasks:
            print("n_baseline:", len(baseline_funcs), "\nn_samples:", num_samples_per_task*n_tasks)
            raise Exception("baseline file does not match the expected number of samples")

    for i, task_id in enumerate(problems):
        for j in range(num_samples_per_task):
            sample = problems[task_id]

            if baseline:
                response = model.call(general_prompt+"\n\n"+sample["prompt"], reset_memory=True, baseline=baseline)
            else:
                # first retrieve the baseline function, then ask the model to improve it
                baseline_func = baseline_funcs[i*(j+1)]
                response = model.call(general_prompt+"\n\n"+sample["prompt"]+"\n\n"+baseline_func, reset_memory=True, baseline=baseline)

            # generate the answer for all task independently
            code = extract_code(response, sample["entry_point"])
            json_sample = convert_to_json(sample["task_id"], response, code)

            # save as jsonl (json line: json objects separated by newline characters)
            with open(benchmark_file, "a") as f:
                f.write(str(json_sample) + "\n")
                f.close()
    """
    samples = [
        dict(
            task_id=task_id, # info on a single problem
            completion=extract_code(problems[task_id], model.call(problems[task_id]["prompt"], memory=False))
        )
        for task_id in problems
        for _ in range(num_samples_per_task)
    ]
    write_jsonl(gen_code_dir+"/samples.jsonl", samples) # basically converts the dictionary to a 'real' jsonl and save it
    """
    #print("\n\nevaluationg samples:")
    #subprocess.run(["evaluate_functional_correctness " + benchmark_file], shell=True)
except Exception as e:
    print(f"\nAn error occurred:\n{e}\n")
