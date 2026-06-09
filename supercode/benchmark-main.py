# libraries
import torch  # pretty-print for dict
import ast
from datasets import load_dataset # load datasets from Hugging Face

from human_eval.data import read_problems

# my packages
from src.agent import Agent
from src.tools.code_processing import *
from src.tools.tools import *

# importing variables from the config file
from src.configs.parse_config import *
from src.configs.system_prompts import baseline_prompt, benchmark_prompt

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
baseline = True # True or False
check_single_task = False ; i_task = 101
"""
Analysis of errors:
   #22  due to faulty logic: (26, 64, 75, 81, 91, 102, 108, 116, 120, 123, 126, 127, 129, 130, 132, 134, 139, 145, 147, 157, 162, 163) 
    #4  due to incorrect syntax: (77, 93, 111, 125) 
    #4  mysterious: (85, 122, 148, 160) should be correct but it not considered so???
    #2  due to numerical fluctuations: (2, 32) 
    #1  token limit: (1), extra examples
    #1  silent: (50), only part of the code was saved for some reasons???

Actually fixed: (38)
"""
print("baseline =", baseline, "; check_single_task =", check_single_task, "; i_task =", i_task)

num_samples_per_task = 1 #200
# if baseline is true then the model does not use any external info
# if it is false, it checks first if it can find the baseline solution and then ask the model to improve it

# importing the benchmark from hugging face
dataset = load_dataset("openai/openai_humaneval"); bench_name = "humaneval"
# IMPORTANT: some tasks (47, 163) in HumanEval contain mismatches or errors!
# /47:  2nd example is incorrect and doesn't match the test
# /163: unclear prompt, should specify to return only digits that are between min(0,a,b) and max(a,b,9)
# dataset = load_dataset("bigcode/crosscodeeval")["test"]; bench_name = "crosscodeeval"

if baseline:
    print("benchmark baseline and model for "+bench_name)
else:
    print("benchmark model for "+bench_name)

extra_note = "" #"+Qwen3-4B"
baseline_file  = gen_code_dir+"/"+bench_name+"_baseline-" +model_id[model_id.find("/")+1:]+extra_note+"_"+str(num_samples_per_task)+".jsonl"
benchmark_file = gen_code_dir+"/"+bench_name+"_benchmark-"+model_id[model_id.find("/")+1:]+extra_note+"_"+str(num_samples_per_task)+".jsonl"

# ----------------------------------------------------------------------------------------------
# "model" is for processing text and generating an answer
model = Agent()

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
# standard HumanEval code
problems = read_problems()
n_tasks = len(problems)

if not baseline: # load the baseline functions
    baseline_samples = []
    print("reading jsonl for "+baseline_file+"...")
    try:
        with open(baseline_file) as f:
            for i, line in enumerate(f, start=1):
                try: # read the file line-by-line
                    baseline_samples.append(ast.literal_eval(line))
                except Exception as e:
                    print("Error on line", i)
                    print("Line content:", repr(line))
                    raise Exception("Error in the baseline file!")
    except Exception as e:
        print("Error with baseline file:", e)
    if len(baseline_samples) != num_samples_per_task*n_tasks:
        print("WARNING: baseline file does not match the expected number of samples")
        print("n_baseline:", len(baseline_samples), "\nn_samples:", num_samples_per_task*n_tasks)
        print("Evaluating baseline first")
        baseline=True

# create an empty file (or overwrite if the file exists)
if baseline: # clear the baseline only if
    with open(baseline_file, "w") as f: f.close()
with open(benchmark_file, "w") as f: f.close()


for i, task_id in enumerate(problems):
    if check_single_task: i=i_task; task_id="HumanEval/"+str(i)
    print("\n======================================")
    print(i, task_id)

    sample = problems[task_id]
    entry_point = sample["entry_point"]
    prompt = sample["prompt"]
    # task_id="HumanEval/47" has an error in the prompt, it should be:
    # >>> median([-10, 4, 6, 1000, 10, 20])
    # 8.0 (instead of 15.0)
    if task_id == "HumanEval/47" : prompt = prompt.replace("15.0", "8.0") ; print(prompt)

    for j in range(num_samples_per_task):
        if baseline: # create the baseline
            baseline_response = model.call(baseline_prompt+"\n\n"+prompt, reset_memory=True, baseline=True)
            print("\n>> extracting code from baseline response")
            baseline_code = extract_code(baseline_response, entry_point)
            print("\n>> checking compiler output for baseline response")
            baseline_compiler_output = sandboxed_compiler(baseline_code)
            baseline_json_sample = convert_to_json(task_id, baseline_response, baseline_code, compiler_output=baseline_compiler_output)

            with open(baseline_file, "a") as f:
                f.write(str(baseline_json_sample) + "\n")
                f.close()
        else:
            # retrieve the baseline function
            baseline_code = baseline_samples[i*(j+1)]["code"]

        # then ask the model to improve the baseline
        response = model.call(benchmark_prompt+"\n\n"+prompt, code=baseline_code, reset_memory=True, baseline=False)
        print("\n>> extracting code from response")
        code = extract_code(response, entry_point)
        print("\n>> checking compiler output for response")
        compiler_output = sandboxed_compiler(code)
        json_sample = convert_to_json(task_id, response, code, compiler_output=compiler_output)

        # save as jsonl (json line: json objects separated by newline characters)
        # except, I'd need to convert to json first, so this is actually a list of python dictionaries...
        with open(benchmark_file, "a") as f:
            f.write(str(json_sample) + "\n")
            f.close()

    if check_single_task: break
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
print("\n\nevaluationg samples:")
subprocess.run(["evaluate_functional_correctness " + benchmark_file], shell=True)
"""

