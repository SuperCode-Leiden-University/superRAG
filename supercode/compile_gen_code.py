print("start...")
import numpy as np
import subprocess, os
import ast
import tempfile

# importing my functions from other files
from supercode.src.tools.code_processing import *
from supercode.src.tools.tools import *

# importing variables from the config file
from supercode.src.configs.parse_config import *

# importing the dataset from hugging face
from datasets import load_dataset # load datasets from Hugging Face
from human_eval.data import write_jsonl, read_problems


#print("current dir:", os. getcwd())

# load vars
benchmark_path = benchmark_args["name"]
bench_name = benchmark_path[benchmark_path.find("/") + 1:]  # NB: benchmark_path="openai/openai_humaneval")
baseline = benchmark_args["baseline"]
bench_subset = benchmark_args["subset"]  # optional argument, some benchmarks are divided in subsets
model_list = benchmark_args["model_list"]
num_samples = coder_multi_sampl_args["num_return_sequences"]  # this is for parallel sampling

# ----------------------------------------------------------------------------------------------
# # paths
# bench_name = "humaneval"
# bench_subset = None # "python" #
# if bench_subset is not None: bench_name = f"{bench_name}_{bench_subset}"
# 
# num_samples = 1 # sequential samplings, num_return_sequences in config is parallel sampling
# x_iter = "" #"_x3" # see model_iter in config
# model_name = "Qwen2.5-Coder-3B-Instruct" # "Qwen3-4B" # "codegemma-7b-it" # "CodeLlama-7b-Instruct-hf" # "Qwen3-Coder-Next" # "Qwen2.5-Coder-3B-Instruct+Qwen3-4B" #

# file names
x_iter = f"_x{model_iter}" if model_iter>0 else ""
file_list = []
for model_name in model_list:
    file_info = f"{bench_dir}/{bench_name}_{model_name[model_name.find("/") + 1:]}_{num_samples}{x_iter}"

    for file_type in ["baseline", "baseline_all", "benchmark", "benchmark_all"]:
        filename = f"{file_info}_{file_type}.jsonl"
        file_list.append(filename)

# ----------------------------------------------------------------------------------------------
# importing the benchmark-results from hugging face
if verbose>0 : print("loading benchmark-results...")
dataset = load_dataset("openai/openai_humaneval")
if verbose>0 : print("reading the problems...")
problems = read_problems()

performance_plot_vals = []
performance_plot_stds = []

for file_name in file_list:
    # initializing vars
    comp_pass = [ 0 for i in range(num_samples) ]
    comp_fail = [ 0 for i in range(num_samples) ]
    test_pass = [ 0 for i in range(num_samples) ]
    test_fail = [ 0 for i in range(num_samples) ]
    samples = []

    print("\n-----------------------------------------------------------------\nEvaluating: "+file_name)
    with open(file_name) as f:
        for i, line in enumerate(f, start=1):
            try:
                samples.append(ast.literal_eval(line))
            except Exception as e:
                raise Exception("Error on line", i, "Line content:", repr(line))

    for i, sample in enumerate(samples):
        task_id = sample["task_id"]
        completion = sample["completion"]
        code = sample["code"]
        compiler_output = sample["compiler_output"]

        if verbose>0 : print("\n------------------------------------------------------------\n### sample:", task_id)
        #print(f"completion:\n{completion}\ncode:\n{code}\ncompiler_output:\n{compiler_output}\n------------------------------------------------------------\n")

        #function = sample["code"] #; print("code: \n```\n", function, "\n```\n", sep='')
        language, function = extract_code(completion)
        if verbose>2: print(f"\nlanguage: {language}, code:\n{function}\n")

        entry_point = problems[task_id]["entry_point"] # name of the function
        test = problems[task_id]["test"]
        j = i % num_samples # for calculating the variance

        # check just the code
        returncode, output, runtime = sandboxed_compiler(function, language)
        print(f"compiler: returncode = {returncode}\n{output}") # if verbose>2 :
        if returncode==0: comp_pass[j]+=1
        else: comp_fail[j]+=1

        # check with test cases
        returncode, output, runtime = sandboxed_compiler(function, language, test, entry_point)
        if verbose>1 : print(f"test: returncode = {returncode}\n{output}")
        if returncode==0: test_pass[j]+=1
        else: test_fail[j]+=1

    comp_pass_tot = sum(comp_pass) ; comp_pass_std = np.var(comp_pass)
    comp_fail_tot = sum(comp_fail) ; comp_fail_std = np.var(comp_fail)
    test_pass_tot = sum(test_pass) ; test_pass_std = np.var(test_pass)
    test_fail_tot = sum(test_fail) ; test_fail_std = np.var(test_fail)
    tot_samples = len(samples)

    # final summary
    print("===========================================")
    print(f"compiler pass:\t{comp_pass_tot}\t({comp_pass_tot/tot_samples*100:.2f}% +/- {comp_pass_std/tot_samples*100:.2f}%)\t{comp_pass}")
    print(f"compiler fail:\t{comp_fail_tot}\t({comp_fail_tot/tot_samples*100:.2f}% +/- {comp_fail_std/tot_samples*100:.2f}%)\t{comp_fail}")
    print("-------------------------------------------")
    print(f"test pass:   \t{test_pass_tot}\t({test_pass_tot/tot_samples*100:.2f}% +/- {test_pass_std/tot_samples*100:.2f}%)\t{test_pass}")
    print(f"test fail:   \t{test_fail_tot}\t({test_fail_tot/tot_samples*100:.2f}% +/- {test_fail_std/tot_samples*100:.2f}%)\t{test_fail}")
    print("-------------------------------------------")
    print(f"Total samples:\t{tot_samples}")
    print("===========================================")

    # for the plot: baseline(ex, test)+bench(ex, test), variance
    performance_plot_vals.extend([
        round(comp_pass_tot/tot_samples*100, 2),
        round(test_pass_tot/tot_samples*100, 2)
    ])
    performance_plot_stds.extend([
        round(comp_pass_std/tot_samples*100, 2),
        round(test_pass_std/tot_samples*100, 2)
    ])

print(f"vals & stds: \n{performance_plot_vals},\n{performance_plot_stds}\n")
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