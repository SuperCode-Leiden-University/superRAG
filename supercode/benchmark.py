# libraries
import torch
import ast
from datasets import load_dataset # load datasets from Hugging Face

#from human_eval.data import read_problems

# my packages
from supercode.src.agent import Agent
from supercode.src.tools.code_processing import *
from supercode.src.tools.tools import *

# importing variables from the config file
from supercode.src.configs.parse_config import *
from supercode.src.configs.system_prompts import baseline_prompt, benchmark_prompt


# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
def run_benchmark(model, benchmark_args):
    # load vars
    benchmark_path = benchmark_args["name"]
    bench_name = benchmark_path[benchmark_path.find("/")+1:] # NB: benchmark_path="openai/openai_humaneval")
    baseline = benchmark_args["baseline"]
    bench_subset = benchmark_args["subset"] # optional argument, some benchmarks are divided in subsets
    check_single_task = benchmark_args["check_single_task"]
    num_samples_per_task = 1 # sequential samplings, num_return_sequences in config is parallel sampling

    # print info
    if baseline: print("benchmark-results baseline and model for", bench_name, "; num_samples_per_task =", num_samples_per_task)
    else: print("benchmark-results model for", bench_name, "; num_samples_per_task =", num_samples_per_task)
    if check_single_task is not None: print("WARNING: mode check_single_task, i_task =", check_single_task)

    if model_iter>1 : iter="_x"+model_iter
    else: iter=""

    if bench_subset is not None : bench_name = bench_name+"_"+bench_subset

    # ----------------------------------------------------------------------------------------------
    # files names
    baseline_file  = bench_dir+"/"+bench_name+"_"+coder_model_id[coder_model_id.find("/")+1:]+"_baseline_" +str(num_samples_per_task)+iter+".jsonl"
    benchmark_file = bench_dir+"/"+bench_name+"_"+coder_model_id[coder_model_id.find("/")+1:]+"_benchmark_"+str(num_samples_per_task)+iter+".jsonl"

    # ----------------------------------------------------------------------------------------------
    # importing the benchmark-results from hugging face,
    if bench_subset is None: dataset = load_dataset(benchmark_path)
    else: dataset = load_dataset(benchmark_path, bench_subset)
    # add the second argument when the dataset has multiple subsets

    problems = dataset["test"] ; n_tasks = len(problems)
    #print("Dataset structure:\n", dataset, sep="")
    """ dataset looks like this:
        DatasetDict({
            test: Dataset({
                features: ['task_id', 'prompt', 'canonical_solution', 'test', 'entry_point'],
                num_rows: 164
            })
        })

    # IMPORTANT: some tasks (47, 163) in humaneval contain mismatches or errors!
    # /47: has an error in the prompt, it should be: median([-10, 4, 6, 1000, 10, 20]) = 8.0 (instead of 15.0)
    # /163: unclear prompt, should specify to return only digits that are between min(0,a,b) and max(a,b,9)

    """

    # ---------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------- #
    # first check if the results for the baseline are available, then load them for later
    if not baseline:  # load the baseline functions
        baseline_samples = []
        print("reading jsonl for " + baseline_file + "...")
        try:
            with open(baseline_file) as f:
                for i, line in enumerate(f, start=1):
                    try: # read the file line-by-line
                        baseline_samples.append(ast.literal_eval(line))
                    except Exception as e:
                        print("Error on line", i)
                        print("Line content:", repr(line))
                        raise Exception("Error in the baseline file: \n",e)
        except Exception as e:
            print("Error with baseline file:", e)
        if len(baseline_samples) != n_tasks:
            print("WARNING: baseline file does not match the expected number of samples!\nn_baseline:", len(baseline_samples), "\nn_samples:", n_tasks)
            print("Evaluating baseline first")
            baseline=True

    # ----------------------------------------------------------------------------------------------
    # create an empty file (or overwrite if the file exists)
    if baseline:
        with open(baseline_file, "w") as f: f.close()
    with open(benchmark_file, "w") as f: f.close()

    # ----------------------------------------------------------------------------------------------
    for i, sample in enumerate(problems):
        if check_single_task is not None:
            i=int(check_single_task)
            sample = problems[i]
        # extract task
        task_id = sample["task_id"]
        entry_point = sample["entry_point"]
        prompt = sample["prompt"]
        test = sample["test"] # code to test the function
        print("\n======================================")
        print(i, task_id)

        # manual fix for error in humaneval prompt...
        if bench_name=="openai_humaneval" and i == 47 : prompt = prompt.replace("15.0", "8.0")


        # ----------------------------------------------------------------------------------------------
        if baseline: # create the baseline
            baseline_response = model.call(
                baseline_prompt+"\n\n"+prompt,
                reset_memory=True, baseline=True
            )
            language, baseline_code = extract_code(baseline_response)
            save_completion(task_id, baseline_response, filepath=baseline_file)

        else:
            # retrieve the baseline function
            baseline_code = baseline_samples[i]["code"]

        # ---------------------------------------------------------------
        # then ask the model to improve the baseline
        response = model.call(
            benchmark_prompt+"\n\n"+prompt, code=baseline_code,
            reset_memory=True, baseline=False
        )
        #language, code = extract_code(response)
        save_completion(task_id, response, filepath=benchmark_file)
        # ---------------------------------------------------------------

        if check_single_task is not None: break # stop before saving a single datapoint




