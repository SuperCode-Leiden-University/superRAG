# libraries
import torch
import ast
from datasets import load_dataset # load datasets from Hugging Face

#from human_eval.data import read_problems

# importing my functions from other files
from supercode.src.agent import Agent
from supercode.src.tools.code_processing import *
from supercode.src.tools.tools import *

# importing variables from the config file
from supercode.src.configs.parse_config import *
from supercode.src.configs.system_prompts import baseline_prompt, benchmark_prompt


# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
def run_benchmark(agent, model_id, benchmark_args):
    # load vars
    benchmark_path = benchmark_args["name"]
    bench_name = benchmark_path[benchmark_path.find("/")+1:] # NB: benchmark_path="openai/openai_humaneval")
    baseline = benchmark_args["baseline"]
    bench_subset = benchmark_args["subset"] # optional argument, some benchmarks are divided in subsets
    check_single_task = benchmark_args["check_single_task"]
    num_samples = coder_multi_sampl_args["num_return_sequences"] # this is for parallel sampling

    # print info
    if baseline:
        if verbose>-1: print(f">> benchmark-results baseline and agent for {bench_name}; num_samples = {num_samples}")
    else:
        if verbose>-1: print(f">> benchmark-results agent for {bench_name}; num_samples = {num_samples}")
    if check_single_task is not None: print(f"\n>> WARNING: mode check_single_task, i_task = {check_single_task}\n")

    if model_iter>1 : x_iter=f"_x{model_iter}"
    else: x_iter="" # model iterations

    if bench_subset is not None : bench_name = f"{bench_name}_{bench_subset}"

    # ----------------------------------------------------------------------------------------------
    # files names
    file_info = f"{bench_dir}/{bench_name}_{model_id[model_id.find("/")+1:]}_{num_samples}{x_iter}"

    baseline_file  = f"{file_info}_baseline.jsonl"
    benchmark_file = f"{file_info}_benchmark.jsonl"
    baseline_all_file  = f"{file_info}_baseline_all.jsonl"
    benchmark_all_file = f"{file_info}_benchmark_all.jsonl"

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
        if verbose>0: print(f">> reading jsonl for {baseline_file}...")
        try:
            with open(baseline_file) as f:
                for i, line in enumerate(f, start=1):
                    try: # read the file line-by-line
                        baseline_samples.append(ast.literal_eval(line))
                    except Exception as e:
                        raise Exception(f">> Error in the baseline file at line {i}\nLine content{repr(line)}\n{e}")
        except Exception as e:
            print(">> Error with opening baseline file:", e)
        if len(baseline_samples) != n_tasks:
            print(f">> WARNING: baseline file does not match the expected number of samples: n_baseline={len(baseline_samples)} while n_samples={n_tasks}")
            print(">> Evaluating baseline first")
            baseline=True

    # ----------------------------------------------------------------------------------------------
    # create an empty file (or overwrite if the file exists)
    if baseline:
        with open(baseline_file, "w") as f: f.close()
        with open(baseline_all_file, "w") as f: f.close()
    with open(benchmark_file, "w") as f: f.close()
    with open(benchmark_all_file, "w") as f: f.close()

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
            if verbose>0: print(".....................................\n>> calling the agent in baseline mode")
            index, completions_results = agent.call(
                f"{baseline_prompt}\n\n{prompt}",
                reset_memory=True, baseline=True
            )
            # saving all the results (the best one is saved in a separate file)
            for i, result in enumerate(completions_results):
                save_completion(task_id, **result, filepath=baseline_all_file)
                if i == index: save_completion(task_id, **result, filepath=baseline_file)

            # use the best result as baseline
            baseline_code = completions_results[index]["code"]
            language = completions_results[index]["language"]
            compiler_output = completions_results[index]["compiler_output"]

        else:
            if verbose>0: print(">> loading baseline code from files")
            # retrieve the baseline function (best result from baseline)
            baseline_code = baseline_samples[i]["code"]
            language = baseline_samples[i]["language"]
            compiler_output = baseline_samples[i]["compiler_output"]

        # ---------------------------------------------------------------
        # then ask the model to improve the baseline
        if verbose>0: print("\n.....................................\n>> calling the agent")
        index, completions_results = agent.call(
            f"{benchmark_prompt}\n\n{prompt}",
            baseline_code=baseline_code, code_language=language, baseline_output=compiler_output,
            reset_memory=True, baseline=False
        )
        # saving all the results (the best one is saved in a separate file)
        for i, result in enumerate(completions_results):
            save_completion(task_id, **result, filepath=benchmark_all_file)
            if i == index: save_completion(task_id, **result, filepath=benchmark_file)  # best result
        # ---------------------------------------------------------------

        if check_single_task is not None: break # stop before saving a single datapoint




