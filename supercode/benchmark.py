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
    model_name = model_id[model_id.find("/")+1:]

    # print info
    if baseline:
        if verbose>-1: print(f">> benchmark-results baseline and agent for {bench_name}; num_samples = {num_samples}")
    else:
        if verbose>-1: print(f">> benchmark-results agent for {bench_name}; num_samples = {num_samples}")
    if check_single_task is not None: print(f"\n>> WARNING: mode check_single_task, i_task = {check_single_task}\n")

    # ----------------------------------------------------------------------------------------------
    # files names
    # if model_iter>1 : x_iter=f"_x{model_iter}"
    # else: x_iter="" # model iterations
    # if bench_subset is not None : bench_name = f"{bench_name}_{bench_subset}"
    # model_name = model_id[model_id.find("/")+1:]
    # file_info = f"{bench_dir}/{bench_name}_{model_name}_{num_samples}{x_iter}"
    #
    # baseline_file  = f"{file_info}_baseline.jsonl"
    # benchmark_file = f"{file_info}_benchmark.jsonl"
    # baseline_all_file  = f"{file_info}_baseline_all.jsonl"
    # benchmark_all_file = f"{file_info}_benchmark_all.jsonl"

    baseline_file, benchmark_file = filenames(model_name, bench_name, bench_subset, num_samples)

    # ----------------------------------------------------------------------------------------------
    # importing the benchmark-results from hugging face,
    if bench_subset is None: dataset = load_dataset(benchmark_path)
    else: dataset = load_dataset(benchmark_path, bench_subset)
    # add the second argument when the dataset has multiple subsets

    problems = dataset[benchmark_args["split"]] ; n_tasks = len(problems)
    id_key = benchmark_args["id_key"] ; prompt_key = benchmark_args["prompt_key"]
    #print("\nDataset structure:\n", dataset, sep="")


    # ---------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------- #
    # first check if the results for the baseline are available, then load them for later
    if not baseline:  # load the baseline functions
        baseline_samples = get_samples(baseline_file)
        # baseline_samples = []
        # if verbose>0: print(f">> reading jsonl for {baseline_file}...")
        # try:
        #     with open(baseline_file) as f:
        #         for i, line in enumerate(f, start=1):
        #             try: # read the file line-by-line
        #                 baseline_samples.append(ast.literal_eval(line))
        #             except Exception as e:
        #                 raise Exception(f">> ERROR in the baseline file at line {i}:\n{e}")
        # except Exception as e:
        #     print(">> ERROR with opening baseline file:", e)
        if len(baseline_samples) != n_tasks:
            print(f">> WARNING: baseline file does not match the expected number of samples: n_baseline={len(baseline_samples)} while n_samples={n_tasks}")
            print(">> Evaluating baseline first")
            raise ValueError("ERROR in baseline!!!!!")
            baseline=True

    # ----------------------------------------------------------------------------------------------
    # create an empty file (or overwrite if the file exists)
    if baseline:
        with open(baseline_file, "w") as f: f.close()
        # with open(baseline_all_file, "w") as f: f.close()
    with open(benchmark_file, "w") as f: f.close()
    # with open(benchmark_all_file, "w") as f: f.close()

    # ----------------------------------------------------------------------------------------------
    for i, sample in enumerate(problems):
        if check_single_task is not None:
            i=int(check_single_task)
            sample = problems[i]

        # extract task
        #entry_point = sample["entry_point"]
        #test = sample["test"] # code to test the function
        task_id = sample[id_key]
        prompt = sample[prompt_key]
        print("\n======================================")
        print(i, task_id, model_name)
        print(prompt)

        # manual fix for error in humaneval prompt...
        if bench_name=="openai_humaneval" and i == 47 : prompt = prompt.replace("15.0", "8.0")
        # IMPORTANT: task/47: has an error in the prompt, it should be:
        # median([-10, 4, 6, 1000, 10, 20]) = 8.0 (instead of 15.0)

        # ----------------------------------------------------------------------------------------------
        if baseline: # create the baseline
            if verbose>0: print("......................................")
            call_args = {
                "user_prompt": f"{baseline_prompt}\n\n{prompt}",
                "reset_memory": True,
                "baseline": True
            }

            """ index, completions_results = agent.call(**call_args) """
            func_results, agent_resources = resource_tracking(agent.call, **call_args)
            index, completions_results = func_results # index of the best result
            save_completion(task_id, agent_resources, completions_results, filepath=baseline_file)
            """
            # saving all the results (the best one is saved in a separate file)
            for i, result in enumerate(completions_results):
                save_completion(task_id, **result, filepath=baseline_all_file)
                if i == index: save_completion(task_id, **result, filepath=baseline_file)
            """
            if not benchmark_args["use_baseline_best"]: index = 0 # use the first result
            baseline_code = completions_results[index]["code"]
            language = completions_results[index]["language"]
            compiler_output = completions_results[index]["code_output"]

        else:
            if verbose>0: print(">> loading baseline code from files")
            # retrieve the baseline function (best result from baseline)
            baseline_code = baseline_samples[i]["code"]
            language = baseline_samples[i]["language"]
            compiler_output = baseline_samples[i]["code_output"]

            #print(f"baseline_code: \n```{language}\n{baseline_code}\n```\n{compiler_output}")

        # ---------------------------------------------------------------
        # then ask the model to improve the baseline
        if verbose>0: print("\n......................................")
        call_args ={
            "user_prompt" : f"{benchmark_prompt}\n\n{prompt}",
            "baseline_code" : baseline_code, "code_language" : language, "baseline_output" : compiler_output,
            "reset_memory" : True, "baseline" : False
        }
        """ index, completions_results = agent.call(**call_args) """
        func_results, agent_resources = resource_tracking(agent.call, **call_args)
        index, completions_results = func_results
        save_completion(task_id, agent_resources, completions_results, filepath=benchmark_file)
        """
        # saving all the results (the best one is saved in a separate file)
        for i, result in enumerate(completions_results):
            save_completion(task_id, **result, filepath=benchmark_all_file)
            if i == index: save_completion(task_id, **result, filepath=benchmark_file)  # best result
        """
        # ---------------------------------------------------------------

        if check_single_task is not None: break # stop before saving a single datapoint




