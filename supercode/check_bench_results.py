print("start...")
import numpy as np
import subprocess, os
import ast
import tempfile
import math
from scipy import stats


# importing my functions from other files
from supercode.benchmark import *
from supercode.src.tools.completion_processing import *
from supercode.src.tools.tools import *

# importing variables from the config file
from supercode.src.configs.parse_config import *

# importing the dataset from hugging face
from datasets import load_dataset # load datasets from Hugging Face



# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
def pass_at_k(n: int, c: int, k: int) -> float:
    """ Calculates the unbiased pass@k metric.
        n: Total number of samples generated for a problem.
        c: Number of those samples that are correct.
        k: The number of samples considered (e.g., 1, 10, 100)
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

def normalize(array, norm=1, string=False):
    array_ave = np.mean(array)
    array_std = np.std(array)
    if string:
        array_str = f"({array_ave*100/norm:.2f}, {array_std*100/norm:.2f})%\t({array_ave} +/- {array_std:.2f})\t{array}"
        return array_ave/norm, array_std/norm, array_str
    else: return array_ave/norm, array_std/norm

def median_std(array):
    return np.median(array), stats.median_abs_deviation(array)



# ----------------------------------------------------------------------------------------------
# load vars
benchmark_path = benchmark_args["name"]
bench_name = benchmark_path[benchmark_path.find("/") + 1:]  # NB: benchmark_path="openai/openai_humaneval")
baseline = benchmark_args["baseline"]
bench_subset = benchmark_args["subset"]  # optional argument, some benchmarks are divided in subsets
model_list = benchmark_args["model_list"]
n_samples = coder_multi_sampl_args["num_return_sequences"]  # this is for parallel sampling

# ----------------------------------------------------------------------------------------------
# importing the benchmark-results from hugging face
if verbose>0 : print(">> reading the problems")
bench_keys = benchmark_keys(benchmark_path)
dataset = load_dataset("openai/openai_humaneval")
problems = dataset[bench_keys["split"]]
n_tasks = len(problems)

tot_samples = n_tasks * n_samples

# ----------------------------------------------------------------------------------------------
if verbose>0 : print(">> loading benchmark-results")

file_list = []
for model_id in model_list:
    model_name = model_id[model_id.find("/")+1:]
    model_files = get_filenames(model_name, bench_name, bench_subset, n_samples) # baseline and benchmark
    file_list.extend(model_files)

performance_plot = {}

for file_path in file_list:
    file_name = file_path[file_path.rfind("/")+1:]

    # llm stats
    llm_runtime = []
    llm_cputime = []
    llm_energy = []
    llm_co2eq = []
    llm_memory = [] # psutil memory is unreliable

    # pass@1
    comp_pass_1 = [ 0 for i in range(n_samples) ]
    test_pass_1 = [ 0 for i in range(n_samples) ]
    # pass@best
    comp_pass_b = 0
    test_pass_b = 0
    # pass@k
    comp_pass_k = []
    test_pass_k = []

    print("\n-----------------------------------------------------------------\n>> "+file_name)
    with open(file_path) as f:
        #print(f">> opened baseline file:\n{file_path}")
        # each line is a json object
        for i, line in enumerate(f, start=1):  # read the file line-by-line
            try:
                # IMPORTANT NOTE:
                # load() expects a file-like object (something with a .read() method, like an open file)
                # loads() expects a string, they're not interchangeable!!!
                task = json.loads(line)
            except ValueError as e:
                raise ValueError(f"\n>> ERROR in the baseline file at line {i}:\n{e}")

            #if i%41 == 0 : print(f">> task: {i}")

            task_id = task["task_id"] #; print(i, task_id)
            samples = task["samples"]
            tests = task["tests"]

            # llm stats
            llm_runtime.append(task["tot_time"]) # in seconds
            llm_cputime.append(task["cpu_time"]) # in seconds
            llm_energy.append(task["energy"]) # in kWh
            llm_co2eq.append(task["emissions"]) # in g CO2eq, calculated as energy*PUE
            llm_memory.append(task["psutil_memory"]) # in MiB

            for j in range(n_samples):
                comp_correct = 0
                test_correct = 0
                sample = samples[j]
                test = tests[j]

                if True: # recheck adding the assert library explicitely for the examples
                    fixed_examples = sandboxed_compiler("#include <assert.h>\n"+str(sample["code"]), sample["language"], k_times=1)
                    if fixed_examples["returncode"] == 0 :
                        comp_correct+=1 ; comp_pass_1[j]+=1
                else:
                    if sample["code_output"]["returncode"] == 0 :
                        comp_correct+=1 ; comp_pass_1[j]+=1
                if test["code_output"]["returncode"] == 0 :
                    test_correct+=1 ; test_pass_1[j]+=1

            # for pass@best
            if comp_correct>0: comp_pass_b+=1/n_tasks
            if test_correct>0: test_pass_b+=1/n_tasks
            # for pass@k
            comp_pass_k.append(pass_at_k(n=n_samples, c=comp_correct, k=n_samples-1))
            test_pass_k.append(pass_at_k(n=n_samples, c=test_correct, k=n_samples-1))

        # LLM stats
        llm_runtime_ave, llm_runtime_std = median_std(llm_runtime)
        llm_cputime_ave, llm_cputime_std = median_std(llm_cputime)
        llm_energy_ave, llm_energy_std = median_std(llm_energy)
        llm_co2eq_ave, llm_co2eq_std = median_std(llm_co2eq)
        llm_memory_ave, llm_memory_std = median_std(llm_memory)

        # pass@1 and std
        comp_pass_1_ave, comp_pass_1_std, comp_pass_1_str = normalize(comp_pass_1, norm=n_tasks, string=True)
        test_pass_1_ave, test_pass_1_std, test_pass_1_str = normalize(test_pass_1, norm=n_tasks, string=True)
        # pass@best
        comp_pass_b_ave, comp_pass_b_std = normalize(comp_pass_b)
        test_pass_b_ave, test_pass_b_std = normalize(test_pass_b)
        # pass@k
        comp_pass_k_ave, comp_pass_k_std = normalize(comp_pass_k)
        test_pass_k_ave, test_pass_k_std = normalize(test_pass_k)
        # -------------------------------------------
        # final summary

        if True:#"baseline" in file_name :
            print( # these values are only reliable for the baseline, because for the agent they fluctuate too much (if the code is incorrect I don't run it k times)
              f"===========================================\n"
              f"llm_runtime = ({llm_runtime_ave:.2f}, {llm_runtime_std:.2f}) s\n"
              f"llm_cputime = ({llm_cputime_ave:.2f}, {llm_cputime_std:.2f}) s\n"
              f"-------------------------------------------\n"
              f"llm_energy  = ({llm_energy_ave*1000:.2f}, {llm_energy_std*1000:.2f}) Wh\n"
              f"llm_co2eq   = ({llm_co2eq_ave*1000:.2f}, {llm_co2eq_std*1000:.2f}) g CO2eq\n"
              # f"llm_memory  = {llm_memory_ave:.2f} +/- {llm_memory_std:.2f} MiB"
              f"===========================================\n"
            )
        print(f"===========================================\n"
              f"comp_pass_1:\t{comp_pass_1_str}\n"
              f"test_pass_1:\t{test_pass_1_str}\n"
              f"-------------------------------------------\n"
              f"comp_pass_b:\t({comp_pass_b_ave*100:.2f}, {comp_pass_b_std*100:.2f})% /// IGNORE /// \n"
              f"test_pass_b:\t({test_pass_b_ave*100:.2f}, {test_pass_b_std*100:.2f})% /// IGNORE /// \n"
              f"-------------------------------------------\n"
              f"comp_pass_k:\t({comp_pass_k_ave*100:.2f}, {comp_pass_k_std*100:.2f})% /// IGNORE /// \n"
              f"test_pass_k:\t({test_pass_k_ave*100:.2f}, {test_pass_k_std*100:.2f})% /// IGNORE /// \n"
              f"===========================================\n"
              )


        # for the plot: baseline(ex, test)+bench(ex, test), variance
        # performance_plot["comp_pass_1_ave"] = round(comp_pass_1_ave, 2)
        # performance_plot["comp_pass_1_std"] = round(comp_pass_1_std, 2)
        # performance_plot["test_pass_1_ave"] = round(test_pass_1_ave, 2)
        # performance_plot["test_pass_1_std"] = round(test_pass_1_std, 2)
        #
        # performance_plot["comp_pass_k_ave"] = round(comp_pass_k_ave, 2)
        # performance_plot["comp_pass_k_std"] = round(comp_pass_k_std, 2)
        # performance_plot["test_pass_k_ave"] = round(test_pass_k_ave, 2)
        # performance_plot["test_pass_k_std"] = round(test_pass_k_std, 2)
        #
        # performance_plot["llm_runtime_ave"] = llm_runtime_ave
        # performance_plot["llm_runtime_std"] = llm_runtime_std
        # performance_plot["llm_cputime_ave"] = llm_cputime_ave
        # performance_plot["llm_cputime_std"] = llm_cputime_std
        # performance_plot["llm_energy_ave"]  = llm_energy_ave
        # performance_plot["llm_energy_std"]  = llm_energy_std
        # performance_plot["llm_co2eq_ave"]   = llm_co2eq_ave
        # performance_plot["llm_co2eq_std"]   = llm_co2eq_std
        # performance_plot["llm_memory_ave"]  = llm_memory_ave
        # performance_plot["llm_memory_std"]  = llm_memory_std

        # print(f"\nstats: {performance_plot}\n")

        if "benchmark" in file_name: print("\n\n")

