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


# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
def run_benchmark(model, benchmark_path, baseline=True, num_samples_per_task=5, check_single_task=False, i_task=102):
    # if baseline is true then the model does not use any external info
    # if it is false, it checks first if it can find the baseline solution and then ask the model to improve it
    print("baseline =", baseline, "; num_samples_per_task =", num_samples_per_task)
    if check_single_task: print("WARNING: mode check_single_task, i_task =", i_task)

    # importing the benchmark from hugging face
    dataset = load_dataset(benchmark_path) ; bench_name = benchmark_path[benchmark_path.find("/")+1:]
    #dataset = load_dataset("openai/openai_humaneval"); bench_name = "humaneval"# dataset = load_dataset("bigcode/crosscodeeval")["test"]; bench_name = "crosscodeeval"

    # print("Dataset structure:\n", dataset, sep="")
    """ dataset looks like this:
        DatasetDict({
            test: Dataset({
                features: ['task_id', 'prompt', 'canonical_solution', 'test', 'entry_point'],
                num_rows: 164
            })
        })
        
    # IMPORTANT: some tasks (47, 163) in HumanEval contain mismatches or errors!
    # /47: has an error in the prompt, it should be: median([-10, 4, 6, 1000, 10, 20]) = 8.0 (instead of 15.0)
    # /163: unclear prompt, should specify to return only digits that are between min(0,a,b) and max(a,b,9)
    
    """
    # ----------------------------------------------------------------------------------------------

    if baseline: print("benchmark baseline and model for "+bench_name)
    else: print("benchmark model for "+bench_name)

    if debugging: debug_model = "+"+debug_model_id
    else: debug_model = ""

    if n_iterations>1 : iter="_x"+n_iterations
    else: iter=""

    baseline_file  = gen_code_dir+"/"+bench_name+"_"+model_id[model_id.find("/")+1:]+debug_model+"_baseline_" +str(num_samples_per_task)+iter+".jsonl"
    benchmark_file = gen_code_dir+"/"+bench_name+"_"+model_id[model_id.find("/")+1:]+debug_model+"_benchmark_"+str(num_samples_per_task)+iter+".jsonl"

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
        test = sample["test"]

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

