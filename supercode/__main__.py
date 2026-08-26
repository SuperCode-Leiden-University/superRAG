# libraries
#import torch
import ast # pretty-print for dict
from datetime import datetime
import getopt, sys # handle flags and pass args from terminal

# benchmark datasets
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
# check for flags and arguments from the command line

# print("Total arguments:", len(sys.argv))
# print("Script name:", sys.argv[0])
# print("Arguments:", sys.argv[1:])

args = sys.argv[1:]
options = "hb:"
long_options = ["help", "benchmark="]
# in long_options, "name=" means the flag expects an argument

benchmark_mode = False # default
try:
    arguments, values = getopt.getopt(args, options, long_options)
    for currentArg, currentVal in arguments:
        if currentArg in ("-h", "--help"):
            print("Activate coding agent.\nPass '-b' to evaluate on benchmark.\nPress 'q' to quit.")
        elif currentArg in ("-b", "--benchmark"):
            benchmark_mode = True
            bench_name = currentVal
            print("Evaluating benchmark:", currentVal)
except getopt.error as err:
    print(str(err))

# ---------------------------------------------------------------------------------------------- #
# check if I have an Nvidia GPU on the machine
#if verbose>1 :
    #print("Is cuda available?", torch.cuda.is_available())
    #print(torch.cuda.device_count(), torch.cuda.get_device_name(0), sep="\n")


# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
##### IMPORTING THE MODEL
model = Agent()

# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
def main():
    if not benchmark_mode:
        ##### CHAT WITH THE MODEL
        while True:
            # ask the user to write a query
            user_prompt = input(
                    "\nEnter your query (type 'q' or 'quit' to exit) \n"
                    "----------------------------------------------------------------------------\n"
                    "## Ut: "
            )
            print(  "----------------------------------------------------------------------------\n")

            # check if user wants to quit
            if user_prompt.lower() == "quit" or user_prompt.lower() == "q":
                print("Goodbye!")
                break

            start = datetime.now()
            if verbose>0 : print(">> processing the query")
            model.call(user_prompt)

            end = datetime.now()
            if verbose>0 : print(">> Time to Answer =", end-start)

    else: # evaluating benchmark

        ##### BENCHMARK SETTINGS
        baseline = True  # True or False
        # if baseline is true then the model does not use any external info
        # if it is false, it checks first if it can find the baseline solution and then ask the model to improve it

        num_samples_per_task = 5  # 200
        # how many times the model tries to solve the same task

        check_single_task = False; i_task = 102 # for debugging

        print("baseline =", baseline,  "; num_samples_per_task =", num_samples_per_task)
        if check_single_task: print("WARNING: mode check_single_task, i_task =", i_task)

        # importing the benchmark from hugging face
        if bench_name == "humaneval" :
            dataset = load_dataset("openai/openai_humaneval")
            # IMPORTANT: some tasks (47, 163) in HumanEval contain mismatches or errors!
            # /47:  2nd example is incorrect and doesn't match the test
            # /163: unclear prompt, should specify to return only digits that are between min(0,a,b) and max(a,b,9)
        elif bench_name == "crosscodeeval" :
            dataset = load_dataset("bigcode/crosscodeeval")["test"]

        if baseline:
            print("benchmark baseline and model for " + bench_name)
        else:
            print("benchmark model for " + bench_name)

        if debugging:
            debug_model = "+" + debug_model_id
        else:
            debug_model = ""

        if n_iterations > 1:
            iter = "_x" + n_iterations
        else:
            iter = ""

        baseline_file = gen_code_dir + "/" + bench_name + "_" + model_id[
            model_id.find("/") + 1:] + debug_model + "_baseline_" + str(num_samples_per_task) + iter + ".jsonl"
        benchmark_file = gen_code_dir + "/" + bench_name + "_" + model_id[
            model_id.find("/") + 1:] + debug_model + "_benchmark_" + str(num_samples_per_task) + iter + ".jsonl"

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
        # print("\nsample 0:"); pprint.pprint(sample)

        # ---------------------------------------------------------------------------------------------- #
        # ---------------------------------------------------------------------------------------------- #
        # ---------------------------------------------------------------------------------------------- #
        # standard HumanEval code
        problems = read_problems()
        n_tasks = len(problems)

        if not baseline:  # load the baseline functions
            baseline_samples = []
            print("reading jsonl for " + baseline_file + "...")
            try:
                with open(baseline_file) as f:
                    for i, line in enumerate(f, start=1):
                        try:  # read the file line-by-line
                            baseline_samples.append(ast.literal_eval(line))
                        except Exception as e:
                            print("Error on line", i)
                            print("Line content:", repr(line))
                            raise Exception("Error in the baseline file!")
            except Exception as e:
                print("Error with baseline file:", e)
            if len(baseline_samples) != num_samples_per_task * n_tasks:
                print("WARNING: baseline file does not match the expected number of samples")
                print("n_baseline:", len(baseline_samples), "\nn_samples:", num_samples_per_task * n_tasks)
                print("Evaluating baseline first")
                baseline = True

        # create an empty file (or overwrite if the file exists)
        if baseline:  # clear the baseline only if
            with open(baseline_file, "w") as f: f.close()
        with open(benchmark_file, "w") as f:
            f.close()

        for i, task_id in enumerate(problems):
            if check_single_task: i = i_task; task_id = "HumanEval/" + str(i)
            print("\n======================================")
            print(i, task_id)

            sample = problems[task_id]
            entry_point = sample["entry_point"]
            prompt = sample["prompt"]
            test = sample["test"]
            # task_id="HumanEval/47" has an error in the prompt, it should be:
            # >>> median([-10, 4, 6, 1000, 10, 20])
            # 8.0 (instead of 15.0)
            if task_id == "HumanEval/47": prompt = prompt.replace("15.0", "8.0"); print(prompt)

            for j in range(num_samples_per_task):
                if baseline:  # create the baseline
                    baseline_response = model.call(baseline_prompt + "\n\n" + prompt, reset_memory=True, baseline=True)
                    print("\n>> extracting code from baseline response")
                    baseline_code = extract_code(baseline_response, entry_point)
                    print("\n>> checking compiler output for baseline response")
                    baseline_compiler_output = sandboxed_compiler(baseline_code)
                    baseline_json_sample = convert_to_json(task_id, baseline_response, baseline_code,
                                                           compiler_output=baseline_compiler_output)

                    with open(baseline_file, "a") as f:
                        f.write(str(baseline_json_sample) + "\n")
                        f.close()
                else:
                    # retrieve the baseline function
                    baseline_code = baseline_samples[i * (j + 1)]["code"]

                # then ask the model to improve the baseline
                response = model.call(benchmark_prompt + "\n\n" + prompt, code=baseline_code, reset_memory=True,
                                      baseline=False)
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

main()