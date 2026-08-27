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
from src.run_benchmark import *
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
# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
def main():
    # ---------------------------------------------------------------------------------------------- #
    # check if I have an Nvidia GPU on the machine
    # if verbose>1 :
    # print("Is cuda available?", torch.cuda.is_available())
    # print(torch.cuda.device_count(), torch.cuda.get_device_name(0), sep="\n")

    # ---------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------- #
    ##### IMPORTING THE MODEL
    model = Agent()

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
        run_benchmark(
            model, "openai/openai_humaneval",
            baseline=True, num_samples_per_task=5,
            check_single_task=False, i_task=102
        )

if __name__ == '__main__':
    main()