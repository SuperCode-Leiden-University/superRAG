# libraries
import torch, pprint # pretty-print for dict
#import ast
from datetime import datetime
import getopt, sys # handle flags and pass args from terminal

# my packages
from supercode.src.agent import Agent
from supercode.src.tools.code_processing import *
from supercode.src.tools.tools import *
from supercode.src.benchmark import *

# importing variables from the config file
from supercode.src.configs.parse_config import *
from supercode.src.configs.system_prompts import baseline_prompt, benchmark_prompt


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

try:
    arguments, values = getopt.getopt(args, options, long_options)
    for currentArg, currentVal in arguments:
        if currentArg in ("-h", "--help"):
            print("\nActivate coding agent.\nPass '-b' to evaluate on benchmark.\nPress 'q' to quit.\n")
        if currentArg in ("-b", "--benchmark"):
            # override the default in config
            benchmark_path = currentVal # example: "openai/openai_humaneval"
            print(f"\nEvaluating benchmark: {currentVal}\n")
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
    tot_start = datetime.now()
    model = Agent()

    if benchmark_path is not None: # evaluating benchmark, given from terminal or default in config
        run_benchmark(
            model, benchmark_path,
            baseline=True, # check the baseline model, then use feedback from tools
            check_single_task=102, # check a specific task
        )

    else:
        ##### CHAT WITH THE MODEL
        while True:
            # ask the user to write a query

            user_start = datetime.now()
            user_prompt = input(
                    "\nEnter your query (type 'q' or 'quit' to exit, 'reset' or 'r' to clear chat history) \n"
                    "----------------------------------------------------------------------------\n"
                    "## Ut: "
            )
            print(  "----------------------------------------------------------------------------\n")
            user_end = datetime.now()

            # check if user wants to quit
            if user_prompt.lower() == "quit" or user_prompt.lower() == "q":
                print("Goodbye!")

                print("chat history:\n") ; model.print_chat_history()
                break
            elif user_prompt.lower() == "reset" or user_prompt.lower() == "r":
                model.reset_memory()
                continue

            start = datetime.now()
            if verbose>-1 : print(">> processing the query")
            model.call(user_prompt)

            end = datetime.now()
            if verbose>-1 : print(">> Time to Answer =", end-start)

        tot_start += (user_end - user_start) # avoid counting the time the user takes to write the message


    tot_end = datetime.now()
    print(">> Total Time: ", tot_end-tot_start)

if __name__ == '__main__':
    main()