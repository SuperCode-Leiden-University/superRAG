# libraries
#import ast
import getopt, sys # handle flags and pass args from terminal

# my packages
from supercode.benchmark import *

# importing variables from the config file
from supercode.src.configs.parse_config import *

# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
# check for flags and arguments from the command line

# print("Total arguments:", len(sys.argv))
# print("Script name:", sys.argv[0])
# print("Arguments:", sys.argv[1:])

args = sys.argv[1:]
options = "hbc:"
long_options = ["help", "benchmark=", "model="]
# in long_options, "name=" means the flag expects an argument

try:
    arguments, values = getopt.getopt(args, options, long_options)
    for currentArg, currentVal in arguments:
        if currentArg in ("-h", "--help"):
            print("\nActivate coding agent.\nPass '-b' to evaluate on benchmark.\nPress 'q' to quit.\n")

        if currentArg in ("-c", "--coder"):
            # override the default in config
            coder_model_id = currentVal # manual selection for the model
            print(f"\nEvaluating coder model: {currentVal}\n")

        if currentArg in ("-b", "--benchmark"):
            # override the default in config
            benchmark_args["name"] = currentVal # example: "openai/openai_humaneval"
            print(f"\nEvaluating benchmark: {benchmark_args}\n")
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
    agent = Agent(coder_model_id)

    if benchmark_args["name"] is not None: # evaluating benchmark-results, given from terminal or default in config
        run_benchmark(agent, benchmark_args)

    else:
        ##### CHAT WITH THE MODEL
        while True:
            # ask the user to write a query
            user_start = datetime.now()
            user_prompt = input(
                    "\nEnter your query (type 'q' or 'quit' to exit, 'reset' or 'r' to clear chat history) \n"
                    "----------------------------------------------------------------------------\n"
                    "## Ut: "
            );print("----------------------------------------------------------------------------\n")
            user_end = datetime.now()

            # check if user wants to quit
            if user_prompt.lower() == "quit" or user_prompt.lower() == "q":
                print("Goodbye!")
                print("\n>> chat history:") ; agent.print_chat_history()
                break
            elif user_prompt.lower() == "reset" or user_prompt.lower() == "r":
                agent.reset_memory()
                continue

            # call the agent to answer the query
            start = datetime.now()
            if verbose>-1 : print(">> processing the query")
            agent.call(user_prompt)
            end = datetime.now()
            if verbose>-1 : print(">> Time to Answer =", end-start)

        tot_start += (user_end-user_start) # avoid counting the time the user takes to write the message


    tot_end = datetime.now()
    print(">> Total Time: ", tot_end-tot_start)

if __name__ == '__main__':
    main()