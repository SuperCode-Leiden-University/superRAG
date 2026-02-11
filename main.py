# libraries
import torch
from datetime import datetime

# my packages
from src.model import Model

# importing variables from the config file
from src.config import verbose, model_id, raw_model, quant_type



# ---------------------------------------------------------------------------------------------- #
# check if I have an Nvidia GPU on the machine
if verbose>1 :
    print("Is cuda available?", torch.cuda.is_available())
    #print(torch.cuda.device_count(), torch.cuda.get_device_name(0), sep="\n")


# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
##### IMPORTING THE MODEL

# "model" is for processing text and generating an answer
model = Model(model_id, raw_model, quant_type)


# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
##### CHAT WITH THE MODEL
while True:
    # ask the user to write a query
    user_prompt = input("\nEnter your query (type 'q' or 'quit' to exit) \n--------------------------------------\n## Ut: ")
    print("--------------------------------------")

    # check if user wants to quit
    if user_prompt.lower() == "quit" or user_prompt.lower() == "q":
        print("Goodbye!")
        break

    try:
        start = datetime.now()
        if verbose>0 : print(">> processing the query")
        response = model.call(user_prompt)

        end = datetime.now()
        if verbose>0 : print(">> Time to First Token =", end-start)
        print("-------------------------------------- \n## AI: ", response, "\n--------------------------------------", sep="")

    except Exception as e:
        print(f"\nAn error occurred:\n{e}\n")
