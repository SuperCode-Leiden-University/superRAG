# libraries
import torch
from datetime import datetime
from datasets import load_dataset

# my packages
from src.model import Model

# importing variables from the config file
from src.configs.parse_config import verbose, model_id, raw_model, quant_type



# ---------------------------------------------------------------------------------------------- #
# check if I have an Nvidia GPU on the machine
if verbose>1 :
    print("Is cuda available?", torch.cuda.is_available())
    #print(torch.cuda.device_count(), torch.cuda.get_device_name(0), sep="\n")


# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
##### IMPORTING THE MODEL & BENCHMARK

# "model" is for processing text and generating an answer
model = Model(model_id, raw_model, quant_type)

# importing the benchmark from hugging face
dataset = load_dataset("openai/openai_humaneval")
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
print("\n# task_id:\n", sample["task_id"], "\n\n# prompt:\n", sample["prompt"], "\n\n# canonical_solution:\n", sample["canonical_solution"], sep="")


# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
##### BENCHMARK THE MODEL
for i in range(L_test):
    print("exemple ", i, "/", L_test, sep ="" )

    sample = test_set[i]
    prompt = sample["prompt"]
    sol = sample["canonical_solution"]

    try:
        start = datetime.now()
        response = model.call(prompt)

        end = datetime.now()
        if verbose>0 : print(">> Time to Answer =", end-start)

    except Exception as e:
        print(f"\nAn error occurred:\n{e}\n")
