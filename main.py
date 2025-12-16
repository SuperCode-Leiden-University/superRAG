# libraries
import torch
from transformers import pipeline # pipeline is for inference

# check if I have an Nvidia GPU on the machine
print(torch.cuda.is_available())
#print(torch.cuda.device_count(), torch.cuda.get_device_name(0), sep="\n")

# this will import the LLM
#model = pipeline(model="codefuse-ai/CodeFuse-CGM-72B")

# importing the model
model = pipeline("text-generation", model="Qwen/Qwen3-Coder-30B-A3B-Instruct")

user_prompt = "how to plot sin(x) in python?"
messages = [
    {"role": "user", "content": user_prompt},
]

response = model(messages)
print(response)
