# libraries
import torch
from transformers import pipeline # pipeline is for inference

print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0), sep="\n")

# this will import the LLM
model = pipeline(model="codefuse-ai/CodeFuse-CGM-72B")

user_prompt = "how to plot sin(x) in python?"
response = model(user_prompt)
print(response)
