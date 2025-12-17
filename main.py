# libraries
import torch
from transformers import pipeline # pipeline is for inference

# check if I have an Nvidia GPU on the machine
print("Is cuda available?", torch.cuda.is_available())
#print(torch.cuda.device_count(), torch.cuda.get_device_name(0), sep="\n")

# this will import the LLM
#model = pipeline(model="codefuse-ai/CodeFuse-CGM-72B")

# importing the model
model = pipeline("text-generation", model="Qwen/Qwen2.5-Coder-1.5B-Instruct")

user_prompt = "how can I plot sin(x) in python?"
messages = [
    {"role": "user", "content": user_prompt},
]

chat_history = model(messages)
"""
chat_history = [ {'generated_text': [
                    {'role': 'user',       'content': '...'}, 
                    {'role': 'assistant',  'content': "..."}
                ] } ]
"""

response = chat_history[0]['generated_text'][-1]['content']

print("--------------------------------------", response, "--------------------------------------", sep="\n")

