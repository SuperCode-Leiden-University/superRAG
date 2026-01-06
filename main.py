# libraries
import torch
from transformers import pipeline # pipeline is for inference
from langchain import embeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------------------------- #
# check if I have an Nvidia GPU on the machine
print("Is cuda available?", torch.cuda.is_available())
#print(torch.cuda.device_count(), torch.cuda.get_device_name(0), sep="\n")

# ---------------------------------------------------------------------------------------------- #
# importing the model
model = pipeline("text-generation", model="Qwen/Qwen2.5-Coder-1.5B-Instruct")

#"""
# ---------------------------------------------------------------------------------------------- #
# creating the database (remove PDFs because they generate errors!)
loader = DirectoryLoader(
    "./test-code/",
    loader_cls=TextLoader,
    glob="**/*.*",
    show_progress=True,
    use_multithreading=True
)
docs = loader.load()
print("n docs =", len(docs)) # just to check that all docs have been loaded correctly
# there should be 7 files

# splitting the docs into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
print("n chunks =", len(chunks))

db_dir = "test-db"

#"""
# ---------------------------------------------------------------------------------------------- #
user_prompt = "Can you copy-paste the code I used for the velocity-Verlet method?"
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

