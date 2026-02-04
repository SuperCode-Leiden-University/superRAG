# libraries
import torch
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings

# my packages
from src.model import Model
from src.database import Database

# ---------------------------------------------------------------------------------------------- #
# check if I have an Nvidia GPU on the machine
print("Is cuda available?", torch.cuda.is_available())
#print(torch.cuda.device_count(), torch.cuda.get_device_name(0), sep="\n")

# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
##### VARIABLES
model_id = \
    "Qwen/Qwen2.5-Coder-1.5B-Instruct" # file size = 3.1 GB
    #"Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4" # file size = 5.59 GB
    #"Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4" # file size = 19.4 GB
emb_model_id = "sentence-transformers/msmarco-bert-base-dot-v5"

raw_model = True # True if the model is loaded directly, False if loaded through pipeline
quant_type = "full" # valid values: ("full", "bits", "GPTQ") --> check file formats!!!
"""
NOTE: the model can be loaded as quantized only if someone published the quantized version (see files):
    --> full precision:   model.safetensors                 (direct load)
    --> full precision:   pytorch_model.bin.index.json      (direct load, NOT with Transformers!)
    --> 4bit quantized:   model-4bit.safetensors            (needs bitsandbytes)
    --> GPTQ quantized:   gptq_model-4bit-128g.safetensors  (direct load)
    --> GGUF quantized:   model.gguf                        (direct load, NOT with Transformers!)
"""

n_retriv = 0 # number of retrieved docs
docs_dir = "test-code" # where I save the files for RAG
db_dir   = "test-db"   # where I save the vector database (db)
update_db = False      # if I want to update the db (for example because I changed some files)

# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
##### IMPORTING THE MODELS

# "model" is for processing text and generating an answer
model = Model(model_id, raw_model, quant_type)

# "emb_model" is for creating the embeddings for the vector database (for RAG)
emb_model = HuggingFaceEmbeddings(model_name=emb_model_id )

# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
# CREATING THE DATABASE

start = datetime.now()
# first check if the db exists and if it needs to be updated
db = Database(emb_model, docs_dir, db_dir, update_db).load()
end = datetime.now()
print(">> Time to Database =", end-start)


while True:
    # Query to send to Claude
    user_prompt = input("Enter your query (type 'q' or 'quit' to exit) \n--------------------------------------\nUt: ")
    print("--------------------------------------")
    # user_prompt = "Do the following tasks: tell me where the velocity-Verlet function is defined and copy-paste the function"

    # Check if user wants to quit
    if user_prompt.lower() == "quit" or user_prompt.lower() == "q":
        print("Goodbye!")
        break

    # Make the API call using Converse API
    try:
        # ---------------------------------------------------------------------------------------------- #
        # ---------------------------------------------------------------------------------------------- #
        # ---------------------------------------------------------------------------------------------- #
        # RETRIEVAL & PROCESSING THE DOCS

        start = datetime.now()
        # find the relevant docs
        if n_retriv>0:
            retriv_docs = db.similarity_search(user_prompt, k=n_retriv) # find the k most relevant documents to the query
            print("n_retriv_docs =", len(retriv_docs), "=", n_retriv)
            #print("most relevant doc\n", retriv_docs[0])
        else:
            retriv_docs = None

        end = datetime.now()
        print(">> Time to Retrieval =", end-start)

        # ---------------------------------------------------------------------------------------------- #
        # ---------------------------------------------------------------------------------------------- #
        # ---------------------------------------------------------------------------------------------- #
        ##### MODEL INFERENCE

        start = datetime.now()
        print("processing the query...")
        # include the retrieved docs as context and feed it to the model
        response = model.call(user_prompt, retriv_docs)

        end = datetime.now()
        print(">> Time to First Token =", end-start)
        print("-------------------------------------- \nAI: ", response, "\n--------------------------------------", sep="")

    except Exception as e:
        print(f"\nAn error occurred:\n{e}\n")
