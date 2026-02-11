# libraries
import torch
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings

# my packages
from src.tools import search_database
from src.model import Model
from src.database import Database

# importing variables from the config file
from src.config import verbose, model_id, raw_model, quant_type, emb_model_id, n_retriv, docs_dir, db_dir, update_db



# ---------------------------------------------------------------------------------------------- #
# check if I have an Nvidia GPU on the machine
if verbose>1 :
    print("Is cuda available?", torch.cuda.is_available())
    #print(torch.cuda.device_count(), torch.cuda.get_device_name(0), sep="\n")



# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
##### IMPORTING THE MODELS

# "model" is for processing text and generating an answer
model = Model(model_id, raw_model, quant_type)

# "emb_model" is for creating the embeddings for the vector database (for RAG)
emb_model = HuggingFaceEmbeddings(model_name=emb_model_id)

# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
# CREATING THE DATABASE
if n_retriv>0 :
    start = datetime.now()
    # this will first check if the database exists and if it needs to be updated
    # then either load or create the database
    db = Database(emb_model, docs_dir, db_dir, update_db).load()
    end = datetime.now()
    if verbose>0 : print(">> Time to Database =", end-start)


while True:
    # Query to send to Claude
    user_prompt = input("\nEnter your query (type 'q' or 'quit' to exit) \n--------------------------------------\n## Ut: ")
    print("--------------------------------------")

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

        if n_retriv>0 : retriv_docs = search_database(db, user_prompt, n_retriv)
        else: retriv_docs=None

        # ---------------------------------------------------------------------------------------------- #
        # ---------------------------------------------------------------------------------------------- #
        # ---------------------------------------------------------------------------------------------- #
        ##### MODEL INFERENCE

        start = datetime.now()
        if verbose>0 : print(">> processing the query")
        # include the retrieved docs as context and feed it to the model
        response = model.call(user_prompt, retriv_docs)

        end = datetime.now()
        if verbose>0 : print(">> Time to First Token =", end-start)
        print("-------------------------------------- \n## AI: ", response, "\n--------------------------------------", sep="")

    except Exception as e:
        print(f"\nAn error occurred:\n{e}\n")
