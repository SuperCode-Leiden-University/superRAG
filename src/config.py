from yaml import load # for the config file
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

# ----------------------------------------------------------------------------------------------
# loading variables from the configuration file (.yaml)
vars = load(open("src/config.yaml", 'r'), Loader=Loader)

# ----------------------------------------------------------------------------------------------
# general
verbose = vars["verbose"] # how much info is printed: 0=none, 1=little, 2=all

# ----------------------------------------------------------------------------------------------
# chat model
model_id = vars["model"]["ID"] # model ID from HuggingFace
raw_model = vars["model"]["raw_model"]   # True if the model is loaded directly, False if loaded through pipeline
quant_type = vars["model"]["quant_type"] # valid values: ("full", "bits", "GPTQ") --> check file formats!!!

# ----------------------------------------------------------------------------------------------
# embedding model for building the database (for RAG)
emb_model_id = vars["emb_model"] # model ID from HuggingFace

# ----------------------------------------------------------------------------------------------
# database (RAG)
n_retriv  = vars["n_retriv"]  # number of retrieved docs
docs_dir  = vars["docs_dir"]  # where I save the files for RAG
db_dir    = vars["db_dir"]    # where I save the vector database (db)
update_db = vars["update_db"] # if I want to update the db (for example because I changed some files)


