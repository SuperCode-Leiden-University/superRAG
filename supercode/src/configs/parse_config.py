from yaml import load # for the config file
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

# ----------------------------------------------------------------------------------------------
# loading variables from the configuration file (.yaml)
vars = load(open("src/configs/config.yaml", 'r'), Loader=Loader)

# ----------------------------------------------------------------------------------------------
# general
verbose = vars["verbose"] # how much info is printed: 0=none, 1=little, 2=all

# ----------------------------------------------------------------------------------------------
# chat assistant model
model_args     = vars["model"]
model_id       = vars["model"]["model_id"] # model ID from HuggingFace
raw_model      = vars["model"]["raw_model"]   # True if the model is loaded directly, False if loaded through pipeline
quant_type     = vars["model"]["quant_type"] # valid values: ("full", "bits", "GPTQ") --> check file formats!!!
temperature    = vars["model"]["gen_args"]["temperature"] # 0 = always select the most likely word, 1 = random
max_new_tokens = vars["model"]["gen_args"]["max_new_tokens"] # max number of tokens that can be generated


# ----------------------------------------------------------------------------------------------
# thinking model for planning
plan_model_args     = vars["plan_model"]
plan_model_id       = vars["plan_model"]["model_id"] # model ID from HuggingFace
plan_raw_model      = vars["plan_model"]["raw_model"]   # True if the model is loaded directly, False if loaded through pipeline
plan_quant_type     = vars["plan_model"]["quant_type"] # valid values: ("full", "bits", "GPTQ") --> check file formats!!!
plan_temperature    = vars["plan_model"]["gen_args"]["temperature"] # 0 = always select the most likely word, 1 = random
plan_max_new_tokens = vars["plan_model"]["gen_args"]["max_new_tokens"] # max number of tokens that can be generated

# ----------------------------------------------------------------------------------------------
# embedding model for building the database (for RAG)
emb_model_id = vars["emb_model_id"] # model ID from HuggingFace

# ----------------------------------------------------------------------------------------------
# where I save the results from the agent's tools
tools_dir     = vars["tools_dir"]     # where I save the results from the agent's tools
docs_dir      = vars["docs_dir"]      # where I save the files for RAG
db_dir        = vars["db_dir"]        # where I save the vector database (db) for RAG
docker_dir    = vars["docker_dir"]    # where I save the dockerfile and docker_compose.yml
gen_code_dir  = vars["gen_code_dir"]  # where I mount with the docker container
gen_code_file = vars["gen_code_file"] # where I save the temp file for generating code

# ----------------------------------------------------------------------------------------------
# tools & database (RAG)
update_db = vars["update_db"] # if I want to update the db (for example because I changed some files)
chunk_size = vars["chunk_size"]
chunk_overlap = vars["chunk_overlap"]



