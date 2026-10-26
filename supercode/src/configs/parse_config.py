import os
from yaml import load # for the config file
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

# ----------------------------------------------------------------------------------------------
# loading variables from the configuration file (.yaml)
#print("current dir:", os.getcwd())
vars = load(open("supercode/src/configs/config.yaml", 'r'), Loader=Loader)

# ----------------------------------------------------------------------------------------------
# general
verbose = vars["verbose"] # how much info is printed: 0=none, 1=little, 2=all
backend = vars["backend"] # which backend is used: ["transformers", "llama-cpp", "vLLM"]

model_iter = vars["model_iter"]
tools_iter = vars["tools_iter"]

k_runtimes = vars["k_runtimes"]
optim_metric = vars["optim_metric"]

benchmark_args = vars["benchmark"] # run benchmark-results


# ----------------------------------------------------------------------------------------------
# chat assistant model
coder_model_args       = vars["coder_model"]
coder_model_id         = vars["coder_model"]["model_id"] # model ID from HuggingFace
coder_quant_type       = vars["coder_model"]["quant_type"] # valid values: ("full", "bits", "GPTQ") --> check file formats!!!
coder_gen_args         = vars["coder_model"]["gen_args"]
coder_multi_sampl_args = vars["coder_model"]["multi_sampl_args"]

# ----------------------------------------------------------------------------------------------
# thinking model for selecting tools
think_model_args       = vars["think_model"]
think_model_id         = vars["think_model"]["model_id"] # model ID from HuggingFace
think_quant_type       = vars["think_model"]["quant_type"] # valid values: ("full", "bits", "GPTQ") --> check file formats!!!
think_gen_args         = vars["think_model"]["gen_args"]
think_multi_sampl_args = vars["think_model"]["multi_sampl_args"]

# ----------------------------------------------------------------------------------------------
# where I save the results
tools_dir = vars["tools_dir"]     # where I save the results from the agent's tools
bench_dir = vars["bench_dir"]     # where I save the results from the benchmarks
docs_dir  = vars["docs_dir"]      # where I save the files for RAG

# ----------------------------------------------------------------------------------------------
# database (RAG)
emb_model_id  = vars["emb_model_id"] # embedding model ID from HuggingFace for building the DB
db_dir        = vars["db_dir"] # where I save the vector database (db) for RAG
update_db     = vars["update_db"] # if I want to update the db (for example because I changed some files)
chunk_size    = vars["chunk_size"]
chunk_overlap = vars["chunk_overlap"]

# ----------------------------------------------------------------------------------------------
# docker
#docker_dir    = vars["docker_dir"]    # where I save the dockerfile and docker_compose.yml
#gen_code_dir  = vars["gen_code_dir"]  # where I mount with the docker container
gen_code_file = vars["gen_code_file"] # where I save the temp file for generating code
docker_args   = vars["docker"]



