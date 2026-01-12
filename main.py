# libraries
import os
import torch
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModel # pipeline is for inference
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------------------------------------------------- #
# check if I have an Nvidia GPU on the machine
print("Is cuda available?", torch.cuda.is_available())
#print(torch.cuda.device_count(), torch.cuda.get_device_name(0), sep="\n")

# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
##### IMPORTING THE MODELS

# model is for processing text and generating an answer
model = pipeline("text-generation", model="Qwen/Qwen2.5-Coder-1.5B-Instruct")

# emb_model is for creating the embeddings for the vector database (for RAG)
emb_model = HuggingFaceEmbeddings( model_name="sentence-transformers/msmarco-bert-base-dot-v5" )

user_prompt = "Do the following tasks: tell me where the velocity-Verlet function is defined and copy-paste the function"

# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
# CREATING THE DATABASE

db_dir = "test-db" # where I save the vector database (db)
update_db = False # if I want to update the db (for example because I changed some files)

start = datetime.now()

# first check if the db exists and if it needs to be updated
if update_db or not(os.path.isdir(db_dir)):
    print("creating the vector database")
    # ---------------------------------------------------------------------------------------------- #
    # load the docs
    loader = DirectoryLoader(
        "./test-code/",
        loader_cls=TextLoader,
        glob="**/*.*",
        show_progress=True,
        use_multithreading=True
    )
    # IMPORTANT: PDFs need a specific loader or they generate errors!
    docs = loader.load()
    print("n_docs =", len(docs)) # just to check that all docs have been loaded correctly

    # ---------------------------------------------------------------------------------------------- #
    # split the docs into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print("n_chunks =", len(chunks))
    # Note: chunk[0] = page_content='text' metadata={'source': 'test-code/filename'}

    # ---------------------------------------------------------------------------------------------- #
    # embed query and docs
    texts = [doc.page_content for doc in chunks] # extract text
    metadatas = [doc.metadata for doc in chunks] # extract the metadata

    ids = [str(i) for i in range(0, len(chunks))]
    emb_query = emb_model.embed_query(user_prompt)
    emb_chunks = emb_model.embed_documents(texts)
    text_embeddings = list(zip(texts, emb_chunks))

    # ---------------------------------------------------------------------------------------------- #
    # create the database from the embeddings
    db = FAISS.from_embeddings(text_embeddings, emb_model, metadatas=metadatas)

    # save db for future use
    db.save_local(db_dir) # create a dir with 2 files: index.faiss, index.pkl

else:
    print("loading the vector database from '"+str(db_dir)+"'")
    # load saved db
    db = FAISS.load_local(db_dir, embeddings=emb_model, allow_dangerous_deserialization=True)

end = datetime.now()
print(">> Time to create Database =", end-start)

# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
# RETRIEVAL & PROCESSING THE DOCS

start = datetime.now()
# find the relevant docs
retriv_docs = db.similarity_search(user_prompt, k=5) # find the k most relevant documents to the query
print("n_retriv_docs =", len(retriv_docs))
#print("most relevant doc\n", retriv_docs[0])

end = datetime.now()
print(">> Time to Retrieval =", end-start)

# ---------------------------------------------------------------------------------------------- #
start = datetime.now()
print("processing the query...")
# include the retrieved docs as context and feed it to the model
messages = [{
    "role": "user",
    "content": f"Use the following context to answer the question.\n\n" 
               f"### Context\n{retriv_docs}\n\n" 
               f"### Question\n{user_prompt}"
}]

chat_history = model(messages)
""" Note: chat structure is the following
chat_history = [ {'generated_text': [
                    {'role': 'user',       'content': '...'}, 
                    {'role': 'assistant',  'content': "..."}
                ] } ]
"""

response = chat_history[0]['generated_text'][-1]['content']
end = datetime.now()
print(">> Time to First Token =", end-start)
print("--------------------------------------", response, "--------------------------------------", sep="\n")

