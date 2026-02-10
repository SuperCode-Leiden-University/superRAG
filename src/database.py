import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.verbose import verbose


class Database():
    def __init__(self, emb_model, docs_dir, db_dir, update_db=False):
        self.emb_model = emb_model
        self.docs_dir = docs_dir
        self.db_dir = db_dir
        self.update_db = update_db

    def load(self):
        if self.update_db or not (os.path.isdir(self.db_dir)):
            if verbose()>0 : print(">> creating the vector database")
            # ---------------------------------------------------------------------------------------------- #
            # load the docs
            loader = DirectoryLoader(
                "./" + self.docs_dir, +"/",
                loader_cls=TextLoader,
                glob="**/*.*",
                show_progress=True,
                use_multithreading=True
            )
            # IMPORTANT: PDFs need a specific loader or they generate errors!
            docs = loader.load()
            if verbose()>1 : print(">> n_docs =", len(docs))  # just to check that all docs have been loaded correctly

            # ---------------------------------------------------------------------------------------------- #
            # split the docs into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            if verbose()>1 : print(">> n_chunks =", len(chunks))
            # Note: chunk[0] = page_content='text' metadata={'source': 'test-code/filename'}

            # ---------------------------------------------------------------------------------------------- #
            # embed query and docs
            texts = [doc.page_content for doc in chunks]  # extract text
            metadatas = [doc.metadata for doc in chunks]  # extract the metadata

            #ids = [str(i) for i in range(0, len(chunks))]
            #emb_query = emb_model.embed_query(user_prompt)
            emb_chunks = self.emb_model.embed_documents(texts)
            text_embeddings = list(zip(texts, emb_chunks))

            # ---------------------------------------------------------------------------------------------- #
            # create the database from the embeddings
            db = FAISS.from_embeddings(text_embeddings, self.emb_model, metadatas=metadatas)

            # save db for future use
            db.save_local(self.db_dir)  # create a dir with 2 files: index.faiss, index.pkl

        else:
            if verbose()>0 : print("loading the vector database from '" + str(self.db_dir) + "'")
            # load saved db
            db = FAISS.load_local(self.db_dir, embeddings=self.emb_model, allow_dangerous_deserialization=True)

        return db