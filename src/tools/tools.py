import re
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings

from src.configs.parse_config import verbose, emb_model_id, tools_dir, docs_dir, db_dir, update_db
from src.database import Database
from src.tools.manage_tools import tool



##################################################################################################################
# Define Tools

# ----------------------------------------------------------------------------------------------
@tool
def calculator(expr: str): # arguments should have a defined type
    if verbose>0 : print(">> using the calculator")

    # remove everything that isn't a math symbol (to avoid evaluating malicious code)
    safe_expr = re.sub(r'[^0-9+\-*/().[] ]', '', expr)
    if verbose>1 : print(">> safe_expr =", safe_expr)
    # Insert implicit multiplication: 5(4+3) --> 5*(4+3)
    safe_expr = re.sub(r'(\d)\s*\(', r'\1*(', safe_expr)
    if verbose>1 : print(">> safe_expr =", safe_expr)
    if safe_expr.strip() == "":
        print("\nAn error occurred: empty expression.")
        return "Couldn't compute the result."

    try:
        result = eval(safe_expr)
        if verbose>1 : print(">> calc result =", result)
        return f"The result is: {result}"
    except Exception as e:
        print(f"\nAn error occurred:\n{e}\n")
        return "Couldn't compute the result."

# describe the use of the tool
calculator.__doc__ = "Calculate the result of a mathematical expression."


# ----------------------------------------------------------------------------------------------
@tool
def draw_graph(expr: str, x_range: list[float]):
    if verbose>0 : print(">> using the draw_graph")
    try:
        if verbose>1 : print(">> converting the function")
        # remove everything that isn't a math symbol (to avoid evaluating malicious code)
        safe_expr = re.sub(r'[^0-9+\-*/(). [],x]', '', expr)
        func = lambda x: eval(safe_expr)

        if verbose>1 : print(">> converting the range")
        if isinstance(x_range, str): # convert the range back to a list if it's passed as a string by mistake
            x_range = [float(x) for x in x_range.strip("[]").split(",")]
            if verbose>1 : print(">> x_range =", x_range)

        x_points = np.linspace(x_range[0], x_range[1], 100)
        y_points = np.array([func(x) for x in x_points])
        plt.plot(x_points, y_points)
        plt.savefig(tools_dir+"/graph.png")
        plt.close()
        return "the graph was saved as graph.png in the directory '"+tools_dir+"'"
    except Exception as e:
        print(f"\nAn error occurred:\n{e}\n")
        return "I couldn't draw the graph."

# describe the use of the tool
draw_graph.__doc__ = "Plot a function given its expression and x range."


# ----------------------------------------------------------------------------------------------
@tool
def search_database(query: str, n_retriv: int):
    if n_retriv < 1: return "No document has been retrieved"

    # "emb_model" is for creating the embeddings for the vector database (for RAG)
    emb_model = HuggingFaceEmbeddings(model_name=emb_model_id)

    start = datetime.now()
    # this will first check if the database exists and if it needs to be updated
    # then either load or create the database
    db = Database(emb_model, docs_dir, db_dir, update_db).load()
    end = datetime.now()
    if verbose>0: print(">> Time to Database =", end - start)

    if verbose>0 : print(">> searching the database")
    start = datetime.now()
    # find the relevant docs

    retriv_docs = db.similarity_search(query, k=n_retriv) # find the k most relevant documents to the query
    if verbose>1 : print(">> n_retriv_docs =", len(retriv_docs), "=", n_retriv)
    #if verbose>2 : print("most relevant doc\n", retriv_docs[0])


    end = datetime.now()
    if verbose>0 : print(">> Time to Retrieval =", end-start)
    return retriv_docs

# describe the use of the tool
search_database.__doc__ = "Look at the codebase for the project and retrieve relevant information."


# ----------------------------------------------------------------------------------------------
#@tool
def run_megalinter(flavor: str):
    path = "megalinter-reports/megalinter.log"
    command = "npx mega-linter-runner --flavor "+flavor
    # subprocess.run([command, "arg1", "arg2"], cwd="path/to/folder/", shell=True) # run shell command
    subprocess.run([command], shell=True) # run shell command
    f = open(path)
    logs = "Logs saved in "+path+"\n\nContents of the logs:\n\n"+f.read()
    f.close()
    return logs

# describe the use of the tool
run_megalinter.__doc__ = """
    Use a static analysis tool to find errors and warnings in the codebase. 
    Available flavors are:
      - 'all': to include all possible linters;
      - 'c_cpp': for pure C/C++ projects;
      - 'ci_light': for CI items (Dockerfile, Jenkinsfile, JSON/YAML schemas,XML);
      - 'cupcake': for the most commonly used languages;
      - 'documentation': for documentation projects;
      - 'dotnet': for C, C++, C# or VB based projects;
      - 'dotnetweb': for C, C++, C# or VB based projects with JS/TS;
      - 'formatters': contains only formatters;
      - 'go': for GO based projects;
      - 'java': for JAVA based projects;
      - 'javascript': for JAVASCRIPT or TYPESCRIPT based projects;
      - 'php': for PHP based projects;
      - 'python': for PYTHON based projects;
      - 'ruby': for RUBY based projects;
      - 'rust': for RUST based projects;
      - 'salesforce': for Salesforce based projects;
      - 'security': for security;
      - 'swift': for SWIFT based projects;
      - 'terraform': for TERRAFORM based projects;
"""



