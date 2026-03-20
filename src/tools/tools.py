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
    # replace '−' (U+2212) with '-' (U+002d), they look similar with this font, but the first one causes an error
    expr.replace("−", "-")

    # remove everything that isn't a math symbol (to avoid evaluating malicious code)
    safe_expr = re.sub(r'[^0-9+\-*/().[] ]', '', expr)
    # Insert implicit multiplication: 5(4+3) --> 5*(4+3)
    safe_expr = re.sub(r'(\d)\s*\(', r'\1*(', safe_expr)
    if verbose>1 : print(">> safe_expr =", safe_expr)
    if safe_expr.strip() == "":
        print("\nAn error occurred: empty expression.")
        return "Couldn't compute the result. The expression was empty."

    try:
        result = eval(safe_expr)
        return f"The result is: {result}"
    except Exception as e:
        print(f"\nAn error occurred in the calculator:\n{e}\n")
        return f"Couldn't compute the result. An error occurred:\n{e}"

# describe the use of the tool
calculator.__doc__ = "Calculate the result of a mathematical expression."
calculator._tool_metadata = {
            "provides": "numerical solution of a math equation",
            "requires": "math expression provided by the user",
            "tags": ["math", "calculator", "equation", "solve", "result"],
            "examples": ["what is the result of (8261+8257)/16290*545-46303?"],
            "req_flag": False,
        }


# ----------------------------------------------------------------------------------------------
@tool
def draw_graph(expr: str, x_range: list[float]):
    if verbose>0 : print(">> using the draw_graph")
    try:
        if verbose>1 : print(">> converting the function")
        # replace '−' (U+2212) with '-' (U+002d), they look similar with this font, but the first one causes an error
        expr.replace("−", "-")

        # remove everything that isn't a math symbol (to avoid evaluating malicious code)
        safe_expr = re.sub(r'[^0-9+\-*/(). [],x]', '', expr)
        if verbose>1 : print(">> safe_expr =", safe_expr)
        func = lambda x: eval(safe_expr)

        if verbose>1 : print(">> converting the range")
        if isinstance(x_range, str): # convert the range back to a list if it's passed as a string by mistake
            x_range = [float(x) for x in x_range.strip("[]").replace("−", "-").split(",")] # removes [] and separate at ","
            if verbose>1 : print(">> x_range =", x_range)

        x_points = np.linspace(x_range[0], x_range[1], 100)
        y_points = np.array([func(x) for x in x_points])
        plt.plot(x_points, y_points)
        plt.savefig(tools_dir+"/graph.png")
        plt.close()
        return "the graph was saved as graph.png in the directory `"+tools_dir+"`"
    except Exception as e:
        print(f"\nAn error occurred in the draw_graph tool:\n{e}\n")
        return f"I couldn't draw the graph. An error occurred:\n{e}"

# describe the use of the tool
draw_graph.__doc__ = "Plot a function given its mathematical expression and x range."
draw_graph._tool_metadata = {
            "provides": "path where the graph was saved",
            "requires": "math function provided by the user and range of the function provided by the user, "
                        "use [-10, +10] if no range is given",
            "tags": ["math", "graph", "plot", "figure"],
            "examples": ["plot y=3*x+2 for x in [0,5]"],
            "req_flag": False,
        }

# ----------------------------------------------------------------------------------------------
@tool
def search_database(query: str, n_retriv: int):
    if n_retriv < 1: return "No document has been retrieved"
    try:
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

    except Exception as e:
        print(f"\nAn error occurred in the search_database tool:\n{e}\n")
        return f"I couldn't search the database. An error occurred:\n{e}"

    return retriv_docs

# describe the use of the tool
search_database.__doc__ = """
    Retrieve relevant information from the codebase of the project, for example:
    - the programming languages used;
    - functions writen in the code;
    - what the code does and how.
"""
search_database._tool_metadata = {
            "provides": "chunks of documents with relevant information and metadata of the documents "
                        "(like source file and programming language)",
            "requires": """
            query for searching the database and the number of retrieved docs.
            n_retriev should be: 
            - around 1-5 for easy queries, with straightforward answer; 
            - around 5-10 for medium difficulty queries, where the answer requires few pieces of information; 
            - around 10-15 for generic or complex queries, where the answer may requires many pieces of information. 
            """,
            "tags": ["search", "database", "codebase", "project"],
            "examples": ["find the file where I defined this function"],
            "req_flag": False,
        }

# ----------------------------------------------------------------------------------------------
@tool
def run_megalinter(flavor: str):
    path = "megalinter-reports/megalinter.log"
    try:
        flavors = ['all', 'c_cpp', 'ci_light', 'cupcake', 'documentation', 'dotnet', 'dotnetweb', 'formatters', 'go', 'java', 'javascript', 'php', 'python', 'ruby', 'rust', 'salesforce', 'security', 'swift', 'terraform']
        if flavor not in flavors : # some are more likely to be misnamed by the LLM, this is a quick and dirty fix
            if flavor=="cpp" or flavor=="c" : flavor="c_cpp"
            if flavor in ["ci", "docker", "jenkins", "json", "yaml", "xml"] : flavor="ci_light"


        command = "npx mega-linter-runner --flavor "+flavor
        # run shell command with: subprocess.run([command, "arg1", "arg2"], cwd="path/to/folder/", shell=True)
        #subprocess.run([command], shell=True) # run shell command
        f = open(path)
        logs = "Logs saved in "+path+"\n\nContents of the logs:\n\n"+f.read()
        f.close()

    except Exception as e:
        print(f"\nAn error occurred in the run_megalinter tool:\n{e}\n")
        return f"I couldn't run megalinter. An error occurred:\n{e}"

    return logs

# describe the use of the tool
run_megalinter.__doc__ = "Use a static analysis tool to find errors and warnings in the codebase."
run_megalinter._tool_metadata = {
            "provides": "report with errors, warnings and best practices suggestions for the codebase",
            "requires": "flavor from the user or programming language, which can be found by searching the database",
            "tags": ["static analysis", "errors", "warnings"],
            "examples": """
            Available flavors are:
            - `all`: to include all possible linters;
            - `c_cpp`: for pure C/C++ projects;
            - `ci_light`: for CI items (Dockerfile, Jenkinsfile, JSON or YAML schemas, XML);
            - `cupcake`: for the most commonly used languages;
            - `documentation`: for documentation projects;
            - `dotnet`: for C, C++, C# or VB based projects;
            - `dotnetweb`: for C, C++, C# or VB based projects with JAVASCRIPT or TYPESCRIPT;
            - `formatters`: contains only formatters;
            - `go`: for GO based projects;
            - `java`: for JAVA based projects;
            - `javascript`: for JAVASCRIPT or TYPESCRIPT based projects;
            - `php`: for PHP based projects;
            - `python`: for PYTHON based projects;
            - `ruby`: for RUBY based projects;
            - `rust`: for RUST based projects;
            - `salesforce`: for Salesforce based projects;
            - `security`: for security;
            - `swift`: for SWIFT based projects;
            - `terraform`: for TERRAFORM based projects;
            """,
            "req_flag": True,
        }


