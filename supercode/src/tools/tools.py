import re, os, subprocess, tempfile
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings

from src.configs.parse_config import *
from src.database.database import Database
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
            #"name": calculator.__name__,
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

        # use the following command from terminal to download the graph (change <USER_NAME> to the actual name)
        # scp -o ProxyJump=user1@server_1 user2@server_2:/path/to/file /local/path
        # scp src-temp:/home/<USER_NAME_REMOTE>/superRAG/src/tools/tools-outputs/graph.png /home/<USER_NAME_LOCAL>/Downloads/

        return "the graph was saved as graph.png in the directory `"+tools_dir+"/`"
    except Exception as e:
        print(f"\nAn error occurred in the draw_graph tool:\n{e}\n")
        return f"I couldn't draw the graph. An error occurred:\n{e}"

# describe the use of the tool
draw_graph.__doc__ = "Plot a function given its mathematical expression and x range."
draw_graph._tool_metadata = {
            #"name": draw_graph.__name__,
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
    if verbose>0 : print(">> using the search_database")
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
            #"name": search_database.__name__,
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
#@tool
def run_megalinter(flavor: str):
    if verbose>0 : print(">> using the run_megalinter")
    path = tools_dir+"/megalinter-reports/"
    try:
        flavors = ['all', 'c_cpp', 'ci_light', 'cupcake', 'documentation', 'dotnet', 'dotnetweb', 'formatters', 'go', 'java', 'javascript', 'php', 'python', 'ruby', 'rust', 'salesforce', 'security', 'swift', 'terraform']
        if flavor not in flavors : # some are more likely to be misnamed by the LLM, this is a quick and dirty fix
            if flavor=="cpp" or flavor=="c" : flavor="c_cpp"
            if flavor in ["ci", "docker", "jenkins", "json", "yaml", "xml"] : flavor="ci_light"

        # TEMPORARY PATCH
        command = "cd "+tools_dir+" && npx mega-linter-runner --flavor "+flavor # save megalinter report in 'tools_dir'
        # run shell command with: subprocess.run([command, "arg1", "arg2"], cwd="path/to/folder/", shell=True)
        subprocess.run([command], shell=True) # run shell command
        f = open(path+"megalinter.log")
        logs = "Logs saved in "+path+"\n\nContents of the logs:\n\n"+f.read()
        f.close()

    except Exception as e:
        print(f"\nAn error occurred in the run_megalinter tool:\n{e}\n")
        return f"I couldn't run megalinter. An error occurred:\n{e}"

    return logs

# describe the use of the tool
run_megalinter.__doc__ = "Use a static analysis tool to find errors and warnings in the codebase."
run_megalinter._tool_metadata = {
            #"name": run_megalinter.__name__,
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


# ----------------------------------------------------------------------------------------------
#@tool
def run_perf(main_file: str):
    if verbose>0 : print(">> using the run_perf")
    exe_name = "myprog"
    path = tools_dir+"/perf-reports"
    os.makedirs(path, exist_ok=True)
    try:
        if ".cpp" in main_file: # compile (for C++)
            subprocess.run(
                ["g++ -O2 -g "+main_file+" -o "+exe_name],
                cwd=docs_dir, shell=True, check=True
            )
            subprocess.run( # record perf data
                ["perf record -g -e cycles -F 9999 -o ../"+path+"/perf.data ./"+exe_name],
                cwd=docs_dir, shell=True, check=True
            )
        elif ".py" in main_file:  # compile (for C++)
            subprocess.run( # record perf data
                ["perf record -g -e cycles -F 9999 -o ../"+path+"/perf.data python ./"+main_file],
                cwd=docs_dir, shell=True, check=True
            )
        else:
            print("WARNING: file format not implemented in run_perf tool")
        subprocess.run( # generate annotated report from perf data
            ["perf annotate --stdio > annotate.txt"],
            cwd=path, shell=True, check=True,
        )

        f = open(path+"/annotate.txt")
        logs = "Logs saved in "+path+"\n\nContents of the logs:\n\n"+f.read()
        f.close()

    except Exception as e:
        print(f"\nAn error occurred in the run_perf tool:\n{e}\n")
        return f"I couldn't run perf. An error occurred:\n{e}"

    return logs

# describe the use of the tool
run_perf.__doc__ = "Use a dynamic analysis tool to find bottlenecks in the codebase."
run_perf._tool_metadata = {
            #"name": run_perf.__name__,
            "provides": "report with how long the code spends for each function and operation, which can be used to improve the code performances",
            "requires": "main file of the code, which can be found by searching the database",
            "tags": ["dynamic analysis", "performance", "bottleneck"],
            "examples": ""
}


@tool

def sandboxed_compiler(function, test=None, entry_point=None):
    if verbose>0 : print(">> using the sandboxed_compiler")
    # this only checks if the code compiles (semantic correctness)
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        print("create temp file")
        tmp.write(function)
        tmp_path = tmp.name
    """
    # IMPORTANT: the docker image can see ONLY files inside "results/"
    # and all paths in the container should be relative to "results/"!

    cwd = os.getcwd() #; print(cwd)
    abs_docker_path = cwd+"/"+docker_dir
    #tmp_dir = abs_docker_path+"/results/gen_code"

    if test == None: code = function
    else: code = function+"\n\n"+test+"\n\ncheck("+entry_point+")" # issue: this would work only for HumanEval!

    with open(gen_code_file, "w") as f:
        if verbose>1 : print(">> saving the code at", gen_code_file)
        f.write(code)
        f.close()
    try:
        docker_code_path = gen_code_file[gen_code_file.find("results/"):]
        if verbose>1 : print(">> docker_code_path", docker_code_path)
        if verbose>1 : print(">> run command with docker run")
        #subprocess.run("pwd")
        # docker run --rm sandbox_code python3 results/gen_code/gen_code_temp.py
        # docker run --mount type=bind,src=/home/elisa/PycharmProjects/phd-leiden-supercode/supercode/src/docker_env/results,dst=/workspace --rm sandbox_code
        result = subprocess.run( # run a program from inside the container (and remove the container afterwards)
            ["docker", "run", "--mount", "type=bind,src="+abs_docker_path+",dst=/workspace", "--rm", "sandbox_code", "python3", docker_code_path],
            capture_output=True,
            text=True,
            timeout=10 # seconds
        )
        if verbose>1 : print(">> command run successfully\n", result)
        return result.returncode, "printed_output='" + result.stdout + "'\nerrors='" + result.stderr + "'"
    except Exception as e:
        print("Error in sandboxed_compiler tool:", e)
        return 1, "Error in sandboxed_compiler tool:"+str(e)

