import re, os, subprocess, tempfile
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from langchain_huggingface import HuggingFaceEmbeddings

# importing my functions from other files
from supercode.src.database.database import Database

# importing variables from the config file
from supercode.src.configs.parse_config import *


__TOOLS = []

##################################################################################################################
# tool decorator
def tool(metadata: dict):
    """Costum decorator for setting the metadata of the tool."""

    def decorator(func):
        func.__is_tool__ = True

        func._tool_metadata  = {
            # use the values above if metadata doesn't set a new value
            # "name": None, # unnecessary
            "provides": None,
            "requires": None,
            "tags": None,
            "examples": None,
            **metadata
        }

        __TOOLS.append(func)

        return func
    return decorator


def get_TOOLS():
    return __TOOLS

##################################################################################################################
# Define Tools

# ----------------------------------------------------------------------------------------------
metadata = {
        "provides": "numerical solution of a simple math equation",
        "requires": "math expression provided by the user",
        "tags": ["math", "calculator", "equation", "solve", "result"],
        "examples": ["what is the result of (8261+8257)/16290*545-46303?"],
}
@tool(metadata)
def calculator(expr: str): # arguments should have a defined type
    """Calculate the result of a mathematical expression."""
    if verbose>0 : print(">> using tool: calculator")
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


# ----------------------------------------------------------------------------------------------
metadata = {
        "provides": "path where the graph was saved",
        "requires": "math function provided by the user and range of the function provided by the user, use [-10, +10] if no range is given",
        "tags": ["math", "graph", "plot", "figure"],
        "examples": ["plot y=3*x+2 for x in [0,5]"],
}
@tool(metadata)
def draw_graph(expr: str, x_range: List[float]):
    """Plot a function given its mathematical expression and x range."""
    if verbose>0 : print(">> using tool: draw_graph")
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

        return f"the graph was saved as graph.png in the directory `{tools_dir}/`"
    except Exception as e:
        print(f"\nAn error occurred in the draw_graph tool:\n{e}\n")
        return f"I couldn't draw the graph. An error occurred:\n{e}"

# ----------------------------------------------------------------------------------------------
metadata = {
        #"name": search_database.__name__,
        "provides": "chunks of documents with relevant information and metadata of the documents (like source file and programming language)",
        "requires": """
        query for searching the database and the number of retrieved docs.
        n_retriev should be: 
        - around 1-5 for easy queries, with straightforward answer; 
        - around 5-10 for medium difficulty queries, where the answer requires few pieces of information; 
        - around 10-15 for generic or complex queries, where the answer may requires many pieces of information. 
        """,
        "tags": ["search", "database", "codebase", "project"],
        "examples": ["find the file where I defined this function"],
}
@tool(metadata)
def search_database(query: str, n_retriv: int, topic: Literal['codebase', 'radioastronomy']):
    """
        Retrieve relevant information from the database, based on a topic.
        In the "codebase" topic you can find info about:
        - the programming languages used;
        - functions writen in the code;
        - what the code does and how.
        In the "radioastronomy" topic you can find info about:
        - LOFAR beamformer pipeline
        - LOFAR imaging pipeline
    """
    if verbose>0 : print(">> using tool: search_database")
    if n_retriv < 1: return "No document has been retrieved"
    try:
        # "emb_model" is for creating the embeddings for the vector database (for RAG)
        emb_model = HuggingFaceEmbeddings(model_name=emb_model_id)

        start = datetime.now()
        # this will automatically check if the database exists and if it needs to be updated
        # then either load or create the database
        db = Database(emb_model, docs_dir, db_dir, topic, update_db).load()
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

# ----------------------------------------------------------------------------------------------
metadata = {
        #"name": run_megalinter.__name__,
        "provides": "report with errors, warnings and best practices suggestions for the codebase",
        "requires": "flavor from the user or programming language, which can be found by searching the database",
        "tags": ["static analysis", "errors", "warnings"],
        "examples": ["check the code for errors", "check if the code follows the best practices"],
}
flavors = ['all', 'c_cpp', 'ci_light', 'cupcake', 'documentation', 'dotnet', 'dotnetweb', 'formatters', 'go', 'java', 'javascript', 'php', 'python', 'ruby', 'rust', 'salesforce', 'security', 'swift', 'terraform']
#@tool(metadata)
def run_megalinter(flavor: Literal['all', 'c_cpp', 'ci_light', 'cupcake', 'documentation', 'dotnet', 'dotnetweb', 'formatters', 'go', 'java', 'javascript', 'php', 'python', 'ruby', 'rust', 'salesforce', 'security', 'swift', 'terraform']):
    """Use a static analysis tool to find errors and warnings in the codebase.
    
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
    """
    
    if verbose>0 : print(">> using tool: run_megalinter")
    path = tools_dir+"/megalinter-reports/"
    try:

        if flavor not in flavors : # some are more likely to be misnamed by the LLM, this is a quick and dirty fix
            if flavor=="cpp" or flavor=="c" : flavor="c_cpp"
            if flavor in ["ci", "docker", "jenkins", "json", "yaml", "xml"] : flavor="ci_light"

        # TEMPORARY PATCH
        command = f"cd {tools_dir} && npx mega-linter-runner --flavor {flavor}" # save megalinter report in 'tools_dir'
        # run shell command with: subprocess.run([command, "arg1", "arg2"], cwd="path/to/folder/", shell=True)
        subprocess.run([command], shell=True) # run shell command
        f = open(path+"megalinter.log")
        logs = f"Logs saved in {path}\n\nContents of the logs:\n\n{f.read()}"
        f.close()

    except Exception as e:
        print(f"\nAn error occurred in the run_megalinter tool:\n{e}\n")
        return f"I couldn't run megalinter. An error occurred:\n{e}"

    return logs


# ----------------------------------------------------------------------------------------------
metadata = {
        #"name": run_perf.__name__,
        "provides": "report with how long the code spends for each function and operation, which can be used to improve the code performances",
        "requires": "main file of the code, which can be found by searching the database",
        "tags": ["dynamic analysis", "performance", "bottleneck"],
        "examples": ""
}
#@tool(metadata)
def run_perf(main_file: str):
    """Use a dynamic analysis tool to find bottlenecks in the codebase."""
    if verbose>0 : print(">> using tool: run_perf")
    exe_name = "myprog"
    path = tools_dir+"/perf-reports"
    os.makedirs(path, exist_ok=True)
    try:
        if ".cpp" in main_file: # compile (for C++)
            subprocess.run(
                [f"g++ -O2 -g {main_file} -o {exe_name}"],
                cwd=docs_dir, shell=True, check=True
            )
            subprocess.run( # record perf data
                [f"perf record -g -e cycles -F 9999 -o ../{path}/perf.data ./{exe_name}"],
                cwd=docs_dir, shell=True, check=True
            )
        elif ".py" in main_file:  # compile (for C++)
            subprocess.run( # record perf data
                [f"perf record -g -e cycles -F 9999 -o ../{path}/perf.data python ./{main_file}"],
                cwd=docs_dir, shell=True, check=True
            )
        else:
            print("WARNING: file format not implemented in run_perf tool")
        subprocess.run( # generate annotated report from perf data
            ["perf annotate --stdio > annotate.txt"],
            cwd=path, shell=True, check=True,
        )

        f = open(f"{path}/annotate.txt")
        logs = f"Logs saved in {path}\n\nContents of the logs:\n\n{f.read()}"
        f.close()

    except Exception as e:
        print(f"\nAn error occurred in the run_perf tool:\n{e}\n")
        return f"I couldn't run perf. An error occurred:\n{e}"

    return logs


# ----------------------------------------------------------------------------------------------
def docker_container(command: Literal['activate', 'deactivate', 'delete']):
    """Activate or deactivate the sandbox docker container for running code."""
    image = docker_args["image"]
    container_name = docker_args["container_name"]
    workspace_path = os.path.join(os.getcwd(), docker_args["workspace"])
    #print("workspace =", workspace_path)

    if verbose>1: print(f">> about to {command} docker container '{container_name}'")
    # for debugging
    # if verbose>2:
    #     print("----------------------------------------------------------------------------------------------\n", subprocess.run(
    #         # 'docker ps' returns info of all (-a) containers filtered (-f) by name
    #         ["docker", "ps", "-a", "-f", f"name={container_name}"],
    #         capture_output=True, text=True, check=True
    #     ).stdout.strip(), "\n", sep="")

    # --------------------------------------------------------------------------------------
    def execute_docker_command(command: Literal['build', 'create', 'activate', 'deactivate', 'delete']):
        if command=="build":
            # build the image for the container (docker build) based on the Dockerfile
            if verbose>1: print(">> building image")
            subprocess.run([
                "docker", "build", "-t", image, docker_args["docker_dir"]
            ], check=True)

        elif command=="create":
            # create a new container (docker run)
            if verbose>1: print(">> creating container")
            subprocess.run([
                "docker", "run", "-d", "-it", # -d for detached mode (run in background and print container ID), -it for Interactive TTY session
                "-m", docker_args["mem_limit"],  # memory limit, defined in config
                "--name", container_name,  # fixed name for the container, defined in config
                "--mount", f"type=bind,src={workspace_path},dst=/workspace",  # mount disk to save/access files
                image  # select image for the docker container
            ], check=True)
            # other options:
            # --rm for auto-cleanup on stop (very slow if done often)
            # -v for binding the workspace directory as volume

        elif command == "activate":
            # activate the container (docker start)
            if verbose>1: print(">> starting container")
            subprocess.run(["docker", "start", container_name], check=True)

        elif command=="deactivate":
            # deactivate the container if it is running (docker stop)
            if verbose>1: print(">> stopping container")
            subprocess.run(["docker", "stop", container_name], check=True)

        elif command=="delete":
            # delete the container (docker rm)
            if verbose>1: print(">> removing container")
            subprocess.run(["docker", "rm", container_name], check=True)

        else:
            print(f"WARNING: unknown command {command} will be ignored")
    # --------------------------------------------------------------------------------------
    try:
        # check if the container exists (active or not)
        exist_docker_id = subprocess.run(
            # 'docker ps' returns the IDs (-q) of all (-a) containers filtered (-f) by name
            ["docker", "ps", "-a", "-q", "-f", f"name={container_name}"],
            capture_output=True, text=True, check=True
        ).stdout.strip()

        # check if the container is active (this is a subset of the previous)
        active_docker_id = subprocess.run(
            # 'docker ps' returns the IDs (-q) of active containers filtered (-f) by name
            ["docker", "ps", "-q", "-f", f"name={container_name}"],
            capture_output=True, text=True, check=True
        ).stdout.strip()

        image_in_mem = image in subprocess.run(["docker", "images"], capture_output=True, text=True, check=True).stdout.strip()

        # decision tree based on the command and whether the container exists/is running
        if exist_docker_id: # check if the container exists
            if active_docker_id: # check if is it active
                if command == 'deactivate' or command == 'delete':
                    execute_docker_command('deactivate')
                    if command == 'delete': execute_docker_command(command)
                # pass if command is activate
            else: # if the container exists, but it is not active
                if command == 'activate' or 'delete': execute_docker_command(command)
                # pass if command is deactivate
        else: # if the container doesn't exist
            if command == 'activate':
                if not image_in_mem: execute_docker_command('build') # build the image from the Dockerfile
                execute_docker_command('create') # then create the container from the image
                execute_docker_command('activate') # the created container automatically stops if no commands are given
            # pass if deactivate/delete

    except subprocess.CalledProcessError as e:
        print(f"ERROR: docker_container tried to {command} {container_name} but failed:\n{e}")

    # for debugging
    if verbose>2:
        print("\n", subprocess.run(
            # 'docker ps' returns info of all (-a) containers filtered (-f) by name
            ["docker", "ps", "-a", "-f", f"name={container_name}"],
            capture_output=True, text=True, check=True
        ).stdout.strip(), "\n----------------------------------------------------------------------------------------------\n", sep="")
# ----------------------------------------------------------------------------------------------
def extract_runtime(code_output):
    pattern = r'real\s(\d+)m(\d+\.\d+)s\suser\s(\d+)m(\d+\.\d+)s\ssys\s(\d+)m(\d+\.\d+)s\s'
    match = re.search(pattern, code_output) # there should be only one match
    # time <command> output:
    # <command output>
    # real	0m0.004s
    # user	0m0.001s
    # sys	0m0.004s

    real_t = float(match.group(1))*60+float(match.group(2))
    user_t = float(match.group(3))*60+float(match.group(4))
    sys_t  = float(match.group(5))*60+float(match.group(6))

    runtime = {
        "real": real_t, # clock time (including I/O and time slices used by other processes)
        "user": user_t, # process time for the current job only
        "sys":  sys_t,  # kernel time for the current job only (this doesn't include library code --> user)
        "cpu":  user_t+sys_t, # cpu time, is <=real with single-thread and can be >real with multi-thread
        "util": (user_t+sys_t)/real_t # cpu utilization, note this can be >1 with multi-threading
    }
    if verbose>1: print(">> code runtime:", runtime)
    return runtime, match.group(0)

# ----------------------------------------------------------------------------------------------
metadata = {
        #"name": sandboxed_compiler.__name__,
        "provides": "runs the code and returns the compiler output",
        "requires": "function to be run",
        "tags": ["dynamic analysis", "errors", "warnings", "remarks"],
        "examples": "run the code to check if the output is correct"
}
@tool(metadata)
def sandboxed_compiler(code: object, language: object, unit_test: object=None, entry_point: object=None, k_times: int=1):
    """Run the code in a sandboxed Docker environment."""
    if verbose>0 : print(">> using tool: sandboxed_compiler")
    container_name = docker_args["container_name"]
    # IMPORTANT: the docker image can see ONLY files inside "results/"
    # and all paths in the container should be relative to "results/"!

    if unit_test is not None: code = f"{code}\n\n{unit_test}\n\ncheck({entry_point})" # issue: this would work only for HumanEval!

    # ------------------------------------------------------------------------------------
    def run_code_command(language):
        #if verbose>0 : print(">> define command based on the language and saving the code")

        docker_cmd = f"docker exec -t {container_name}"
        time_cmd = "time"
        # time: to measure real (clocktime), user (processes) and system (kernels) times
        # note: CPU percentage = (user + sys)/real * 100
        # -i, --interactive, keep STDIN open even if not attached

        # NOTE: when using cmd list, use "sh", "-c", f"g++ for c/cpp
        # needed because I'm running 2 commands in one step (compile && run exe)
        # ----------------------------
        if language == "python":
            ext=".py" # file extension
            cmd = f"python3 {gen_code_file+ext}"
        # ----------------------------
        elif language == "c":
            ext=".c" # file extension
            cmd = f"gcc -c {gen_code_file+ext} -o {gen_code_file}.out && ./{gen_code_file}.out"
        # ----------------------------
        elif language == "cpp":
            ext=".cpp" # file extension
            cmd = f"g++ -c {gen_code_file+ext} -o {gen_code_file}.out && ./{gen_code_file}.out"
        # ----------------------------
        else: raise ValueError(f"Error: Unsupported language '{language}'")

        # --------------------------------------------------------------------------------
        # save the code in docker workspace
        gen_code_file_path = os.path.join(os.getcwd(), docker_args["workspace"], gen_code_file+ext)
        with open(gen_code_file_path, "w") as f:
            if verbose>1: print(">> saving the code at", gen_code_file_path)
            f.write(code)
            f.close()

        # 'bash -c' to force bas like interface
        return f"{docker_cmd} bash -c '{time_cmd} {cmd}'" # command for running the code
    # ------------------------------------------------------------------------------------
    try:
        docker_container('activate') # to make sure the container is running
        # check docker container status
        if verbose>2:
            # debugging
            print("----------------------------------------------------------------------------------------------\n",
                  subprocess.run(
                      # 'docker ps' returns info of all (-a) containers filtered (-f) by name
                      ["docker", "ps", "-a", "-f", f"name={container_name}"],
                      capture_output=True, text=True, check=True
                  ).stdout.strip(), "\n", sep="")

        k_runtimes = {"real": [], "user": [], "sys": [], "cpu": [], "util": []}
        ave_std_time_str = "" ; ave_std_time_dict = {}

        # run the code multiple times to capture runtime variance
        for k in range(k_times):
            result = subprocess.run( # run a program from inside the container (and remove the container afterwards)
                run_code_command(language),
                #["docker", "run", "-m", "0.5g", "--mount", "type=bind,src="+abs_docker_path+",dst=/workspace", "--rm", "sandbox_code", "python3", docker_code_path],
                # -m is memory, must be between 6m and 1g (adding "--memory-swap", "512m" will forbit memory swaps)
                shell=True,
                capture_output=True,
                text=True,
                timeout=docker_args['timeout'] # seconds
            )

            # remove escape characters
            ansi_escape = re.compile(r'\x1b\[[0-9;]*m') # \x is for asci characters
            output = ansi_escape.sub('', result.stdout)

            runtime, time_string = extract_runtime(output)

            for key in k_runtimes.keys(): k_runtimes[key].append(runtime[key])
            if result.returncode==1 : k_times=k+1 ; break # run the code only once if there are errors

        output = output.replace(time_string, "") # remove the time string from the output

        for key in k_runtimes.keys():
            mean = np.mean(k_runtimes[key])
            std = np.std(k_runtimes[key])
            if key=="util":
                ave_std_time_str += f"cpu {key}:\t({mean*100:.2f} +/- {std*100:.2f})%\n"
            else:
                ave_std_time_str += f"{key} time:\t({mean*1000:.2f} +/- {std*1000:.2f}) ms\n"
            ave_std_time_dict[key] = (mean, std)

        # check docker container status
        if verbose>2:
            # debugging
            print("\n", subprocess.run(
                # 'docker ps' returns info of all (-a) containers filtered (-f) by name
                ["docker", "ps", "-a", "-f", f"name={container_name}"],
                capture_output=True, text=True, check=True
            ).stdout.strip(), "\n----------------------------------------------------------------------------------------------\n", sep="")
        #docker_container('deactivate')

        compiler_output = (
            result.returncode,
            f"printed_output=\'{output}{result.stderr}\'\n"
            f"runtime statistics (mean +/- std) for {k_times} runs:\n{ave_std_time_str}",
            ave_std_time_dict
        )
        if verbose>1 : print(f">> command completed:\n{compiler_output}")
        return compiler_output

    except Exception as e:
        print(f"Error in sandboxed_compiler tool: {e}")
        return 1, f"Error in sandboxed_compiler tool: {e}"

