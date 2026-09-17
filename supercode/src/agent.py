#from awq import AutoAWQForCausalLM
#from transformers import SinqConfig

"""
- pipeline is for direct inference, with AutoModelForCausalLM, AutoTokenizer you load the raw model
- BitsAndBytesConfig and awq are for quantization
- TextIteratorStreamer and threading are for printing the answer as it is being generated
"""

from src.model_loader.model_loader import Model
from src.tools.manage_tools import * # import all the tools
from src.configs.system_prompts import *
from src.tools.tools import *
from src.tools.code_processing import *

"""
Roles:
- system: general instructions
- user: the user
- assistent: the model
- tool: feedback from tools
- thinking/reasoning: thinking mode or reasoning schemas (ex: CoT)
"""

class Agent():
    # define variables and import the model
    def __init__(self):
        # vars not defined here are defined in the config file!

        # roles:
        self.system_role = "system"
        self.user_role = "user"
        self.tool_role = "tool"
        self.assistant_role = "assistant" # coder assistant
        self.tool_manager_role = "reasoning" # thinking model for selecting tools

        # tools related parameters
        self.tool_results = []
        self.tools = get_tools() # tools for the agent
        self.schemas = [build_tool_schema(f) for f in self.tools.values()]
        self.details = [f._tool_metadata for f in self.tools.values()] # info about requirements
        #pprint.pprint(self.schemas)

        # ----------------------------------------------------------------------------------------------
        ##### IMPORTING THE MODELS
        # coding assistant (expert model)
        self.assistant = Model(
            coder_model_args,
            system_prompt=assistant_prompt,
            prequery_prompt=assistant_prequery,
        )

        # tool manager is a reasoning model for selecting tools
        tool_manager_prompt = manager_prompt_1+f"{self.schemas}"+manager_prompt_2
        if tools_iter>0: self.tool_manager = Model(
            think_model_args,
            system_prompt=tool_manager_prompt,
            prequery_prompt=tool_manager_prequery,
        )

        if verbose > 1: print(">> defining system prompt")
        self.reset_memory() # keep only the system prompts

        self.models_list = [self.assistant ]
        if tools_iter>0: self.models_list.append(self.tool_manager)

    def reset_memory(self):
        self.assistant.reset_memory()
        if tools_iter>0: self.tool_manager.reset_memory()

    def print_chat_history(self):
        print("\n****************************************************************************\n## assistant messages history:")
        pprint.pprint(self.assistant.get_messages())
        print("****************************************************************************\n")

        if tools_iter>0:
            print("\n\n****************************************************************************\n## tool_manager messages history:")
            pprint.pprint(self.tool_manager.get_messages())
            print("****************************************************************************\n")

    """
    # ----------------------------------------------------------------------------------------------
    def parse_tools(self, response, revise=False):
        if '"name": "' in response:
            if verbose > 0: print(">> parsing response for tools")
            # Note: the LLM is not always consistent, sometimes there are multiple separate json objs instead of a list,
            # or the JSON list is repeated multiple times, but the JSON obj is always written as:
            # { "name": "tool_name", "arguments": { "arg_name": "value"} }

            tool_end = 0
            end = response.rfind("}")+1 # find last occurrence (the +1 is just bc I use it in tool_end to include '}' in the string)
            tool_request_list = []

            while tool_end < end:
                # find where the tool begins
                tool_check = response.find('"name": "', tool_end) # this is a more robust signature for finding tools
                if tool_check == -1: break # this cover the case in which I have no tool left but there are some '{}'
                tool_begin = response.rfind("{", tool_end, tool_check) # find '{' between the last tool and the signature
                tool_temp = response.find("}", tool_begin)+1 # there are two '}', this find the first
                tool_end  = response.find("}", tool_temp)+1  # and this find where the tool actually ends

                tool_request = json.loads( response[tool_begin:tool_end].strip("  ") ) # remove extra spaces and load
                print(">> JSON OBJ (PARTIAL): \n", tool_request, sep="")

                if tool_request not in tool_request_list: # to avoid duplicates
                    tool_request_list.append(tool_request)

                    tool_name = tool_request["name"]
                    if tool_name in self.tools.keys():
                        if verbose > 0: print(">> found tool:", tool_name)

                        # skip tools that require dependencies (req_flag=True) the first time (revise=False)
                        # this way the first time it will compute only results for tools with no dependencies
                        for s in self.schemas:
                            fn = s.get('function', {}) # the second value is returned if the first cannot be found
                            if fn.get('name') == tool_name:
                                req_flag = fn.get('x_metadata', {}).get('req_flag', False)
                        skip = req_flag and not revise
                        if skip :
                            tool_result = "This tool has requirements."
                            if verbose > 2: print(">> skipping tool:", tool_name)
                        else:
                            tool_result = dispatch_tool(self.tools, tool_name, tool_request["arguments"])
                            #if verbose > 2: print(">> tool result:", tool_result)

                        # save the results to pass them to the model
                        tool_result = {
                            "name": tool_name,
                            "result": tool_result
                        }
                        if tool_result not in self.tool_results:
                            self.tool_results.append(tool_result)
                else:
                    if verbose > 2: print(">> this tool was already called:", tool_name)
        else:
            if verbose > 0: print(">> no tool was found")

        # self.tool_messages.pop() # remove the last item
    """

    # ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------
    def call(self, user_prompt, code=None, reset_memory=False, baseline=False, **kwargs):
        """
        WORKFLOW:
            1) the model checks if it needs to call a tool or retrieve docs
            2) the tool is called and/or docs are retrieved
            3) the model integrates the results with its answer
        if baseline is True, then the model doesn't use any external info
        """
        # ----------------------------------------------------------------------------------------------
        # reset tool results and tool_manager
        self.tool_results = []
        if tools_iter>0: self.tool_manager.reset_memory()

        if reset_memory: self.reset_memory() # forget previous answers and keep only the system prompts, useful for benchmarks

        # save the user_prompt in the message history of all active models
        for m in self.models_list: 
            m.add_message(role=self.user_role, content=user_prompt)

            # save the baseline code as context #TODO: decide if tool manager needs the code
            if code is not None:# and m!=self.tool_manager:
                m.add_message(role=self.tool_role, content="```\n"+code+"\n```", name="baseline code")
                #TODO: check if I can specify the language!!!

        for i in range(model_iter): # sequential iterations on the code
            # ----------------------------------------------------------------------------------------------
            # use standalone model (no iteration and no tools allowed, baseline to measure tools effectiveness)
            if baseline:
                print(">> testing the baseline")
                # apply chat templates and return an answer
                print("\n-------------------------------------- \n"
                      "## assistant (i="+str(i)+", baseline="+str(baseline)+"): ")
                response = self.assistant.call()
                print("--------------------------------------")
                break # there is no need to iterate multiple times

            # ----------------------------------------------------------------------------------------------
            # use model with tools (compiler, profiler, info from database, websearch, etc...)
            else:
                print(">> testing the workflow")
                # ----------------------------------------------------------------------------------------------
                # let the thinking model choose a tool
                if tools_iter>0:
                    print(f">> selecting tools")
                    revise = False # the first iteration skip tools with requirements

                    # refinement loop, useful when tools have requirements and need info from other tools
                    for r in range(tools_iter):
                        tool_index = len(self.tool_results)  # to avoid duplications

                        # choose the tools
                        if verbose > 1: print(f"-------------------------------------- \n## tool manager (r={r+1}): ")
                        response = self.tool_manager.call(tool_schemas=self.schemas)
                        if verbose > 1: print("--------------------------------------")

                        # skip the rest if there are no tools are called
                        if "```json\n[]\n```" in response: print(">> NO NEW TOOLS INCLUDED") ; break

                        # parse the tool manager answer, find the tools, call them and report the results
                        self.tool_results = parse_tools(response, self.tools, self.schemas, self.tool_results, revise)
                        if verbose>1: print(">> TOOL RESULTS: \n", self.tool_results, "\n", sep="")

                        # add the tool results to the chat history of all models
                        for tool in self.tool_results[tool_index:]: # only include new results
                            for m in self.models_list: # all models need the tool results
                                m.add_message(role=self.tool_role, content=tool["result"], name=tool["name"])

                        # revise the answer to implement the correct dependencies
                        if r<tools_iter-1: self.tool_manager.add_message(role=self.user_role, content=tool_manager_revise)
                        revise = True

                # ----------------------------------------------------------------------------------------------
                # predetermined use of tools to analyze code (if given)
                elif code is not None:
                    print(">> predetermined tools")
                    compiler_result = sandboxed_compiler(code)
                    # perf_result = run_perf(gen_code_file)

                    for m in self.models_list:
                        # save the tool results in the message history of all models
                        m.add_message(role=self.tool_role, content=str(compiler_result), name="sandboxed_compiler")
                        #m.add_message(role=self.tool_role, content=str(perf_result), name="run_perf")

                        # prompt to improve the code if the compiler returns an error
                        if compiler_result[0] != 0: m.add_message(role=self.user_role, content=compiler_prompt)

                    if compiler_result[0] == 0: # check if the code compiled correctly
                        response = "There is nothing to improve."+"\nPrevious code:\n```\n"+code+"\n```"
                        print("\n-------------------------------------- \n## assistant (i=" + str(i) + ", baseline=" + str(baseline) + "): ")
                        print(response, "\n--------------------------------------")
                        break

                # ----------------------------------------------------------------------------------------------
                # ask the coder assistant to improve the code
                print("\n-------------------------------------- \n## assistant (i="+str(i)+", baseline="+str(baseline)+"): ")
                response = self.assistant.call()
                print("--------------------------------------")

                # failsafe in case the model doesn't return any code
                if extract_code(response)=="":
                    if code is not None:
                        print(">> appending prev code")
                        response = response+"\nPrevious code:\n```"+code+"```"
                        break
                    else:
                        print("WARNING: failed to extract code and no previous code to fall back to")
                        response = response+"\nNo code:\n```raise Exception('NO CODE')```"
                        continue # failed to extract code and no previous code to fall back to
                else:
                    code = extract_code(response)

        return response

