import json, torch, sys
import pprint
import threading
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

#from awq import AutoAWQForCausalLM
#from transformers import SinqConfig

"""
- pipeline is for direct inference, with AutoModelForCausalLM, AutoTokenizer you load the raw model
- BitsAndBytesConfig and awq are for quantization
- TextIteratorStreamer and threading are for printing the answer as it is being generated
"""

from src.model import Model
from src.tools.manage_tools import * # import all the tools
from src.configs.parse_config import *
from src.configs.system_prompts import *
from src.tools.tools import *
from src.tools.code_processing import *


class Agent():
    # define variables and import the model
    def __init__(self):
        # reasoning model for planning/debugging
        #self.debug_model_id = debug_model_id
        self.n_debug = 1

        # tools related parameters
        self.tool_results = []
        self.tool_selection = False # use the same model either to choose which tool to use or to generate an answer
        self.tools = get_tools() # tools for the agent
        self.schemas = [build_tool_schema(f) for f in self.tools.values()]
        self.details = [f._tool_metadata for f in self.tools.values()] # info about requirements

        # ----------------------------------------------------------------------------------------------
        ##### IMPORTING THE MODELS
        # coding assistant (expert model)
        self.assistant = Model(model_args, system_prompt=assistant_prompt, prequery_prompt=assistant_prequery, tool_schemas=None)

        # tool manager (expert model) # Is this the best choice? Maybe the thinking model would be better?
        tool_manager_prompt = manager_prompt_1+f"{self.schemas}"+manager_prompt_2
        self.tool_manager = Model(model_args, system_prompt=tool_manager_prompt, prequery_prompt=tool_manager_prequery, tool_schemas=self.schemas)

        # planner/debugger (thinking model), better at finding logic-based errors
        #self.debugger = Model(debug_model_args, system_prompt=debugger_prompt, prequery_prompt=debugger_prequery, tool_schemas=None)

        if verbose > 1: print(">> defining system prompt")
        self.reset_memory() # keep only the system prompts

    def reset_memory(self):
        self.assistant.reset_memory()
        self.tool_manager.reset_memory()
        #self.debugger.reset_memory()

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
        # decide if it needs to retrieve context and/or to use tools
        if reset_memory: self.reset_memory() # forget previous answers and keep only the system prompts, useful for benchmarks

        for m in [self.assistant, self.tool_manager]:#, self.debugger]:
            # save the user_prompt in the message history of all models
            m.add_message(role="user", content=user_prompt)

        """ 
            n_revise = -1#3
            revise = False  # most often the model calls the tools for the requirements by the second try
            self.tool_selection = True

            for r in range(n_revise+1): # refinement loop, by the 3rd iteration it should have the correct tools selected
                # apply chat templates and return an answer
                if verbose > 1: print(f"-------------------------------------- \n## tool manager {r+1}: ")
                response = self.tool_manager.call()
                if verbose>1 : print("--------------------------------------")

                if "```json\n[]\n```" in response :
                    print(">> NO NEW TOOLS INCLUDED")
                    break # no new tools were added

                tool_index = len(self.tool_results) # backup the tool results in case the last tool manager returns an empty list

                self.parse_tools(response, revise)
                print(">> TOOL RESULTS: \n", self.tool_results, "\n", sep="")

                for tool in self.tool_results[tool_index:]: # add new tool results to the chat history
                    for m in [self.assistant, self.tool_manager, self.debugger]:
                        # save the tool results in the message history of all models
                        m.add_message(role="tool", content=tool["result"], name=tool["name"])

                if r<n_revise: # revise the answer to implement the correct dependencies
                    self.tool_manager.add_message(role="user", content=tool_manager_revise)

                revise = True  # then it can call the tools that need the requirements

            # include the retrieved docs as context and feed it to the model
            self.tool_selection = False
            """

        #########################################################################################################
        #########################################################################################################
        #########################################################################################################

        if code is not None:  # initial code from baseline or codebase
            for m in [self.assistant]:#, self.debugger]:
                # save the starting code in the message history of the assistant and debugger models
                m.add_message(role="user", content="Use the following code as a starting point:\n"+code)

        for i in range(self.n_debug):
            if not baseline and code is not None: # test the code on compiler
                if verbose > 1: print("\n>> evaluating baseline code")
                # tool_result = dispatch_tool(self.tools, tool_name, tool_args)
                compiler_result = sandboxed_compiler(code)
                #perf_result = run_perf(gen_code_file)

                for m in [self.assistant, self.tool_manager]:#, self.debugger]:
                    # save the tool results in the message history of all models
                    m.add_message(role="tool", content=compiler_result, name="sandboxed_compiler")
                    #m.add_message(role="tool", content=perf_result, name="run_perf")

                if compiler_result[0] == 0: # check if the code compiled correctly
                    response = "There is nothing to improve."+"\nPrevious code:\n```"+code+"```"
                    break

                for m in [self.assistant, self.tool_manager]:#, self.debugger]:
                    # save the tool results in the message history of all models
                    m.add_message(role="system", content=compiler_prompt)

                # debug incorrect code
                #if verbose > 1: print("-------------------------------------- \n## debugger (i="+str(i)+", baseline="+str(baseline)+"): ")
                #response = self.debugger.call()
                #if verbose > 1: print("--------------------------------------")
                #self.assistant.add_message(role="debugger", content=debugger_revise+response)

            # apply chat templates and return an answer
            if verbose > 1: print("-------------------------------------- \n## assistant (i="+str(i)+", baseline="+str(baseline)+"): ")
            response = self.assistant.call()
            if verbose > 1: print("--------------------------------------")
            #self.debugger.add_message(role="assistant", content=response)

            if extract_code(response)=="":
                print(">> appending prev code")
                response = response+"\nPrevious code:\n```"+code+"```"
                break
            else:
                code = extract_code(response)
            if baseline: break

        if False:
            print("\n\n**************************************************************************** \n## tool_manager messages history:")
            pprint.pprint(self.tool_manager.get_messages())
            print("****************************************************************************\n")
        if False:
            print("\n**************************************************************************** \n## assistant messages history:")
            pprint.pprint(self.assistant.get_messages())
            print("****************************************************************************\n")

        # reset tool results, tool_manager and debugger
        self.tool_results = []
        self.tool_manager.reset_memory()
        #self.debugger.reset_memory()

        return response