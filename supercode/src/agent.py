import json, torch
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
        ##### extract model's name and parameters

        # reasoning model for planning
        self.planning = False
        self.plan_model_id = plan_model_id

        self.tool_results = []
        self.tool_selection = False # use the same model either to choose which tool to use or to generate an answer
        self.tools = get_tools() # tools for the agent
        self.schemas = [build_tool_schema(f) for f in self.tools.values()]
        self.details = [f._tool_metadata for f in self.tools.values()] # info about requirements

        if verbose > 1: print(">> defining system prompt")
        self.reset_memory() # keep only the system prompts

        ##### IMPORTING THE MODELS
        self.assistant = Model(model_args, system_prompt=chat_assistant_prompt, prequery_prompt=chat_assistant_prequery, tools_schemas=None)

        tool_manager_prompt = f"You are a tool manager, tools at your disposal are described in:\n\n{self.schemas}\n"+tool_manager_prompt
        self.tool_manager = Model(model_args, system_prompt=tool_manager_prompt, prequery_prompt=tool_manager_prequery, tools_schemas=self.schemas)

        self.planner = Model(plan_model_args, system_prompt=planner_prompt, prequery_prompt=planner_prequery, tools_schemas=None)

    def reset_memory(self):
        self.messages = [{
            "role": "system",
            "content": chat_assistant_prompt
        }] # history of the chat
        self.tool_messages = [{
            "role": "system",
            "content": f"You are a tool manager, tools at your disposal are described in:\n\n{self.schemas}\n" + tool_manager_prompt
        }] # messages for tool selection

    # ----------------------------------------------------------------------------------------------
    # format the messages to be able to include tools and RAG
    def message_format(self, recipient, role, content, **kwargs):
        # recipient can be: assistant, tool_manager, planner or all

        # add message for the chat assistant
        if recipient!="tool_manager": # messages for the assistant
            if role=="user" : content=chat_assistant_prequery+content
            self.messages.append({
                "role": role,
                "content": content,
                **kwargs # add optional extra arguments (ex: tool's name)
            })

        # add message for the tool manager
        if recipient!="assistant": # messages for the tool_manager
            if role=="user" : content=tool_manager_prequery+content
            self.tool_messages.append({
                "role": role,
                "content": content,
                **kwargs # add optional extra arguments (ex: tool's name)
            })

        # add message for the tool manager
        if recipient!="planner": # messages for the tool_manager
            #if role=="user" : content=planner_prequery+content
            self.plan_messages.append({
                "role": role,
                "content": content,
                **kwargs # add optional extra arguments (ex: tool's name)
            })

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

        self.message_format(recipient="both", role="user", content=user_prompt) # save the user_prompt in the message history


        """ 
            n_revise = -1#3
            revise = False  # most often the model calls the tools for the requirements by the second try
            self.tool_selection = True

            for r in range(n_revise+1): # refinement loop, by the 3rd iteration it should have the correct tools selected
                # apply chat templates and return an answer
                if verbose > 1: print(f"-------------------------------------- \n## tool manager {r+1}: ")
                response = self.chat_template()
                if verbose>1 : print("--------------------------------------")

                if "```json\n[]\n```" in response :
                    print(">> NO NEW TOOLS INCLUDED")
                    break # no new tools were added

                tool_index = len(self.tool_results) # backup the tool results in case the last tool manager returns an empty list

                self.parse_tools(response, revise)
                print(">> TOOL RESULTS: \n", self.tool_results, "\n", sep="")

                for tool in self.tool_results[tool_index:]: # add new tool results to the chat history
                    self.message_format(recipient="both", role="tool", content=tool["result"], name=tool["name"])

                if r<n_revise: # revise the answer to implement the correct dependencies
                    self.message_format(recipient="tool_manager", role="user", content=tool_manager_revise)

                revise = True  # then it can call the tools that need the requirements

            # include the retrieved docs as context and feed it to the model
            self.tool_selection = False
            """

        #########################################################################################################
        #########################################################################################################
        #########################################################################################################

        if code is not None:  # initial code from baseline or codebase
            self.message_format(recipient="both", role="user", content="Use the following code as a starting point:\n"+code)

        for i in range(3):
            """
            # apply chat templates and return an answer
            if verbose > 1: print("-------------------------------------- \n## (temp) AI: ")
            response = self.chat_template()
            if verbose > 1: print("--------------------------------------")
    
            code = extract_code(response)
            """
            if not baseline and code is not None: # test the code on compiler
                if verbose > 1: print("\n>> evaluating baseline code")
                # tool_result = dispatch_tool(self.tools, tool_name, tool_args)
                #megalinter_result = run_megalinter("python) ; self.message_format(recipient="both", role="tool", content=megalinter_result, name="run_megalinter")
                compiler_result = sandboxed_compiler(code) ; self.message_format(recipient="both", role="tool", content=compiler_result, name="sandboxed_compiler")
                #perf_result = run_perf(gen_code_file) ; self.message_format(recipient="both", role="tool", content=perf_result, name="run_perf")

                if compiler_result[0] == 0: # check if the code compiled correctly
                    response = "There is nothing to improve."+"\nPrevious code:\n```"+code+"```"
                    break
                self.message_format(recipient="both", role="user", content=compiler_prompt)

            # apply chat templates and return an answer
            if verbose > 1: print("-------------------------------------- \n## AI (i="+str(i)+", baseline="+str(baseline)+"): ")
            response = self.chat_template()
            if verbose > 1: print("--------------------------------------")

            if extract_code(response)=="":
                print(">> appending prev code")
                response = response+"\nPrevious code:\n```"+code+"```"
                break
            else:
                code = extract_code(response)
            if baseline: break

        if False:
            print("\n\n**************************************************************************** \n## tool_manager messages history:")
            pprint.pprint(self.tool_messages)
            print("****************************************************************************\n")
        if True:
            print("\n**************************************************************************** \n## assistant messages history:")
            pprint.pprint(self.messages)
            print("****************************************************************************\n")

        self.tool_results = [] # reset tool results
        self.tool_messages = self.tool_messages[:1] # clear all, except the system prompt

        return response