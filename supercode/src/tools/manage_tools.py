import functools
import inspect
import importlib
import pprint
import json
from typing import get_type_hints

from src.configs.parse_config import *
import src.tools.tools  # do NOT remove this because the decorators have to run!
from src.tools.tools import get_TOOLS

##################################################################################################################
# functions for handling the tools

# ----------------------------------------------------------------------------------------------
# return all the tools as a list of functions
def get_tools():
    if verbose>0 : print(">> fetching available tools")
    __TOOLS = get_TOOLS()
    tools = {
        f.__name__ : f # is the tool function
        for f in __TOOLS if getattr(f, "__is_tool__", False)
    }
    if verbose>0 : print(">> tools available: ", tools.keys())
    return tools

# ----------------------------------------------------------------------------------------------
# build the json file to call the tool
def build_tool_schema(func):
    #if verbose>0 : print(">> building tool schema for:", func.__name__)
    hints = get_type_hints(func)
    props = {name: {"type": python_to_json_type(tp)} for name, tp in hints.items()}

    scheme = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": props,
                "required": list(props.keys())
            },
            "x_metadata" : func._tool_metadata,
        }
    }
    #if verbose>2 : pprint.pprint(scheme) ; print("")
    return scheme

# ----------------------------------------------------------------------------------------------
# convert python type to json type (needed for build_tool_schema)
def python_to_json_type(tp):
    if tp == int: return "integer"
    if tp == float: return "number"
    if tp == str: return "string"
    if tp == bool: return "boolean"
    return "string"

# ----------------------------------------------------------------------------------------------
# actually call the tool
def dispatch_tool(tools, name, args):
    if verbose>0 : print(">> dispatching tool: ", name)
    tool = tools[name]
    return tool(**args)

# ----------------------------------------------------------------------------------------------
def parse_tools(response, tools, schemas, tool_results, revise=False):
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
                if tool_name in tools.keys():
                    if verbose > 0: print(">> found tool:", tool_name)

                    # skip tools that require dependencies (req_flag=True) the first time (revise=False)
                    # this way the first time it will compute only results for tools with no dependencies
                    for s in schemas:
                        fn = s.get('function', {}) # the second value is returned if the first cannot be found
                        if fn.get('name') == tool_name:
                            req_flag = fn.get('x_metadata', {}).get('req_flag', False)
                    skip = req_flag and not revise
                    if skip :
                        tool_result = "This tool has requirements."
                        if verbose > 2: print(">> skipping tool:", tool_name)
                    else:
                        tool_result = dispatch_tool(tools, tool_name, tool_request["arguments"])
                        #if verbose > 2: print(">> tool result:", tool_result)

                    # save the results to pass them to the model
                    tool_result = {
                        "name": tool_name,
                        "result": tool_result
                    }
                    if tool_result not in tool_results:
                        tool_results.append(tool_result)
            else:
                if verbose > 2: print(">> this tool was already called:", tool_name)
    else:
        if verbose > 0: print(">> no tool was found")

    # self.tool_messages.pop() # remove the last item
    return tool_results