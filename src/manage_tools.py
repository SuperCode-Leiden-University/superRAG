import inspect
import sys
import pkgutil
import importlib
from pathlib import Path
from typing import get_type_hints

from src.config import verbose


##################################################################################################################
# tool decorator
def tool(func):
    func.__is_tool__ = True
    return func


##################################################################################################################
# functions for handling the tools

# ----------------------------------------------------------------------------------------------
# return all the tools as a list of functions
def get_tools():
    if verbose>0 : print(">> fetching available tools")
    #module = sys.modules[__name__] # only check this file
    module = importlib.import_module("src.tools") # only check another file
    """ # for checking all files inside a directory named "tools"
    package = "tools"
    package_path = Path(__file__).parent
    for module_info in pkgutil.iter_modules([str(package_path)]): module = importlib.import_module(
        f"{package}.{module_info.name}")
    #"""
    tools = {
        name : obj # obj is the tool function
        for name, obj in inspect.getmembers(module, inspect.isfunction) # check all objects defined in this module
        if getattr(obj, "__is_tool__", False)
    }
    if verbose>0 : print(">> tools available: ", tools.keys())
    return tools

# ----------------------------------------------------------------------------------------------
# build the json file to call the tool
def build_tool_schema(func):
    if verbose>0 : print(">> building tool schema for:", func.__name__)
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
            }
        }
    }
    if verbose>2 : print(scheme)
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

