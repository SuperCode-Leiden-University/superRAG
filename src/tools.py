import re
import inspect
import sys
from langchain.tools import tool, BaseTool


# Define Tools
@tool
def calculate_expression(expression: str) -> str:
    """Calculator: Evaluate a mathematical expression"""
    safe_expr = re.sub(r'[^0-9+\-*/(). ]', '', expression)
    if safe_expr.strip() == "":
        return "I couldn't compute that."
    try:
        result = eval(safe_expr)
        print("--- Tool: calculator")
        return f"The result is: {result}"
    except:
        return "I couldn't compute that."



##################################################################################################################
# return all the tools as a list of functions
def get_tools():
    module = sys.modules[__name__] # only check this module
    tools = [
        obj # this is the tool function
        for name, obj in inspect.getmembers(module) # check all objects defined in this module
        if isinstance(obj, BaseTool) # filter only those with '@tool' decorator
    ]
    print("\n--- tools available:", [t.name for t in tools],"\n") # I only want to print the name, not all the info
    return tools


