import re
import ast
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from src.verbose import verbose
from .manage_tools import tool




##################################################################################################################
# Define Tools

# TEST:
# calculate the following mathematical expression: (8276021*5410−5140906382)/65918910   # SOL = 601.229104486
# calculate the following mathematical expression: 516038^(5/2)                         # SOL = 1.912952597×10¹⁴
# draw the graph of 2*x*x+1 for x in [-5,10]

# ----------------------------------------------------------------------------------------------
@tool
def calculator(expr: str): # arguments should have a defined type
    if verbose()>0 : print(">> using the calculator")
    calculator.__doc__ = "Calculate the result of a mathematical expression."

    # remove everything that isn't a math symbol (to avoid evaluating malicious code)
    safe_expr = re.sub(r'[^0-9+\-*/().[] ]', '', expr)
    if verbose()>1 : print(">> safe_expr =", safe_expr)
    # Insert implicit multiplication: 5(4+3) --> 5*(4+3)
    safe_expr = re.sub(r'(\d)\s*\(', r'\1*(', safe_expr)
    if verbose()>1 : print(">> safe_expr =", safe_expr)
    if safe_expr.strip() == "":
        print("\nAn error occurred: empty expression.")
        return "Couldn't compute the result."

    try:
        result = eval(safe_expr)
        if verbose()>1 : print(">> calc result =", result)
        return f"The result is: {result}"
    except Exception as e:
        print(f"\nAn error occurred:\n{e}\n")
        return "Couldn't compute the result."

# ----------------------------------------------------------------------------------------------
@tool
def draw_graph(expr: str, x_range: list[float]):
    if verbose()>0 : print(">> using the draw_graph")
    draw_graph.__doc__ = "Draw the graph of a function given the expression and x range."
    try:
        if verbose()>1 : print(">> converting the function")
        # remove everything that isn't a math symbol (to avoid evaluating malicious code)
        safe_expr = re.sub(r'[^0-9+\-*/(). [],x]', '', expr)
        func = lambda x: eval(safe_expr)

        if verbose()>1 : print(">> converting the range")
        if isinstance(x_range, str): # convert the range back to a list if it's passed as a string by mistake
            x_range = [float(x) for x in x_range.strip("[]").split(",")]
            if verbose()>1 : print(">> x_range =", x_range)

        x_points = np.linspace(x_range[0], x_range[1], 100)
        y_points = np.array([func(x) for x in x_points])
        plt.plot(x_points, y_points)
        plt.savefig("test-tools-results/graph.png")
        plt.close()
        return "the graph was saved as graph.png in the directory 'test-tools-results/'"
    except Exception as e:
        print(f"\nAn error occurred:\n{e}\n")
        return "I couldn't draw the graph."


# ----------------------------------------------------------------------------------------------
#@tool
def search_database(db, query, n_retriv):
    if verbose()>0 : print(">> searching the database")
    start = datetime.now()
    # find the relevant docs
    if n_retriv>0:
        retriv_docs = db.similarity_search(query, k=n_retriv) # find the k most relevant documents to the query
        if verbose()>1 : print(">> n_retriv_docs =", len(retriv_docs), "=", n_retriv)
        #if verbose()>2 : print("most relevant doc\n", retriv_docs[0])
    else:
        retriv_docs = None

    end = datetime.now()
    if verbose()>0 : print(">> Time to Retrieval =", end-start)
    return retriv_docs
