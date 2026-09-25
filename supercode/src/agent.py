from supercode.src.model_loader.model_loader import Model
from supercode.src.tools.manage_tools import * # import all the tools
from supercode.src.configs.system_prompts import *
from supercode.src.tools.tools import *
from supercode.src.tools.code_processing import *

"""
Standard/officially-used roles:
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
        #self.tool_manager_role = "reasoning" # thinking model for selecting tools

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
        print("\n****************************************************************************\n"
              "## assistant messages history (excluding system prompt):")
        pprint.pprint(self.assistant.get_messages()[1:]) # exclude the system prompt
        print("****************************************************************************\n")

        if tools_iter>0:
            print("\n\n****************************************************************************\n"
                  "## tool_manager messages history (excluding system prompt):")
            pprint.pprint(self.tool_manager.get_messages()[1:]) # exclude the system prompt
            print("****************************************************************************\n")

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

            # save the baseline code as context
            # if code is not None:# and m!=self.tool_manager:
            #     m.add_message(
            #         role=self.tool_role,
            #         content="baseline code:\n```\n"+code+"\n```",
            #         #TODO: check if I can specify the language!!!
            #     )

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
                    #revise = False # the first iteration skip tools with requirements

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
                        self.tool_results = parse_tools(response, self.tools, self.schemas, self.tool_results)#, revise)
                        if verbose>1: print(">> TOOL RESULTS: \n", self.tool_results, "\n", sep="")

                        # add the tool results to the chat history of all models
                        for tool in self.tool_results[tool_index:]: # only include new results
                            for m in self.models_list: # all models need the tool results
                                m.add_message(role=self.tool_role, content=tool)

                        # revise the answer to implement the correct dependencies
                        #if r<tools_iter-1: self.tool_manager.add_message(role=self.user_role, content=tool_manager_revise)
                        #revise = True

                # ----------------------------------------------------------------------------------------------
                # predetermined use of tools to analyze code (if given)
                elif code is not None:
                    print(">> predetermined tools")
                    compiler_result = sandboxed_compiler(code)
                    # perf_result = run_perf(gen_code_file)

                    for m in self.models_list:
                        # save the tool results in the message history of all models
                        m.add_message(
                            role=self.tool_role,
                            content= {
                                "name": "sandboxed_compiler",
                                "input": code,
                                "result": str(compiler_result)
                            }
                        )
                        # m.add_message(
                        #     role=self.tool_role,
                        #     content= {
                        #         "name": "run_perf",
                        #         "input": gen_code_file,
                        #         "result": str(perf_result)
                        #     }
                        # )

                        # prompt to improve the code if the compiler returns an error
                        #if compiler_result[0] != 0: m.add_message(role=self.user_role, content=compiler_prompt)

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
                # if extract_code(response)=="":
                #     if code is not None:
                #         print(">> appending prev code")
                #         response = response+"\nPrevious code:\n```"+code+"```"
                #         break
                #     else:
                #         print("WARNING: failed to extract code and no previous code to fall back to")
                #         response = response+"\nNo code:\n```raise Exception('NO CODE')```"
                #         continue # failed to extract code and no previous code to fall back to
                # else:
                #     code = extract_code(response)

        return response

