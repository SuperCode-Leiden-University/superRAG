from pprint import pp as pprint

# importing my functions from other files
from supercode.src.model_loader.model_loader import Model
from supercode.src.tools.manage_tools import * # import all the tools
from supercode.src.tools.tools import *
from supercode.src.tools.code_processing import *

# importing variables from the config file
from supercode.src.configs.parse_config import *
from supercode.src.configs.system_prompts import *

"""
Standard/officially-used roles:
- system: general instructions
- user: the user
- assistent: the model
- tool: feedback from tools
- thinking: thinking mode/reasoning schemas (ex: CoT)
"""

class Agent():
    # ----------------------------------------------------------------------------------------------
    # define variables and import the model, vars not defined here are in the config file!
    def __init__(self, coder_id):
        # tools related parameters
        self.tool_results = []
        self.tools = get_tools() # tools for the agent

        coder_model_args["model_id"] = coder_id # to change the model from terminal without editing the config file
        #print(">> coder args\n", coder_model_args)

        ### MODELS
        # coding assistant (expert model)
        self.assistant = Model(
            coder_model_args,
            system_prompt=coder_prompt,
            prequery_prompt=coder_prequery,
        )
        # tool manager is a reasoning model for selecting tools
        if tools_iter>0:
            self.schemas = [build_tool_schema(f) for f in self.tools.values()]
            self.details = [f._tool_metadata for f in self.tools.values()] # info about requirements
            #pprint(self.schemas)
            tool_manager_prompt = f"{manager_prompt_1}{self.schemas}{manager_prompt_2}"
            self.tool_manager = Model(
                think_model_args,
                system_prompt=tool_manager_prompt,
                prequery_prompt=tool_manager_prequery,
            )

        # list of active models
        self.models_list = [self.assistant]
        if tools_iter>0: self.models_list.append(self.tool_manager)

        # add system prompts
        self.reset_memory()

    # ----------------------------------------------------------------------------------------------
    def reset_memory(self):
        for m in self.models_list: m.reset_memory()

    # ----------------------------------------------------------------------------------------------
    def print_chat_history(self):
        for m in self.models_list:
            print("\n****************************************************************************\n"
                  "## messages history (excluding system prompt):")
            pprint(m.get_messages()[1:]) # exclude the system prompt
            print("****************************************************************************\n")

    # ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------
    def call(self,
             user_prompt,
             baseline_code=None, code_language=None, baseline_output=None,
             reset_memory=False, baseline=False,
             **kwargs
        ):
        """
        WORKFLOW:
            1) the model checks if it needs to call a tool or retrieve docs
            2) the tool is called and/or docs are retrieved
            3) the model reads the results and generate an answer
        if baseline is True, then the model doesn't use any external info
        """

        def get_response(role, i):
            if role=="assistant" or verbose>-1:
                print(f"\n-------------------------------------- \n## {role} (i={i}, baseline={baseline}): ")

            # call the model and get a list of completions
            if role == "assistant":
                samples = self.assistant.call()
            elif role == "tool_manager":
                samples = self.tool_manager.call(tool_schemas=self.schemas)
            else: raise ValueError(f"ERROR: get_response() got an invalid role: {role}")
            # for i, sample in enumerate(samples):
            #     print(f"\ni={i}-----------------------------------\n{sample}")
            # print("--------------------------------------\n")

            # select the best code
            index, code_results = select_response(samples)
            # note: code_results = list({"language", "code", "compiler_output"})
            completions_results = []
            for i, code_result in enumerate(code_results):
                code_result["completion"] = samples[i] # add the completion to the dict
                completions_results.append(code_result)

            self.assistant.add_message(role="assistant", content=samples[index])

            if role=="assistant" or verbose>-1:
                print(#samples[index],
                      "!!! completion temp removed"
                      "\n--------------------------------------")

            return index, completions_results # note: code_results = list({"language", "code", "compiler_output", "completion"})

        # ----------------------------------------------------------------------------------------------
        # always reset tool results and tool_manager
        self.tool_results = []
        if tools_iter>0: self.tool_manager.reset_memory()

        # optional: forget previous answers and keep only the system prompts, useful for benchmarks
        if reset_memory: self.reset_memory()

        if not baseline and baseline_output[0]==0: # add optimization instructions if the code is correct
            print(">> baseline code is correct, attempting optimization")
            user_prompt = f"{user_prompt}\n{optim_prompt}"

        # save the user_prompt in the message history of all active models
        for m in self.models_list:
            m.add_message(role="user", content=user_prompt)

            # save the baseline code as context
            if baseline_code is not None:# and m!=self.tool_manager:
                m.add_message(
                    role="code",
                    content=f"baseline code:\n```{code_language}\n{baseline_code}\n```",
                )

        for i in range(model_iter): # sequential iterations on the same code
            # ----------------------------------------------------------------------------------------------
            # use standalone model (no iteration and no tools allowed, baseline to measure effectiveness)
            if baseline:
                # apply chat templates and return an answer
                # response, responses
                index, completions_results = get_response("assistant", i)
                # note: code_results = list({"language", "code", "compiler_output", "completion"})
                break # there is no need to iterate multiple times

            # ----------------------------------------------------------------------------------------------
            # use model with tools (compiler, profiler, info from database, websearch, etc...)
            else:
                # ----------------------------------------------------------------------------------------------
                # let the thinking model choose a tool
                if tools_iter>0:
                    if verbose>0: print(f">> selecting tools with tool manager")
                    #revise = False # the first iteration skip tools with requirements

                    # refinement loop, useful when tools have requirements and need info from other tools
                    for r in range(tools_iter):
                        tool_index = len(self.tool_results) # to avoid duplications

                        # choose the tools
                        tool_response = self.tool_manager.call(tool_schemas=self.schemas)[0]

                        # shortcut: skip the rest if there are no tools to be called
                        if "```json\n[]\n```" in tool_response:
                            print(">> WARNING: empty jsonl for tools")
                            break

                        # parse the tool manager answer, find the tools, call them and report the results
                        self.tool_results = parse_tools(tool_response, self.tools, self.schemas, self.tool_results)#, revise)
                        if verbose>1: print(f">> TOOL RESULTS: \n{self.tool_results}\n")

                        # add the tool results to the chat history of all models
                        for tool in self.tool_results[tool_index:]: # only include new results
                            for m in self.models_list: # all models need the tool results
                                m.add_message(role="tool", content=tool)

                        # revise the answer to implement the correct dependencies
                        #if r<tools_iter-1: self.tool_manager.add_message(role="user", content=tool_manager_revise)
                        #revise = True

                # ----------------------------------------------------------------------------------------------
                # predetermined use of tools to analyze code (if given)
                elif baseline_code is not None:
                    if verbose>0: print(">> using predetermined list of tools")
                    # compiler_output = sandboxed_compiler(baseline_code, code_language)
                    # perf_result = run_perf(gen_code_file)

                    # in case I have some baseline code with no compiler output
                    if baseline_output==None: baseline_output = sandboxed_compiler(baseline_code, code_language)

                    for m in self.models_list:
                        # save the tool results in the message history of all models
                        m.add_message(
                            role="tool",
                            content= {
                                "name": "sandboxed_compiler",
                                "input": {"code": baseline_code, "language": code_language},
                                "result": str(baseline_output)
                            }
                        )
                        # m.add_message(
                        #     role="tool",
                        #     content= {
                        #         "name": "run_perf",
                        #         "input": gen_code_file,
                        #         "result": str(perf_result)
                        #     }
                        # )

                        # prompt to improve the code if the compiler returns an error
                        #if compiler_output[0] != 0: m.add_message(role="user", content=compiler_prompt)

                    # if compiler_output[0] == 0: # check if the code compiled correctly
                    #     responses = []
                    #     response = f"There is nothing to improve.\nPrevious code:\n```{code_language}\n{baseline_code}\n```"
                    #     print(
                    #         "\n-------------------------------------- \n"
                    #         f"## assistant (i={i}, baseline={baseline}): \n{response}"
                    #         "\n--------------------------------------"
                    #     )
                    #     break

                # ----------------------------------------------------------------------------------------------
                else: print(f">> WARNING: tools not enabled because tools_iter={tools_iter} and baseline_code={baseline_code}")

                # ----------------------------------------------------------------------------------------------
                # ask the coder assistant to improve the code
                # response, responses
                index, completions_results = get_response("assistant", i)
                # note: code_results = list({"language", "code", "compiler_output", "completion"})

                # failsafe in case the model doesn't return any code
                # if completions_results["code"] == None:
                #     if baseline_code is not None:
                #         print(">> appending prev code")
                #         response = response+f"\nPrevious code:\n```{code_language}\n{baseline_code}\n```"
                #         break
                #     else:
                #         print("WARNING: failed to extract code and no previous code to fall back to")
                #         response = response+"\nNo code:\n```raise Exception('NO CODE')```"
                #         continue
                # else:
                #     code_language, baseline_code = extract_code(response)

        return index, completions_results

