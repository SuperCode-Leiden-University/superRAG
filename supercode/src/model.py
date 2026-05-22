import json, torch
import pprint
import threading
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, SinqConfig

#from awq import AutoAWQForCausalLM
"""
- pipeline is for direct inference, with AutoModelForCausalLM, AutoTokenizer you load the raw model
- BitsAndBytesConfig and awq are for quantization
- TextIteratorStreamer and threading are for printing the answer as it is being generated
"""

from src.tools.manage_tools import * # import all the tools
from src.configs.parse_config import model_id, raw_model, quant_type, max_new_tokens, temperature
from src.configs.system_prompts import *
from src.tools.tools import *
from src.tools.code_processing import *



class Model():
    # define variables and import the model
    def __init__(self):
        # "model" here refers only to the one used for processing text and generating an answer
        print("### model_id = "+model_id)

        ##### model's variables
        self.model_id = model_id # name of the model from Hugging Face
        self.raw_model = raw_model # True if the model is loaded directly, False if loaded through pipeline
        self.quant_type = quant_type # valid values: ("full", "bits", "GPTQ") --> check file formats!!!
        self.gen_args = None

        self.tool_results = []
        self.tool_selection = False # use the same model either to choose which tool to use or to generate an answer
        self.tools = get_tools() # tools for the agent
        self.schemas = [build_tool_schema(f) for f in self.tools.values()]
        self.details = [f._tool_metadata for f in self.tools.values()] # info about requirements

        if verbose > 1: print(">> defining system prompt")
        self.reset_memory() # keep only the system prompts

        ##### IMPORTING THE MODEL
        if self.raw_model:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

            if self.quant_type == "full":  # full precision
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map="auto",  # automatically places layers on GPU(s) if possible
                    #dtype="auto"
                )
            # quantization structures (temporarily disabled):
            """ 
            if self.quant_type == "sinq":
                print(">> quantizing")
                quant_config = SinqConfig(
                    nbits=4,
                    group_size=64,
                    tiling_mode="1D",
                    method="sinq",
                    modules_to_not_convert=["lm_head"]
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    quantization_config=quant_config,
                    device_map="auto",  # automatically places layers on GPU(s) if possible
                    dtype=torch.bfloat16
                )
                #model.save_pretrained("/path/to/save/"+self.model_id+"-sinq-4bit") # save quantized version
            
            if self.quant_type == "AWQ":  # AWQ quantization
                self.model = AutoAWQForCausalLM.from_quantized(
                    self.model_id,
                    device_map="auto",  # automatically places layers on GPU(s) if possible
                    #dtype="auto"
                )
            
            if self.quant_type == "GPTQ":  # GPTQ quantization
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
                self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map="auto",  # automatically places layers on GPU(s) if possible
                    dtype="auto", #torch.float16,  # recommended for GPTQ
                    trust_remote_code=True,
                    quantization_config=None # fix incompatibility between Transformers and Optimum GPTQ
                    # consequences: no CUDA kernels optimization, lose correct dequantization behavior,
                    # lose numerical stability guarantees, may have unexpected runtime errors
                )

            if self.quant_type == "bits":  # for 4-8 bits quantization
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,  # or load_in_8bit=True
                    bnb_4bit_compute_dtype="float16",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map="auto",  # automatically places layers on GPU(s) if possible
                    quantization_config=quant_config
                )
            
            #"""
        else:
            self.model = pipeline("text-generation", model=self.model_id)

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
        # recipient can be: assistant, tool_manager or both

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

    # ----------------------------------------------------------------------------------------------
    # apply chat templates, generate and return an answer
    def chat_template(self):
        """ Note: the chat history (messages) is structured like this:
        chat_history = [ {'generated_text': [
                            {'role': 'user',       'content': '...'},
                            {'role': 'assistant',  'content': '...'}
                        ] } ]
        """

        if self.raw_model:
            if self.tool_selection :
                inputs = self.tokenizer.apply_chat_template(
                    self.tool_messages,
                    tools=self.schemas,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.model.device)
            else:
                inputs = self.tokenizer.apply_chat_template(
                    self.messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.model.device)

            def generate():
                # check if kwargs are passed in model.call(), otherwise use the values in the config file
                self.model.generate(
                    **inputs,
                    max_new_tokens=self.gen_args.get("max_new_tokens",max_new_tokens),
                    temperature=self.gen_args.get("temperature",temperature),
                    streamer=self.streamer
                )
            #response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True) # wait for the whole answer before printing

            # this is for printing the tokens as they are generated
            thread = threading.Thread(target=generate)
            thread.start()
            response = ""
            for token in self.streamer:
                print(token, end="", flush=True)
                response += token
            print()

        else: # using pipeline
            if self.tool_selection :
                outputs = self.model(self.messages)
            else:
                outputs = self.model(self.tool_messages)
            response = outputs[0]['generated_text'][-1]["content"]

        if self.tool_selection : # save the response in the message history
            self.message_format(recipient="tool_manager", role="tool_manager", content=response)
        else:
            self.message_format(recipient="assistant", role="assistant", content=response)

        return response

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
        self.gen_args = kwargs # model's settings like temperature and max tokens (default vals in config)
        if reset_memory: self.reset_memory() # forget previous answers and keep only the system prompts, useful for benchmarks

        self.message_format(recipient="both", role="user", content=user_prompt) # save the user_prompt in the message history

        if not baseline:
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

            #########################################################################################################
            #########################################################################################################
            #########################################################################################################
            # TEMPORARY PATCH
            #""" write a python function that returns all even numbers from 0 to n, then assert return_even(5)==[0,2,4] and print("no errors") at the end
            gen_code_file = "gen_code_temp.py"
            gen_code_path = gen_code_dir + "/gen_code/"+gen_code_file
            """
            # apply chat templates and return an answer
            if verbose > 1: print("-------------------------------------- \n## (temp) AI: ")
            response = self.chat_template()
            if verbose > 1: print("--------------------------------------")

            code = extract_code(response)
            """
            print(">> extracting the code:\n", code)
            with open(gen_code_path, "w") as f:
                print(">> saving the code at", gen_code_path)
                f.write(code)
                f.close()
            self.message_format(recipient="both", role="user", content="Use the following code as a baseline"+code)
    
            # tool_result = dispatch_tool(self.tools, tool_name, tool_args)
            #megalinter_result = run_megalinter("python)         ; self.message_format(recipient="both", role="tool", content=megalinter_result, name="run_megalinter")
            compiler_result = sandboxed_compiler(gen_code_path) ; self.message_format(recipient="both", role="tool", content=compiler_result, name="sandboxed_compiler")
            #perf_result = run_perf(gen_code_file)               ; self.message_format(recipient="both", role="tool", content=perf_result, name="run_perf")

            #########################################################################################################
            #########################################################################################################
            #########################################################################################################

        # apply chat templates and return an answer
        if verbose > 1: print("-------------------------------------- \n## AI (baseline="+str(baseline)+"): ")
        response = self.chat_template()
        if verbose > 1: print("--------------------------------------")

        if False and verbose>3 :
            print("\n\n**************************************************************************** \n## tool_manager messages history:")
            pprint.pprint(self.tool_messages)
            print("****************************************************************************\n")
            print("\n**************************************************************************** \n## assistant messages history:")
            pprint.pprint(self.messages)
            print("****************************************************************************\n")
            print("****************************************************************************")

        self.tool_results = [] # reset tool results
        self.tool_messages = self.tool_messages[:1] # clear all, except the system prompt

        return response