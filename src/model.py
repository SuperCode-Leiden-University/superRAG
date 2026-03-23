import json
import threading
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
#from awq import AutoAWQForCausalLM
"""
- pipeline is for direct inference, with AutoModelForCausalLM, AutoTokenizer you load the raw model
- BitsAndBytesConfig and awq are for quantization
- TextIteratorStreamer and threading are for printing the answer as it is being generated
"""

from src.tools.manage_tools import * # import all the tools
from src.configs.parse_config import verbose, max_new_tokens
from src.configs.system_prompts import chat_assistant_prompt, tool_manager_prompt

class Model():
    # define variables and import the model
    def __init__(self, model_id, raw_model=True, quant_type="full"):
        # "model" here refers only to the one used for processing text and generating an answer

        ##### model's variables
        self.model_id = model_id # name of the model from Hugging Face
        self.raw_model = raw_model # True if the model is loaded directly, False if loaded through pipeline
        self.quant_type = quant_type # valid values: ("full", "bits", "GPTQ") --> check file formats!!!
        """
        NOTE: the model can be loaded as quantized only if someone published the quantized version (see files):
            --> full precision:   model.safetensors                 (direct load)
            --> full precision:   pytorch_model.bin.index.json      (direct load, NOT with Transformers!)
            --> 4bit quantized:   model-4bit.safetensors            (needs bitsandbytes)
            --> GPTQ quantized:   gptq_model-4bit-128g.safetensors  (direct load)
            --> GGUF quantized:   model.gguf                        (direct load, NOT with Transformers!)
        """

        self.tool_results = []
        self.tool_selection = False # use the same model either to choose which tool to use or to generate an answer
        self.tools = get_tools() # tools for the agent
        self.schemas = [build_tool_schema(f) for f in self.tools.values()]

        self.messages = [] # history of the chat
        self.tool_messages = [] # messages for tool selection

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

    # ----------------------------------------------------------------------------------------------
    # format the messages to be able to include tools and RAG
    def message_format(self, user_prompt):
        # include the system prompts at the beginning of the chat
        if not self.messages:
            if verbose>1 : print(">> defining system prompt")
            # this is the chat history
            self.messages.append({
                "role": "system",
                "content": chat_assistant_prompt
            })

            # this is for deciding if a tool is needed (and which one)
            self.tool_messages.append({
                "role": "system",
                "content": f"You are a tool manager, tools at your disposal are described in:\n\n{self.schemas}\n"+tool_manager_prompt
            })

        # decide if the model needs a tool or to retrieve info from the DB
        if self.tool_selection:
            if verbose>0 : print(">> calling the tool manager")
            self.tool_messages.append({
                "role": "user",
                "content": f"Based on the following question decide if there is a tool you can use.\n\n"
                           f"### Question\n{user_prompt}\n"
            })
        else:
            if verbose>0 : print(">> elaborating the results from tools")

            self.messages.append({
                "role": "user",
                "content": f"Use the following information to answer the question in natural language.\n\n"
                           f"### Question\n{user_prompt}"
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
                self.model.generate(**inputs, max_new_tokens=max_new_tokens, streamer=self.streamer)
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

        if self.tool_selection :
            self.tool_messages.append({"role": "tool_manager", "content": response })
        else:
            self.messages.append({"role": "assistant", "content": response })

        return response

    def parse_tools(self, response, revise=False):
        if '"name": "' in response:
            if verbose > 0: print(">> parsing response for tools")
            # Note: the LLM is not always consistent, sometimes there are multiple separate json objs instead of a list,
            # or the json list is repeated multiple times, but the json obj is always written as:
            # { "name": "tool_name", "arguments": { "arg_name": "value"} }

            tool_end = 0
            end = response.rfind("}")+1 # find last occurrence (the +1 is just bc I use it in tool_end to include '}' in the string)
            tool_request_list = []

            while tool_end < end:
                # find where the tool begins
                tool_check = response[tool_end:].find('"name": "')
                if tool_check == -1: break # this cover the case in which I have no tool left but there are '}'
                tool_temp = tool_end + tool_check # this is more robust signature than '{'
                tool_begin = tool_end + response[tool_end:tool_temp].rfind("{") # find '{' before the signature
                # find where the tool ends
                tool_temp  = tool_begin + response[tool_begin:].find("}")+1
                tool_end   = tool_temp  + response[tool_temp: ].find("}")+1

                tool_request = json.loads( response[tool_begin:tool_end].strip("  ") ) # remove extra spaces and load
                print(">> JSON OBJ (PARTIAL): \n", tool_request, sep="")

                # load the json obj as a dictionary
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
    def call(self, user_prompt):
        """
        WORKFLOW:
            1) the model checks if it needs to call a tool or retrieve docs
            2) the tool is called and/or docs are retrieved
            3) the model integrates the results with its answer
        """
        # ----------------------------------------------------------------------------------------------
        # decide if it needs to retrieve context and/or to use tools
        self.tool_selection = True
        self.message_format(user_prompt)

        n_revise = 3
        revise = False  # most often the model calls the tools for the requirements by the second try

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
            #print(">> TOOL RESULTS: \n", self.tool_results, "\n", sep="")

            for tool in self.tool_results[tool_index:]: # add new tool results to the chat history
                tool_message = [{
                    "role": "tool",
                    "name": tool["name"],
                    "content": tool["result"]
                }]
                self.messages.extend(tool_message)
                self.tool_messages.extend(tool_message)

            if r<n_revise: # revise the answer to implement the correct dependencies
                self.tool_messages.append({
                    "role": "user",
                    "content": """
                    Use the tools results to improve your previous answer and check the schema to make your answer compliant 
                    with the tools requirements. Treat tool results as correct and final.
                    """
                })
            revise = True  # then it can call the tools that need the requirements

        # include the retrieved docs as context and feed it to the model
        self.tool_selection = False
        self.message_format(user_prompt)

        # apply chat templates and return an answer
        if verbose > 1: print("-------------------------------------- \n## AI: ")
        response = self.chat_template()
        if verbose>1 : print("--------------------------------------")

        self.tool_results = [] # reset tool results

        #if verbose>3 : print("\n\n**************************************************************************** \n## tool_messages history: \n", self.tool_messages, "\n****************************************************************************\n", sep="")
        #if verbose>3 : print("**************************************************************************** \n## messages history: \n", self.messages, "\n****************************************************************************\n\n", sep="")


        return response