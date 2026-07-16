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

from src.configs.parse_config import *
from src.configs.system_prompts import *
#from src.tools.tools import *
#from src.tools.manage_tools import * # import all the tools
from src.tools.code_processing import *



class Model():
    # define variables and import the model
    def __init__(self, model_args, system_prompt=None, prequery_prompt=None, tool_schemas=None):
        ##### extract model's name and parameters
        self.model_id   = model_args["model_id"]   # name of the model from Hugging Face
        self.raw_model  = model_args["raw_model"]  # True if the model is loaded directly, False if loaded through pipeline
        self.quant_type = model_args["quant_type"] # valid values: ("full", "bits", "GPTQ") --> check file formats
        self.gen_args   = model_args["gen_args"]   # other settings, such as temperature and max tokens (default vals in config)
        print("### model_id = "+self.model_id)

        # defining prompts and chat history (short term memory)
        if verbose > 1: print(">> defining system prompt")
        self.system_prompt = system_prompt
        self.prequery_prompt = prequery_prompt
        self.messages = []
        self.reset_memory() # initialize messages with only the system prompt

        self.tool_schemas = tool_schemas # for calling tools

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
        #self.messages = []
        self.messages = [{
            "role": "system",
            "content": self.system_prompt
        }] # system not supported by gemma, it also needs user/assistant/user/assistant only

    # ----------------------------------------------------------------------------------------------
    # format the messages to be able to include tools and RAG
    def add_message(self, role, content, **kwargs):
        # add message for the model
        if role=="user" : content=self.prequery_prompt+content
        self.messages.append({
            "role": role,
            "content": content,
            **kwargs # add optional extra arguments (ex: tool's name)
        })

    def get_messages(self):
        return self.messages

    # ----------------------------------------------------------------------------------------------
    # apply chat templates, generate and return an answer
    def call(self):
        """ Note: the chat history (messages) is structured like this:
        chat_history = [ {'generated_text': [
                            {'role': 'user',       'content': '...'},
                            {'role': 'assistant',  'content': '...'}
                        ] } ]
        """

        if self.raw_model:
            inputs = self.tokenizer.apply_chat_template(
                self.messages,
                tools=self.tool_schemas,
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
            outputs = self.model(self.messages)
            response = outputs[0]['generated_text'][-1]["content"]

        self.add_message(role="model", content=response)

        return response
