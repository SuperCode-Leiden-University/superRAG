import json

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# pipeline is for direct inference, with AutoModelForCausalLM, AutoTokenizer you load the raw model
# BitsAndBytesConfig is for quantization
#from awq import AutoAWQForCausalLM

from src.tools.manage_tools import * # import all the tools
from src.configs.parse_config import verbose

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

        self.tool_classifier = False # use the same model either to choose which tool to use or to generate an answer
        self.tools = get_tools() # tools for the agent
        self.schemas = [build_tool_schema(f) for f in self.tools.values()] # no errors, good

        self.messages = [] # history of the chat
        self.tool_messages = [] # messages for tool selection

        ##### IMPORTING THE MODEL
        if self.raw_model:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            if self.quant_type == "full":  # full precision
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map="auto",  # automatically places layers on GPU(s) if possible
                    #dtype="auto"
                )
            """
            if self.quant_type == "AWQ":  # AWQ quantization
                self.model = AutoAWQForCausalLM.from_quantized(
                    self.model_id,
                    device_map="auto",  # automatically places layers on GPU(s) if possible
                    #dtype="auto"
                )
            """
            if self.quant_type == "GPTQ":  # GPTQ quantization
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
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

        else:
            self.model = pipeline("text-generation", model=self.model_id)

    # ----------------------------------------------------------------------------------------------
    # format the messages to be able to include tools and RAG
    def message_format(self, user_prompt, tool_results=None, retriv_docs=None):
        # include the system prompts at the beginning of the chat
        if not self.messages:
            if verbose>1 : print(">> defining system prompt")
            # this is the chat history
            self.messages.append({
                "role": "system",
                "content": "You are a helpful AI assistant with access to a database and external tools. "
                           "When tool results are provided, treat them as authoritative facts. "
                           "Your job is to:"
                           "\n"
                           "- Use the tool outputs as the primary source of truth."
                           "\n"
                           "- Incorporate the tool results directly into your final answer."
                           "\n"
                           "- Keep all names, numbers, and factual details exactly as returned."
                           "\n"
                           "- Never describe the tool‑use process, reasoning steps, or how results were obtained."
                           "\n"
                           "- Never invent alternative answers when tool results are available. "
                           "You may summarize the results if the result is very long, "
                           "but you must not add unsupported details."
            })

            # this is for deciding if a tool is needed (and which one)
            self.tool_messages.append({
                "role": "system",
                "content": f"You are a classier for tool selection."
                           f"There are some tools described in:\n{self.schemas}\n"
                           "Your main job is to decide if there are suitable tools among the ones available "
                           "and in which order they should be used."
                           "Use the doc and name arguments to decide if a tool is relevant or not."
                           "\n"
                           "If you need to search the database you should request a number of retrieved documents "
                           "proportional to the difficulty of the query: generic or multistep queries are more difficult."
                           "Moreover, if you search the database you should use the user prompt as it is "
                           "to query the database: do not simply the query."
                           "\n"
                           "If you need to run the static analysis tool, you should first check the database "
                           "and find out the language was used in order to use the most appropriate flavor."
                           "\n"
                           "You must always respond with a list of JSON objects and nothing else, "
                           "you can include multiple tools if needed, or an empty list if none of the tools fits "
                           "(comments are NOT allowed inside the JSON object)."
            })

        # decide if the model needs a tool or to retrieve info from the DB
        if self.tool_classifier:
            if verbose>0 : print(">> check if there are useful tools")
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
                           #f"### Context\n{retriv_docs}\n\n"
                           f"### Question\n{user_prompt}"
                # it would be nice to automatically retrieve the machine and language of the code as extra info
            })

            for tool in tool_results:
                # """
                self.messages.append({
                    "role": "system",
                    "content": "The following data comes from tools. Treat it as correct and final. "
                               "Incorporate the tool results directly into your final answer. "
                               "Do not mention tools or describe how the data was obtained."
                })  # """
                self.messages.append({
                    "role": "tool",
                    "name": tool["name"],
                    "content": tool["result"]
                })

    # ----------------------------------------------------------------------------------------------
    # apply chat templates, generate and return an answer
    def chat_template(self, messages):
        if self.raw_model:
            if self.tool_classifier :
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    tools=self.schemas,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.model.device)
            else:
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.model.device)

            outputs = self.model.generate(**inputs, max_new_tokens=1200)
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        else:
            """ Note: chat structure is the following
            chat_history = [ {'generated_text': [
                                {'role': 'user',       'content': '...'},
                                {'role': 'assistant',  'content': '...'}
                            ] } ]
            """
            outputs = self.model(messages)
            response = outputs[0]['generated_text'][-1]["content"]

        return response


    # ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------
    def call(self, user_prompt, retriv_docs=None):
        """
        WORKFLOW:
            1) the model checks if it needs to call a tool or retrieve docs
            2) the tool is called and/or docs are retrieved
            3) the model integrates the results with its answer
        """
        # ----------------------------------------------------------------------------------------------
        # decide if it needs to retrieve context and/or to use tools
        self.tool_classifier = True
        self.message_format(user_prompt)

        # apply chat templates and return an answer
        response = self.chat_template(self.tool_messages)
        if verbose>1 : print("-------------------------------------- \n## tool: ", response, "\n--------------------------------------", sep="")
        tool_results = []

        if "{" in response:
            if verbose>0 : print(">> parsing response for tools")
            # sometimes the model adds ```json at the beginning and ``` at the end
            # or it may add some unrequired explanation
            # json = [ { "name": "tool_name", "arguments": { "arg_name": "value"} }, ... ]

            # find the actual start and end of the json
            if verbose>2 :
                begin = response.find("[")
                end = response.rfind("]") + 1
                print(">> json obj:\n", response[begin:end], sep='')

            tool_end = 0 ; tool_temp = 1 # initializing
            while tool_temp>0 :
                # find the actual start and end of each tool
                tool_begin = tool_end + response[tool_end:].find("{")
                tool_temp = response[tool_end:].find("},")+1 # this will be 0 for the last tool
                if tool_temp!=0:
                    tool_end = tool_end + tool_temp # find the second "}" for each tool
                else:
                    tool_end = response.rfind("}")+1

                #print(">> parsing tool:", response[tool_begin:tool_end])
                tool_request = json.loads(response[tool_begin:tool_end])
                tool_name = tool_request["name"]
                if tool_name in self.tools.keys():
                    if verbose > 0: print(">> found tool:", tool_name)
                    tool_result = dispatch_tool(self.tools, tool_name, tool_request["arguments"])
                    # save the results to pass them to the model
                    tool_results.append({
                        "name" : tool_name,
                        "result" : tool_result
                    })
                    if verbose > 1: print(">> tool result:", tool_result)

        #self.tool_messages.pop() # remove the last item

        # include the retrieved docs as context and feed it to the model
        self.tool_classifier = False
        self.message_format(user_prompt, tool_results, retriv_docs)

        # apply chat templates and return an answer
        response = self.chat_template(self.messages)

        return response