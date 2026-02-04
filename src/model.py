import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# pipeline is for direct inference, with AutoModelForCausalLM, AutoTokenizer you load the raw model
# BitsAndBytesConfig is for quantization

from src.tools import * # import all the tools

class Model():
    def __init__(self, model_id, raw_model=True, quant_type="full"):
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

        self.tools = get_tools() # tools for the agent

        ##### IMPORTING THE MODELS
        # "model" is for processing text and generating an answer
        if self.raw_model:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            if self.quant_type == "full":  # full precision
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map="auto"  # automatically places layers on GPU(s) if possible
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

            if self.quant_type == "GPTQ":  # GPTQ quantization
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map="auto",  # automatically places layers on GPU(s) if possible
                    dtype=torch.float16,  # recommended for GPTQ
                    trust_remote_code=True
                )

        else:
            self.model = pipeline("text-generation", model=self.model_id)

    def call(self, user_prompt, retriv_docs=None):
        # include the retrieved docs as context and feed it to the model
        if retriv_docs is None: # this might be unnecessary, but if feels weird to pass an empty context
            messages = [{
                "role": "user",
                "content": f"Use the following context to answer the question.\n\n"
                           f"### Question\n{user_prompt}"
            }]
        else:
            messages = [{
                "role": "user",
                "content": f"Use the following context to answer the question.\n\n"
                           f"### Context\n{retriv_docs}\n\n"
                           f"### Question\n{user_prompt}"
                # it would be nice to automatically retrieve the machine and language of the code as extra info
            }]

        if self.raw_model:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            outputs = self.model.generate(**inputs, max_new_tokens=1200)
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])

        else:
            """ Note: chat structure is the following
            chat_history = [ {'generated_text': [
                                {'role': 'user',       'content': '...'},
                                {'role': 'assistant',  'content': "..."}
                            ] } ]
            """

            outputs = self.model(messages)
            response = outputs[0]['generated_text'][-1]['content']


        return response