from typing import Optional, List, Dict, Any, Literal
import torch, threading, pprint
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from compressed_tensors.offload import dispatch_model
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# importing my functions from other files
from supercode.src.configs.parse_config import *  # model's name and parameters
from supercode.src.configs.system_prompts import *  # prompts
from supercode.src.tools.code_processing import *  # tools
from supercode.src.model_loader.base_backend import BaseLLM


class Transformers_import_model(BaseLLM):
    """Load a LLM using Hugging Face Transformers."""
    def __init__(self,
                 model_id: str,
                 quant_type: Literal["pretrained", "bits", "compressor", "sinq"],
                 # specify that only these values are allowed
                 gen_args: Dict[str, Any] # other settings, such as temperature and max tokens (default vals in config)
                 ):
        print(">> loading with transformers")
        ##### model's name and parameters are saved in the config
        self.model_id = model_id  # name of the model from Hugging Face
        self.quant_type = quant_type
        self.gen_args = gen_args  # other settings, such as num_return_sequences, temperature and max tokens (default vals in config)

        # ----------------------------------------------------------------------------------------------
        ##### IMPORTING THE MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # load full precision models
        if quant_type == "pretrained":
            if verbose > 1: print(">> loading full precision model")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto", # automatically places layers on GPU(s) if possible
                dtype="auto"
            )
        # quantizing a model with BitsAndBytes (aka BnB)
        elif quant_type == "bits":  # for 4-8 bits quantization
            if verbose > 1: print(">> quantizing model with BitsAndBytes")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,  # or load_in_8bit=True
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                quantization_config=quant_config
            )
        # quantizing a model with llm-compressor for AWQ/GPTQ/etc...
        # NOTE: some methods may require finetuning, this is not implemented yet!
        # NOTE: AutoAWQ is deprecated and no longer maintained, it has been adopted by: https://github.com/vllm-project/llm-compressor
        elif quant_type == "compressor":  # apply vllm-project/llm-compressor
            if verbose > 1: print(">> quantizing model with llm-compressor")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                dtype="auto"
            )
            # Configure the quantization algorithm and scheme.
            # scheme=FP8, algo=GPTQ,
            recipe = QuantizationModifier(
                targets="Linear",
                scheme="FP8_BLOCK",
                ignore=["lm_head", "re:.*mlp.gate$"],
            )
            oneshot(model=self.model, recipe=recipe)
            dispatch_model(self.model)
        # quantizing a model with sinq (no finetuning)
        # if self.quant_type == "sinq": # DOESN'T WORK!!!
        #     if verbose > 1: print(">> quantizing model with SINQ")
        #     quant_config = SinqConfig(
        #         nbits=4,
        #         group_size=64,
        #         tiling_mode="1D",
        #         method="sinq",
        #         modules_to_not_convert=["lm_head"]
        #     )
        #     self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        #     self.model = AutoModelForCausalLM.from_pretrained(
        #         self.model_id,
        #         quantization_config=quant_config,
        #         device_map="auto",  # automatically places layers on GPU(s) if possible
        #         dtype="auto"
        #     )

        elif quant_type == "gguf":
            raise ValueError("GGUF format is not compatible with transformers, it should be used with Llama.cpp for loading on CPU")
        #     if verbose > 1: print(">> loading GGUF model")
        #     self.model = AutoModel.from_pretrained(
        #         self.model_id,
        #         device_map="auto",
        #         dtype="auto"
        #     )
        else: raise ValueError(f"Unsupported quantization type: {quant_type}")


    def apply_chat_template(self,
                       messages: List[Dict[str, str]],
                       tools: Optional[List[Dict[str, Any]]] = None
                       ):
        """Apply chat template/formatting."""
        inputs = self.tokenizer.apply_chat_template(
            messages, # chat history
            tools=tools, # tools schemas
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        return inputs


    def generate(self,
                 messages: List[Dict[str, str]],
                 tools: Optional[List[Dict[str, Any]]] = None,
                 **kwargs # override default generation arguments
                 ) -> str:
        """Generate a model response given messages and optional feedback from tools."""
        inputs = self.apply_chat_template(messages, tools=tools)

        """
        # streamer and threads are needed to see the response while it is being generated
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        # use a separate thread for generation and streaming
        def generate():
            self.model.generate(
                **inputs,
                **self.gen_args, # config such as temperature, max_new_tokens, etc...
                streamer=streamer,
                # **kwargs # override default generation arguments
            )
        # begin generation thread
        thread = threading.Thread(target=generate)
        thread.start()
        # collect streamed responses
        response = ""
        for token in streamer:
            print(token, end="", flush=True)
            response += token
        print()
        """



        # Generate multiple sequences from the single prompt
        input_length = inputs.input_ids.shape[1]
        with torch.no_grad():
            # this is a tensor with n elements, where n = num_return_sequences
            outputs = self.model.generate(
                **inputs,
                pad_token_id=self.tokenizer.eos_token_id,
                **self.gen_args, # config such as temperature, max_new_tokens, etc...
                # **kwargs # override default generation arguments
            )

        # Extract the generated tokens (skip the input prompt)
        generated_ids = outputs[
            :, #i*n_samples+i : i*n_samples+i+1,
            input_length :
        ]

        # Decode the generated part only
        responses = self.tokenizer.decode(
            generated_ids,
            skip_prompt=True,
            skip_special_tokens=True
        )
        return responses
