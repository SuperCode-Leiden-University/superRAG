import os, pprint
from typing import Optional, List, Dict, Any, Literal
from llama_cpp import Llama

# importing my functions from other files
from supercode.src.configs.parse_config import *  # model's name and parameters
from supercode.src.model_loader.base_backend import BaseLLM

class LlamaCpp_import_model(BaseLLM):
    def __init__(self,
                 model_id: str,
                 quant_type: Literal["pretrained", "bits", "compressor", "sinq"],
                 gen_mode: Literal["multisample", "stream"],  # Literal specify which values are allowed
                 gen_args: Dict[str, Any],
                 # other settings, such as temperature and max tokens (default vals in config)
                 multi_sampl_args: Optional[Dict[str, Any]],  # optional for multi-sampling
                 # n_ctx: int = 2048, # context window
                 # n_gpu_layers: int = 0,  # 0 = CPU only
                 # chat_format: str = "llama-2",  # e.g., "llama-2", "chatml", "zephyr"
                 **kwargs,
                 ):
        print(">> loading with llama.cpp")
        ##### model's name and parameters are saved in the config
        self.model_id = model_id  # name of the model from Hugging Face
        self.gen_mode = gen_mode
        self.gen_args = gen_args  # other settings, such as num_return_sequences, temperature and max tokens (default vals in config)
        if gen_mode == "multisample": self.multi_sampl_args = multi_sampl_args

        # llama has different key names from transformers, so I need to rename them first:
        keys = self.gen_args.keys()
        if 'top-p' in keys: self.gen_args['top_p'] = self.gen_args.pop('top-p')
        if 'max_new_tokens' in keys: self.gen_args['max_tokens'] = self.gen_args.pop('max_new_tokens')

        # self.model = Llama(
        #     model_path=model_path,
        #     n_ctx=n_ctx,
        #     n_gpu_layers=n_gpu_layers,
        #     chat_format=chat_format,
        #     logits_all=False,
        #     embedding=False,
        #     **kwargs
        # )
        if "gguf" not in model_id.lower():
            raise ValueError(f"Model is not GGUF: {model_id}")

        self.model = Llama.from_pretrained(
            repo_id=self.model_id,
            filename="*8_0.gguf", # can be "*q8_0.gguf" or "*Q8_0.gguf"
            verbose=False, # otherwise it prints A LOT of useless stuff
            n_ctx=32000, # context window (default is 512...)
        )

    def apply_chat_template(self, prompt: str) -> str:
        """
        In llama-cpp, chat templates are handled via chat_format & messages,
        but you can still apply custom prompt formatting here if needed.
        """
        return prompt  # Or: self.model.apply_chat_template(messages, tokenize=False)

    def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # llama-cpp uses `tools` via structured output (if supported),
        # but basic tool calling is usually handled via prompt formatting.
        # For now: assume messages include tool calls in user/system prompts.

        # Optional: Convert tools into prompt text (see "Tool Handling" below)
        # Example: if tools are provided, prepend system message or tool schema

        # Call model chat completion
        if self.gen_mode == "stream":
            completion = self.model.create_chat_completion(
                messages=messages,
                tools=tools,
                #stream=True, # to see tokens while they are generated
                **self.gen_args, # default param
                # **kwargs # for changing the param at each call
            )
            response = completion["choices"][0]["message"]["content"]

        elif self.gen_mode == "multisample":
            n_samples = self.multi_sampl_args["num_return_sequences"]
            response = []
            for i in range(n_samples):
                completion = self.model.create_chat_completion(
                    messages=messages,
                    seed=-1, # random seed
                    **self.gen_args,  # includes temperature, top_k, etc.
                    stream=False,  # disable stream for batch processing
                )
                response.append(completion["choices"][0]["message"]["content"])

        else: raise ValueError(f"Unsupported generation mode: {self.gen_mode}")

        pprint.pprint(completion)
        """
        completion = {
            'id': 'chatcmpl-7b3f3ca1-1367-4736-9044-65cbf2d9c437', 
            'object': 'chat.completion', 
            'created': 1789387144, 
            'model': '/path/to/model/qwen2.5-coder-1.5b-instruct-q8_0.gguf', 
            
            'choices': [{
                'index': 0, 
                'message': {
                    'role': 'assistant', 
                    'content': 'OK!' }, 
                'logprobs': None, 
                'finish_reason': 'stop' }], 
            
            'usage': {
                'prompt_tokens': 165, 
                'completion_tokens': 2, 
                'total_tokens': 167 }}
        """
        # Normalize to match transformers output format (e.g., 'content', 'tool_calls')
        # response = {
        #     "content": response["choices"][0]["message"]["content"],
        #     "tool_calls": response["choices"][0]["message"].get("tool_calls", []),
        #     "usage": response.get("usage", {}),
        # } # response= {'content': '', 'tool_calls': [], 'usage': {'prompt_tokens': 166, 'completion_tokens': 2, 'total_tokens': 168}}

        #print(response)
        return response

    def get_tokenizer(self):
        # Return a minimal tokenizer interface
        class Tokenizer:
            def __init__(self, llm):
                self.llm = llm
            def encode(self, text: str) -> List[int]:
                return self.llm.model.tokenize(text.encode("utf-8"))
            def decode(self, tokens: List[int]) -> str:
                return self.llm.model.detokenize(tokens).decode("utf-8", errors="ignore")

        return Tokenizer(self)