from typing import Literal, Optional, Dict, Any

from supercode.src.configs.parse_config import *
from supercode.src.model_loader.base_backend import BaseLLM
from supercode.src.model_loader.transformers_backend import Transformers_Import_Model
from supercode.src.model_loader.llama_cpp_backend import LlamaCpp_Import_Model
# from supercode.src.model_loader.vllm_backend import vLLM_Import_Model



# Main Model class that uses the backend
class Model():
    def __init__(self, model_args, system_prompt=None, prequery_prompt=None):
        print("\n### model_id = " + model_args["model_id"])

        # define prompts and tools schemas
        self.system_prompt = system_prompt
        self.prequery_prompt = prequery_prompt

        # initialize chat history (short term memory)
        self.messages = []
        self.reset_memory() # Initialize messages with only the system prompt

        # initialize the backend (transformers in this case)
        self.model = self.load_llm(backend, **model_args)

    # this func takes the config as input and returns a model for any implemented backend
    def load_llm(self,
                 backend: Literal["transformers", "llama_cpp", "vLLM"],
                 **model_args # model configurations
                 ) -> BaseLLM:
        """
        Load the model with the specified backend.

        Args:
            backend: "transformers", "llama_cpp", "vLLM"
            **model_args: backend-specific config (e.g., model_path, device_map, etc.)
        """
        # config is different for different backends
        if backend == "transformers":
            return Transformers_Import_Model(**model_args)
        elif backend == "llama_cpp":
            return LlamaCpp_Import_Model(**model_args)
        # elif backend == "vLLM":
        #     return vLLM_Import_Model(**model_args)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def reset_memory(self):
        """Reset conversation memory."""
        self.messages = [{
            "role": "system",
            "content": self.system_prompt
        }]

    def add_message(self, role, content):
        """Add a message to the conversation history."""

        # EXAMPLE:
        """
        messages = [
            {
                "role": "user", 
                "content": "Read hello.py"
            },
            {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_01ABC",
                    "name": "read_file",
                    "input": {"file_path": "hello.py"},
                }],
            },
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_01ABC",
                    "content": "print('hello')",
                }],
            },
        ]                           
        """
        if role == "assistant":
            self.messages.append({
                "role": role,
                "content": content,
            })

        elif role == "user":
            self.messages.append({
                "role": role,
                "content": self.prequery_prompt + content,
            })

        # elif role == "code":
        #     self.messages.append({
        #         "role": "user",
        #         "content": "baseline code:\n```\n" + code + "\n```",
        #     })

        elif role == "tool":
            # tool request
            self.messages.append({
                "role": "assistant",
                "content": f"calling tool {content["name"]} with the following arguments: {content["input"]}",
            })
            # tool result
            self.messages.append({
                "role": "user",
                "content": f"the tool {content["name"]} retuned the following result: {content["result"]}",
            })

        # if role == "user": content = self.prequery_prompt + content
        # self.messages.append({
        #     "role": role,
        #     "content": content,
        #     **kwargs
        # })

    def get_messages(self):
        """Get the conversation messages."""
        return self.messages

    def call(self, tool_schemas=None):
        """Call the model with the current message history."""
        # Generate response using the backend
        response = self.model.generate(
            self.messages,
            tools=tool_schemas,
            # **self.gen_args
        )
        #print(response)

        self.add_message(role="assistent", content=response)

        return response
