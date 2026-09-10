from typing import Literal, Optional, Dict, Any

from src.configs.parse_config import *
from src.model_loader.base_backend import BaseLLM
from src.model_loader.transformers_backend import TransformersLLM
from src.model_loader.llama_cpp_backend import LlamaCppLLM



# Main Model class that uses the backend
class Model():
    def __init__(self, model_args, system_prompt=None, prequery_prompt=None, tool_schemas=None):
        ##### extract model's name and parameters
        self.model_id = model_args["model_id"]  # name of the model from Hugging Face
        self.gen_args = model_args["gen_args"]  # other settings, such as temperature and max tokens (default vals in config)
        print("### model_id = " + self.model_id)

        # define prompts and tools schemas
        self.system_prompt = system_prompt
        self.prequery_prompt = prequery_prompt
        self.tool_schemas = tool_schemas

        # initialize chat history (short term memory)
        self.messages = []
        self.reset_memory() # Initialize messages with only the system prompt

        # initialize the backend (transformers in this case)
        self.model = self.load_llm(backend, **model_args)

    # this func takes the config as input and returns a model for any implemented backend
    def load_llm(self,
                 backend: Literal["transformers", "llama-cpp", "vLLM"],
                 **config # model configurations
                 ) -> BaseLLM:
        """
        Load the model with the specified backend.

        Args:
            backend: "transformers", "llama-cpp", "vLLM"
            **config: backend-specific config (e.g., model_path, device_map, etc.)
        """
        if backend == "transformers":
            return TransformersLLM(**config) # config is different for different backends
        elif backend == "llama-cpp":
            return LlamaCppLLM(**config)
        # elif backend == "vLLM":
        #     return vLLM(**config)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def reset_memory(self):
        """Reset conversation memory."""
        self.messages = [{
            "role": "system",
            "content": self.system_prompt
        }]

    def add_message(self, role, content, **kwargs):
        """Add a message to the conversation history."""
        if role == "user":
            content = self.prequery_prompt + content
        self.messages.append({
            "role": role,
            "content": content,
            **kwargs
        })

    def get_messages(self):
        """Get the conversation messages."""
        return self.messages

    def call(self):
        """Call the model with the current message history."""
        # Generate response using the backend
        response = self.model.generate(
            self.messages,
            tools=self.tool_schemas,
            **self.gen_args
        )
        self.add_message(role="assistent", content=response)

        return response
