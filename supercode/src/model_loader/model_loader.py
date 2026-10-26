from typing import Literal, Optional, Dict, Any

# importing my functions from other files
from supercode.src.model_loader.base_backend import BaseLLM
from supercode.src.model_loader.transformers_backend import Transformers_Import_Model
from supercode.src.model_loader.llama_cpp_backend import LlamaCpp_Import_Model
# from supercode.src.model_loader.vllm_backend import vLLM_Import_Model
from supercode.src.tools.code_processing import *

# importing variables from the config file
from supercode.src.configs.parse_config import *

# Main Model class that uses the backend
class Model():
    # ----------------------------------------------------------------------------------------------
    def __init__(self, model_args, system_prompt=None, prequery_prompt=None):
        print(f"\n### model_id = {model_args["model_id"]}")

        # define prompts and tools schemas
        self.system_prompt = system_prompt
        self.prequery_prompt = prequery_prompt

        # initialize chat history (short term memory)
        self.messages = []
        self.reset_memory() # Initialize messages with only the system prompt

        # initialize the backend (transformers in this case)
        self.model = self.load_llm(backend, **model_args)

    # ----------------------------------------------------------------------------------------------
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
        if "gguf" in model_args["model_id"].lower():
            return LlamaCpp_Import_Model(**model_args)
        elif backend == "transformers":
            return Transformers_Import_Model(**model_args)
        # elif backend == "vLLM":
        #     return vLLM_Import_Model(**model_args)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    # ----------------------------------------------------------------------------------------------
    def reset_memory(self):
        """Reset conversation memory."""
        self.messages = [{
            "role": "system",
            "content": self.system_prompt
        }]

    # ----------------------------------------------------------------------------------------------
    def add_message(self, role, content):
        """Add a message to the conversation history."""
        # NOTE: some models accept only "assistant/user/assistant/..." sequences
        #if verbose>0: print(">> saving message, role =", role)

        def message_format(role, content):
            self.messages.append({
                "role": role,
                "content": content,
                # "content": [ # for vision-language models
                #     #{"type": "image", "url": "https://path/to/img.jpg"},
                #     {"type": "text", "text": content}
                # ],
            })

        if role == "assistant":
            message_format(role, content)
        elif role == "user":
            message_format(role, self.prequery_prompt+content)

        elif role == "tool":
            # tool request
            message_format("assistant", f"calling tool {content["name"]} with the following arguments: {content["input"]}")
            # tool result
            message_format("user", f"the tool {content["name"]} retuned the following result: {content["result"]}")

        elif role == "code":
            # code request
            message_format("assistant", f"retrieving baseline code")
            # code result
            message_format("user", f"baseline code:\n```\n{content}\n```")


    # ----------------------------------------------------------------------------------------------
    def get_messages(self):
        """Get the conversation messages."""
        return self.messages

    # ----------------------------------------------------------------------------------------------
    def call(self, tool_schemas=None):
        """Call the model with the current message history."""
        # Generate response using the backend
        responses = self.model.generate(
            self.messages,
            tools=tool_schemas,
            # **self.gen_args
        )
        #print(responses)

        # this is actually a list of responses, now we select one:
        # response = select_response(responses) # def in code_processing
        #
        # self.add_message(role="assistant", content=response)

        return responses
