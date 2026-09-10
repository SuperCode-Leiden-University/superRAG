import os
from typing import List, Dict, Any, Optional
from llama_cpp import Llama

from src.model_loader.base_backend import BaseLLM

class LlamaCppLLM(BaseLLM):
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,  # 0 = CPU only
        chat_format: str = "llama-2",  # e.g., "llama-2", "chatml", "zephyr"
        **kwargs,
    ):
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            chat_format=chat_format,
            logits_all=False,
            embedding=False,
            **kwargs
        )
        self.model_path = model_path
        self.chat_format = chat_format

    def compile_prompt(self, prompt: str) -> str:
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
        response = self.model.create_chat_completion(
            messages=messages,
            **kwargs
        )
        # Normalize to match transformers output format (e.g., 'content', 'tool_calls')
        return {
            "content": response["choices"][0]["message"]["content"],
            "tool_calls": response["choices"][0]["message"].get("tool_calls", []),
            "usage": response.get("usage", {}),
        }

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