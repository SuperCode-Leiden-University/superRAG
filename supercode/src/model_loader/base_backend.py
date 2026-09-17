from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


# blueprint for building subclasses to load the model with different backends
# this way all subclasses are consistent
class BaseLLM(ABC):
    """Interface for language models backends (transformers, llama.cpp, vLLM, etc...)."""

    @abstractmethod
    def apply_chat_template(self, prompt: str) -> str:
        """Apply chat template/formatting (e.g., Jinja-based)."""
        pass

    # this decorator let us declare a method that must be implemented in the subclasses
    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a model response given messages (and optionally tools)."""
        # note that the input include previous messages and optional tools, the output will be the answer
        pass

