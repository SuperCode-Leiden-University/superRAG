import os, pprint
from typing import Optional, List, Dict, Any, Literal

from src.model_loader.base_backend import BaseLLM

from vllm import LLM, SamplingParams
from typing import List, Optional
from src.model_loader.base_backend import BaseLLM

class vLLM_import_model(BaseLLM):
    def __init__(
        self,
        model_name: str,  # e.g., "meta-llama/Llama-2-7b-chat-hf"
        dtype: str = "auto",  # "auto", "half", "bfloat16", "float16"
        tensor_parallel_size: int = 1,
        max_num_seqs: int = 256,  # max concurrent sequences
        max_model_len: Optional[int] = None,  # override context length if needed
        gpu_memory_utilization: float = 0.9,
        **kwargs,
    ):
        # vLLM loads full model (no quantization by default)
        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,  # needed for many HF chat models
            **kwargs
        )

        # Preconfigure sampling params (reusable)
        self.default_sampling = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
            stop_token_ids=[],  # we'll handle stops manually
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        sampling = self.default_sampling.copy()
        sampling.max_tokens = max_tokens
        sampling.temperature = temperature
        sampling.top_p = top_p
        if stop:
            sampling.stop = stop

        outputs = self.llm.generate(prompt, sampling_params=sampling, **kwargs)
        # outputs is a list of RequestOutput (len=1 here)
        return outputs[0].outputs[0].text.strip()


    def batch_generate(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[str]:
        outputs = self.llm.generate(prompts, **kwargs)
        return [out.outputs[0].text.strip() for out in outputs]


    def get_tokenizer(self):
        """Return a tokenizer-compatible object (e.g., for encoding/decoding)."""
        pass


    def compile_prompt(self, prompt: str) -> str:
        """Apply chat template/formatting (e.g., Jinja-based)."""
        pass