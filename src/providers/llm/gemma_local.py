import os
from typing import AsyncIterator, List
import structlog
from src.providers.base import ILLMProvider
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

logger = structlog.get_logger(__name__)

class GemmaLocalLLMProvider(ILLMProvider):
    def __init__(self, model_path: str):
        if Llama is None:
            raise ImportError("llama-cpp-python not installed. Run `pip install llama-cpp-python` with appropriate CMAKE_ARGS.")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=-1, # Use all GPU layers if possible
            verbose=False
        )

    async def generate_stream(self, messages: List[dict], system: str) -> AsyncIterator[str]:
        # Format for Gemma (usually ChatML or similar)
        prompt = f"<|system|>\n{system}\n"
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt += f"<|{role}|>\n{content}\n"
        prompt += "<|assistant|>\n"

        try:
            # llama-cpp-python's create_chat_completion is synchronous, we wrap it
            # or use the lower level __call__ with stream=True
            stream = self.llm(
                prompt,
                max_tokens=512,
                stop=["<|endoftext|>", "<|user|>", "<|system|>"],
                stream=True
            )
            
            for chunk in stream:
                token = chunk['choices'][0]['text']
                if token:
                    yield token
                    
        except Exception as e:
            logger.error("gemma_local_error", error=str(e))
            yield "Error: Local LLM failed."
