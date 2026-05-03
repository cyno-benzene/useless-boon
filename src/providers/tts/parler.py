import torch
import os
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import numpy as np
from typing import AsyncIterator
import structlog
from src.providers.base import ITTSProvider

logger = structlog.get_logger(__name__)

class ParlerTTSProvider(ITTSProvider):
    def __init__(self, model_name: str = "ai4bharat/indic-parler-tts", device: str = "cuda"):
        # Auto-detect CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            logger.warn("cuda_not_available_for_parler", requested=device)
            device = "cpu"
        
        self.device = device
        token = os.getenv("HF_TOKEN")
        logger.info("initializing_parler_tts", model=model_name, device=device, authenticated=bool(token))
        
        # Load model and move to device
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_name, token=token).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        self.description = "A female speaker with a clear and natural voice." 

    async def synthesize_stream(self, text_chunks: AsyncIterator[str]) -> AsyncIterator[np.ndarray]:
        # Accumulate text until we have a sentence or a significant chunk
        # Parler is usually not "streaming" at the token level, but we can stream by sentences
        
        current_text = ""
        async for chunk in text_chunks:
            current_text += chunk
            
            # Simple sentence splitting
            if any(p in chunk for p in ['.', '?', '!', '\n']):
                # Synthesize the current sentence
                sentence = current_text.strip()
                if sentence:
                    logger.info("synthesizing_sentence", text=sentence)
                    audio = await self._synthesize_sentence(sentence)
                    yield audio
                current_text = ""
        
        # Final bit
        if current_text.strip():
            audio = await self._synthesize_sentence(current_text.strip())
            yield audio

    async def _synthesize_sentence(self, text: str) -> np.ndarray:
        input_ids = self.tokenizer(self.description, return_tensors="pt").input_ids.to(self.device)
        prompt_input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

        with torch.no_grad():
            generation = self.model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        
        audio_output = generation.cpu().numpy().squeeze().astype(np.float32)
        return audio_output
