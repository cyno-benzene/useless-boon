import torch
import numpy as np
from typing import AsyncIterator
import structlog
from src.providers.base import ITTSProvider
from pocket_tts import TTSModel

logger = structlog.get_logger(__name__)

class PocketTTSProvider(ITTSProvider):
    def __init__(self, voice: str = "alba"):
        logger.info("initializing_pocket_tts", voice=voice)
        self.model = TTSModel.load_model()
        self.voice_state = self.model.get_state_for_audio_prompt(voice)
        self.sample_rate = self.model.sample_rate

    async def synthesize_stream(self, text_chunks: AsyncIterator[str]) -> AsyncIterator[np.ndarray]:
        # Pocket TTS is fast but generate_audio is typically one-shot.
        # We process text by sentences to simulate streaming.
        
        current_text = ""
        async for chunk in text_chunks:
            current_text += chunk
            
            # Simple sentence splitting
            if any(p in chunk for p in ['.', '?', '!', '\n']):
                sentence = current_text.strip()
                if sentence:
                    logger.info("synthesizing_sentence_pocket", text=sentence)
                    audio = await self._synthesize_sentence(sentence)
                    if audio is not None:
                        yield audio
                current_text = ""
        
        # Final bit
        if current_text.strip():
            audio = await self._synthesize_sentence(current_text.strip())
            if audio is not None:
                yield audio

    async def _synthesize_sentence(self, text: str) -> np.ndarray:
        try:
            # Pocket TTS generate_audio returns a torch tensor
            audio_tensor = self.model.generate_audio(self.voice_state, text)
            # Convert to numpy float32
            return audio_tensor.numpy().astype(np.float32)
        except Exception as e:
            logger.error("pocket_tts_error", error=str(e), text=text)
            return None
