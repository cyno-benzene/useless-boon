import asyncio
import numpy as np
from typing import AsyncIterator
import structlog
from faster_whisper import WhisperModel
from src.providers.base import ISTTProvider

logger = structlog.get_logger(__name__)

class FasterWhisperSTTProvider(ISTTProvider):
    def __init__(self, model_size: str = "base", device: str = "cuda", compute_type: str = "int8_float16"):
        # Try to initialize with CUDA, fallback to CPU if libraries are missing
        try:
            logger.info("initializing_faster_whisper", device=device, model_size=model_size)
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception as e:
            logger.warn("faster_whisper_cuda_failed", error=str(e))
            logger.info("falling_back_to_cpu")
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.buffer = []

    async def transcribe_stream(self, audio_chunks: AsyncIterator[np.ndarray]) -> AsyncIterator[str]:
        # Process each chunk immediately. The engine now handles turn-level concatenation.
        async for current_audio in audio_chunks:
            if len(current_audio) == 0:
                continue
                
            try:
                # Use beam_size=1 for faster inference in real-time
                segments, info = await asyncio.to_thread(
                    self.model.transcribe, 
                    current_audio, 
                    beam_size=1,
                    language="en", # Hardcode to English to avoid language detection latency
                    task="transcribe"
                )
                
                text = "".join([s.text for s in segments]).strip()
                if text:
                    yield text
            except Exception as e:
                logger.error("transcription_error", error=str(e))
                if "cublas" in str(e).lower() or "cudnn" in str(e).lower():
                    logger.info("switching_to_cpu_due_to_missing_libs")
                    self.model = WhisperModel("base", device="cpu", compute_type="int8")
                    segments, info = await asyncio.to_thread(self.model.transcribe, current_audio, beam_size=1)
                    text = "".join([s.text for s in segments]).strip()
                    if text:
                        yield text
                else:
                    raise e
