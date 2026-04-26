from abc import ABC, abstractmethod
from typing import AsyncIterator, List, Tuple, Optional
import numpy as np

class INoiseSuppressionProvider(ABC):
    @abstractmethod
    async def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Process audio to remove noise."""
        pass

class IVADProvider(ABC):
    @abstractmethod
    async def is_speech(self, audio: np.ndarray, sr: int) -> bool:
        """Check if the audio chunk contains speech."""
        pass

    @abstractmethod
    async def get_segments(self, audio: np.ndarray, sr: int) -> List[np.ndarray]:
        """Detect and return speech segments from audio."""
        pass

class ISTTProvider(ABC):
    @abstractmethod
    async def transcribe_stream(self, audio_chunks: AsyncIterator[np.ndarray]) -> AsyncIterator[str]:
        """Transcribe an incoming stream of audio chunks."""
        pass

class ITurnDetectionProvider(ABC):
    @abstractmethod
    async def is_turn_complete(self, partial_transcript: str) -> Tuple[bool, float]:
        """Determine if the user has finished their turn based on transcript."""
        pass

class ILLMProvider(ABC):
    @abstractmethod
    async def generate_stream(self, messages: List[dict], system: str) -> AsyncIterator[str]:
        """Generate a response stream from the LLM."""
        pass

class ITTSProvider(ABC):
    @abstractmethod
    async def synthesize_stream(self, text_chunks: AsyncIterator[str]) -> AsyncIterator[np.ndarray]:
        """Synthesize audio from a stream of text chunks."""
        pass
