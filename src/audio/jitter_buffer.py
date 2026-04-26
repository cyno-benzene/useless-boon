import asyncio
import numpy as np
import collections
import structlog

logger = structlog.get_logger(__name__)

class JitterBuffer:
    def __init__(self, target_depth_ms: int = 80, sr: int = 16000):
        self.sr = sr
        self.target_depth_samples = int((target_depth_ms / 1000) * sr)
        self.buffer = collections.deque()
        self.current_depth = 0

    def push(self, audio: np.ndarray):
        self.buffer.append(audio)
        self.current_depth += len(audio)

    def pop(self) -> Optional[np.ndarray]:
        if not self.buffer:
            return None
        
        chunk = self.buffer.popleft()
        self.current_depth -= len(chunk)
        return chunk

    def is_ready(self) -> bool:
        return self.current_depth >= self.target_depth_samples

    def clear(self):
        self.buffer.clear()
        self.current_depth = 0
