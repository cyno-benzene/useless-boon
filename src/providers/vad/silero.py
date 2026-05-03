import torch
import numpy as np
from typing import List
import structlog
from src.providers.base import IVADProvider

logger = structlog.get_logger(__name__)

class SileroVADProvider(IVADProvider):
    def __init__(self, threshold: float = 0.5, device: str = "cpu"):
        # Auto-detect CUDA if requested
        if device == "cuda" and not torch.cuda.is_available():
            logger.warn("cuda_not_available_for_silero", requested=device)
            device = "cpu"
        
        self.device = device
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                          model='silero_vad',
                                          force_reload=False,
                                          onnx=False)
        (self.get_speech_timestamps, _, _, *_) = utils
        self.threshold = threshold
        self.model.to(device)
        self.model.eval()

    async def is_speech(self, audio: np.ndarray, sr: int) -> bool:
        if len(audio) == 0:
            return False
            
        # Silero VAD at 16000Hz is strict about chunk sizes (prefers 512)
        # We slice the incoming audio into 512-sample windows
        chunk_size = 512
        is_any_speech = False
        
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if len(chunk) < chunk_size:
                # Pad the last chunk with zeros if it's too small
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            
            # Convert to torch tensor and add batch dimension [1, 512]
            audio_tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(self.device)
            
            try:
                with torch.no_grad():
                    speech_prob = self.model(audio_tensor, 16000).item()
                if speech_prob > self.threshold:
                    is_any_speech = True
                    break # Found speech, no need to check other sub-chunks
            except Exception as e:
                logger.error("silero_vad_error", error=str(e), chunk_len=len(chunk))
        
        return is_any_speech

    async def get_segments(self, audio: np.ndarray, sr: int) -> List[np.ndarray]:
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        with torch.no_grad():
            timestamps = self.get_speech_timestamps(audio_tensor, self.model, sampling_rate=16000)
        
        segments = []
        for ts in timestamps:
            segments.append(audio[ts['start']:ts['end']])
        return segments

