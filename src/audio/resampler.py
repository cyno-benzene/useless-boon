import librosa
import numpy as np

class Resampler:
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr

    def resample(self, audio: np.ndarray, source_sr: int) -> np.ndarray:
        if source_sr == self.target_sr:
            return audio
        return librosa.resample(audio, orig_sr=source_sr, target_sr=self.target_sr)
