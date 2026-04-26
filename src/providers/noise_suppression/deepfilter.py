import numpy as np
import torch
from df.enhance import enhance, init_df, load_audio, save_audio
from src.providers.base import INoiseSuppressionProvider
import structlog

logger = structlog.get_logger(__name__)

class DeepFilterNetProvider(INoiseSuppressionProvider):
    def __init__(self):
        # init_df returns (model, df_state, nb_config)
        self.model, self.df_state, _ = init_df()
        
    async def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        # DeepFilterNet expects torch tensor and specific sample rate
        # It usually works best at 48kHz internally but can handle others
        # We need to wrap it carefully for real-time chunks
        
        # For now, a simplified version using the enhance utility
        # In a real-time loop, we'd use the model directly frame-by-frame
        audio_tensor = torch.from_numpy(audio).float()
        
        # enhance expects (model, df_state, audio)
        # and returns enhanced audio
        try:
            enhanced = enhance(self.model, self.df_state, audio_tensor.unsqueeze(0))
            return enhanced.squeeze(0).numpy()
        except Exception as e:
            logger.error("deepfilter_error", error=str(e))
            return audio
