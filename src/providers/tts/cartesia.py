import os
import json
import base64
import asyncio
import numpy as np
import websockets
import structlog
from typing import AsyncIterator
from src.providers.base import ITTSProvider
from dotenv import load_dotenv

load_dotenv()
logger = structlog.get_logger(__name__)

class CartesiaTTSProvider(ITTSProvider):
    def __init__(self, voice_id: str = "79a125e8-cd45-4c13-8a25-30e0a5cb4481"): # Default "British Lady"
        self.api_key = os.getenv("CARTESIA_API_KEY")
        if not self.api_key:
            raise ValueError("CARTESIA_API_KEY not found in environment")
        
        self.voice_id = voice_id
        self.sample_rate = 44100
        self.ws_url = f"wss://api.cartesia.ai/tts/websocket?api_key={self.api_key}&cartesia-version=2024-06-10"

    async def synthesize_stream(self, text_chunks: AsyncIterator[str]) -> AsyncIterator[np.ndarray]:
        async with websockets.connect(self.ws_url) as ws:
            # Task to pipe text chunks to Cartesia
            async def send_text():
                async for text in text_chunks:
                    if not text.strip(): continue
                    payload = {
                        "context_id": "vc-gemini-turn",
                        "model_id": "sonic-english",
                        "transcript": text,
                        "voice": {
                            "mode": "id",
                            "id": self.voice_id
                        },
                        "output_format": {
                            "container": "raw",
                            "encoding": "pcm_f32le",
                            "sample_rate": self.sample_rate
                        },
                        "continue": True
                    }
                    await ws.send(json.dumps(payload))
                
                # Signal end of stream
                await ws.send(json.dumps({"context_id": "vc-gemini-turn", "cancel": False, "continue": False}))

            sender_task = asyncio.create_task(send_text())

            try:
                async for message in ws:
                    response = json.loads(message)
                    if "audio" in response:
                        audio_data = base64.b64decode(response["audio"])
                        # Cartesia sends pcm_f32le
                        chunk = np.frombuffer(audio_data, dtype=np.float32)
                        yield chunk
                    
                    if response.get("done"):
                        break
            finally:
                sender_task.cancel()
