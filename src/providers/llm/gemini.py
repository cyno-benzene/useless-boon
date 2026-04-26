import os
import google.generativeai as genai
from typing import AsyncIterator, List
import asyncio
import structlog
from src.providers.base import ILLMProvider
from dotenv import load_dotenv

load_dotenv()
logger = structlog.get_logger(__name__)

class GeminiLLMProvider(ILLMProvider):
    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    async def generate_stream(self, messages: List[dict], system: str) -> AsyncIterator[str]:
        # Convert messages to Gemini format
        # Note: This is a simplified conversion
        history = []
        for msg in messages[:-1]:
            history.append({"role": "user" if msg["role"] == "user" else "model", "parts": [msg["content"]]})
        
        chat = self.model.start_chat(history=history)
        last_message = messages[-1]["content"]
        
        try:
            # We add the system prompt to the first message if history is empty or as context
            # Better way is to use system_instruction in GenerativeModel init
            # but for this simplified version:
            full_prompt = f"{system}\n\nUser: {last_message}"
            
            response = await chat.send_message_async(full_prompt, stream=True)
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error("gemini_error", error=str(e))
            yield "Error: Could not generate response from Gemini."
