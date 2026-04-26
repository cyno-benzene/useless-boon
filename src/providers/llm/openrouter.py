import os
from openai import AsyncOpenAI
from typing import AsyncIterator, List
import structlog
from src.providers.base import ILLMProvider
from dotenv import load_dotenv

load_dotenv()
logger = structlog.get_logger(__name__)

class OpenRouterLLMProvider(ILLMProvider):
    def __init__(self, model_name: str = "openai/gpt-4o"):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")
        
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model_name = model_name

    async def generate_stream(self, messages: List[dict], system: str) -> AsyncIterator[str]:
        # Include system prompt in the messages
        full_messages = [{"role": "system", "content": system}] + messages
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=full_messages,
                stream=True,
                extra_headers={
                    "HTTP-Referer": "https://github.com/useless-boon", # Placeholder
                    "X-OpenRouter-Title": "VC-Gemini Service",
                }
            )
            
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error("openrouter_error", error=str(e))
            yield "Error: Could not generate response from OpenRouter."
