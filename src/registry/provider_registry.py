from typing import Dict, Any, List, Optional
import structlog
from src.providers.base import (
    IVADProvider, ISTTProvider, ILLMProvider, ITTSProvider,
    INoiseSuppressionProvider, ITurnDetectionProvider
)
from src.registry.circuit_breaker import CircuitBreaker, CircuitState

logger = structlog.get_logger(__name__)

class ProviderRegistry:
    def __init__(self):
        self._providers: Dict[str, List[Any]] = {} # role -> list of providers (primary, fallback)
        self._breakers: Dict[str, CircuitBreaker] = {}

    def register(self, role: str, provider: Any):
        if role not in self._providers:
            self._providers[role] = []
        
        self._providers[role].append(provider)
        provider_name = f"{role}_{type(provider).__name__}"
        if provider_name not in self._breakers:
            self._breakers[provider_name] = CircuitBreaker(provider_name)
            
        logger.info("registering_provider", role=role, type=type(provider).__name__)

    def _get_active_provider(self, role: str) -> Optional[Any]:
        providers = self._providers.get(role, [])
        for provider in providers:
            provider_name = f"{role}_{type(provider).__name__}"
            breaker = self._breakers.get(provider_name)
            if breaker and breaker.can_execute():
                return provider
        return None

    def get_vad(self) -> Optional[IVADProvider]:
        return self._get_active_provider("vad")

    def get_stt(self) -> Optional[ISTTProvider]:
        return self._get_active_provider("stt")

    def get_llm(self) -> Optional[ILLMProvider]:
        return self._get_active_provider("llm")

    def get_tts(self) -> Optional[ITTSProvider]:
        return self._get_active_provider("tts")

    def get_ns(self) -> Optional[INoiseSuppressionProvider]:
        return self._get_active_provider("ns")

    def get_turn(self) -> Optional[ITurnDetectionProvider]:
        return self._get_active_provider("turn")

    def record_success(self, role: str, provider: Any):
        provider_name = f"{role}_{type(provider).__name__}"
        breaker = self._breakers.get(provider_name)
        if breaker:
            breaker.record_success()

    def record_failure(self, role: str, provider: Any):
        provider_name = f"{role}_{type(provider).__name__}"
        breaker = self._breakers.get(provider_name)
        if breaker:
            breaker.record_failure()

# Singleton instance
registry = ProviderRegistry()
