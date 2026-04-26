import time
import structlog
from enum import Enum
from typing import Optional, Callable

logger = structlog.get_logger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 3, recovery_timeout: int = 30):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.last_failure_time = 0

    def record_success(self):
        if self.state == CircuitState.HALF_OPEN:
            logger.info("circuit_breaker_closed", name=self.name)
            self.state = CircuitState.CLOSED
            self.failures = 0

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.failures >= self.failure_threshold:
            if self.state != CircuitState.OPEN:
                logger.warn("circuit_breaker_opened", name=self.name)
                self.state = CircuitState.OPEN

    def can_execute(self) -> bool:
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                logger.info("circuit_breaker_half_open", name=self.name)
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True
