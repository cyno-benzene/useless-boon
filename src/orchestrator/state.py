from enum import Enum, auto
import asyncio
import structlog

logger = structlog.get_logger(__name__)

class State(Enum):
    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()

class StateManager:
    def __init__(self):
        self._state = State.IDLE
        self._state_changed = asyncio.Event()
        self._barge_in_event = asyncio.Event()

    @property
    def state(self) -> State:
        return self._state

    async def transition_to(self, new_state: State):
        if self._state == new_state:
            return

        logger.info("state_transition", from_state=self._state.name, to_state=new_state.name)
        self._state = new_state
        self._state_changed.set()
        self._state_changed.clear()

        # Only clear barge-in event when starting to listen for a NEW turn 
        # or when explicitly requested. Don't clear it immediately upon 
        # transitioning out of SPEAKING to allow workers to see it.
        if new_state == State.IDLE:
            self._barge_in_event.clear()

    def set_barge_in(self):
        if self._state == State.SPEAKING:
            logger.info("barge_in_detected")
            self._barge_in_event.set()

    def clear_barge_in(self):
        self._barge_in_event.clear()

    @property
    def barge_in_event(self) -> asyncio.Event:
        return self._barge_in_event

    async def wait_for_state_change(self):
        await self._state_changed.wait()
