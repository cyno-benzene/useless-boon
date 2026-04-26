import asyncio
import numpy as np
import structlog
import time
from typing import Optional, List
from src.orchestrator.state import StateManager, State
from src.providers.base import (
    INoiseSuppressionProvider,
    IVADProvider,
    ISTTProvider,
    ITurnDetectionProvider,
    ILLMProvider,
    ITTSProvider
)
from src.ui.dashboard import emit_event

logger = structlog.get_logger(__name__)

class PipelineEngine:
    def __init__(
        self,
        state_manager: StateManager,
        vad_provider: IVADProvider,
        stt_provider: ISTTProvider,
        llm_provider: ILLMProvider,
        tts_provider: Optional[ITTSProvider] = None,
        ns_provider: Optional[INoiseSuppressionProvider] = None,
        turn_provider: Optional[ITurnDetectionProvider] = None,
    ):
        self.state_manager = state_manager
        self.vad_provider = vad_provider
        self.stt_provider = stt_provider
        self.llm_provider = llm_provider
        self.tts_provider = tts_provider
        self.ns_provider = ns_provider
        self.turn_provider = turn_provider

        # Queues
        self.raw_audio_q = asyncio.Queue(maxsize=10)
        self.clean_audio_q = asyncio.Queue(maxsize=10)
        self.segments_q = asyncio.Queue(maxsize=10)
        self.transcript_q = asyncio.Queue(maxsize=10)
        self.llm_token_q = asyncio.Queue(maxsize=10)
        self.output_audio_q = asyncio.Queue(maxsize=10)

        self.tasks: List[asyncio.Task] = []
        self.running = False
        self.turn_start_time = 0

    async def start(self):
        self.running = True
        self.tasks = [
            asyncio.create_task(self._ns_worker()),
            asyncio.create_task(self._vad_worker()),
            asyncio.create_task(self._stt_worker()),
            asyncio.create_task(self._llm_worker()),
            asyncio.create_task(self._tts_worker()),
        ]
        logger.info("pipeline_engine_started")

    async def stop(self):
        self.running = False
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        logger.info("pipeline_engine_stopped")

    async def _update_dashboard_state(self, state: State):
        await emit_event("state", {"state": state.name})

    async def _ns_worker(self):
        while self.running:
            try:
                audio = await self.raw_audio_q.get()
                if self.ns_provider:
                    audio = await self.ns_provider.process(audio, 16000)
                await self.clean_audio_q.put(audio)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("ns_worker_error", error=str(e))

    async def _vad_worker(self):
        while self.running:
            try:
                audio = await self.clean_audio_q.get()
                
                # Emit waveform to dashboard (higher resolution)
                if len(audio) > 0:
                    await emit_event("waveform", {"amplitude": audio[::64].tolist()})

                is_speech = await self.vad_provider.is_speech(audio, 16000)
                
                if is_speech:
                    if self.state_manager.state == State.IDLE:
                        self.turn_start_time = time.time()
                        await self.state_manager.transition_to(State.LISTENING)
                        await self._update_dashboard_state(State.LISTENING)
                    elif self.state_manager.state == State.SPEAKING:
                        self.state_manager.set_barge_in()
                        await self.state_manager.transition_to(State.LISTENING)
                        await self._update_dashboard_state(State.LISTENING)
                    
                    await self.segments_q.put(audio)
                else:
                    # Logic for end of speech/silence timeout could go here
                    pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("vad_worker_error", error=str(e))

    async def _stt_worker(self):
        async def audio_iterator():
            while self.running:
                yield await self.segments_q.get()

        while self.running:
            try:
                async for transcript in self.stt_provider.transcribe_stream(audio_iterator()):
                    logger.debug("stt_partial", transcript=transcript)
                    
                    if self.turn_provider:
                        is_complete, confidence = await self.turn_provider.is_turn_complete(transcript)
                        logger.debug("turn_check", is_complete=is_complete, confidence=confidence)
                    else:
                        is_complete = True
                    
                    if is_complete:
                        await self.state_manager.transition_to(State.THINKING)
                        await self._update_dashboard_state(State.THINKING)
                        await self.transcript_q.put(transcript)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("stt_worker_error", error=str(e))

    async def _llm_worker(self):
        while self.running:
            try:
                transcript = await self.transcript_q.get()
                logger.info("llm_processing", prompt=transcript)
                
                messages = [{"role": "user", "content": transcript}]
                system_prompt = "You are a helpful voice assistant."
                
                async for token in self.llm_provider.generate_stream(messages, system_prompt):
                    if self.state_manager.barge_in_event.is_set():
                        logger.info("llm_generation_interrupted")
                        break
                    
                    if self.state_manager.state != State.SPEAKING:
                         await self.state_manager.transition_to(State.SPEAKING)
                         await self._update_dashboard_state(State.SPEAKING)
                         
                    await self.llm_token_q.put(token)
                
                # End of stream indicator
                await self.llm_token_q.put(None) 
                
                if not self.tts_provider:
                    await self.state_manager.transition_to(State.IDLE)
                    await self._update_dashboard_state(State.IDLE)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("llm_worker_error", error=str(e))

    async def _tts_worker(self):
        if not self.tts_provider:
            while self.running:
                await asyncio.sleep(1)
            return

        while self.running:
            try:
                async def token_iterator():
                    while self.running:
                        token = await self.llm_token_q.get()
                        if token is None:
                            break
                        if self.state_manager.barge_in_event.is_set():
                            logger.info("token_iterator_interrupted")
                            break
                        yield token

                async for audio_chunk in self.tts_provider.synthesize_stream(token_iterator()):
                    if self.state_manager.barge_in_event.is_set():
                        logger.info("tts_synthesis_interrupted")
                        while not self.output_audio_q.empty():
                            self.output_audio_q.get_nowait()
                        break
                    await self.output_audio_q.put(audio_chunk)
                
                if not self.state_manager.barge_in_event.is_set():
                    await self.state_manager.transition_to(State.IDLE)
                    await self._update_dashboard_state(State.IDLE)
                else:
                    self.state_manager.clear_barge_in()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("tts_worker_error", error=str(e))
