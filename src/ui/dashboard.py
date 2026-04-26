import asyncio
import json
import structlog
from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

logger = structlog.get_logger(__name__)
router = APIRouter()

# Global list of queues for SSE events (one per connected client)
event_queues = []

async def emit_event(event_type: str, data: dict):
    event = {"type": event_type, "data": data}
    for q in event_queues:
        await q.put(event)

@router.get("/events")
async def sse_events(request: Request):
    q = asyncio.Queue()
    event_queues.append(q)
    
    async def event_generator():
        # Send initial ping to confirm connection
        yield {
            "event": "ping",
            "data": json.dumps({"status": "connected"})
        }
        
        try:
            while True:
                if await request.is_disconnected():
                    break

                try:
                    # Use wait_for to check for disconnection periodically
                    event = await asyncio.wait_for(q.get(), timeout=20.0)
                    yield {
                        "data": json.dumps(event)
                    }
                except asyncio.TimeoutError:
                    # Send a heartbeat/ping to keep the connection alive
                    yield {
                        "event": "ping",
                        "data": json.dumps({"status": "heartbeat"})
                    }
                except asyncio.CancelledError:
                    break
        finally:
            event_queues.remove(q)

    return EventSourceResponse(event_generator())

@router.get("/config/providers")
async def get_providers():
    from src.registry.provider_registry import registry
    providers_info = {}
    for role, providers in registry._providers.items():
        providers_info[role] = [
            {"name": type(p).__name__, "state": registry._breakers[f"{role}_{type(p).__name__}"].state.value}
            for p in providers
        ]
    return providers_info
