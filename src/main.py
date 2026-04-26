import asyncio
import numpy as np
import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from src.orchestrator.state import StateManager
from src.orchestrator.engine import PipelineEngine
from src.registry.provider_registry import registry
from src.providers.vad.silero import SileroVADProvider
from src.providers.stt.faster_whisper import FasterWhisperSTTProvider
from src.providers.llm.openrouter import OpenRouterLLMProvider
import json

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger(__name__)

from src.ui.dashboard import router as dashboard_router, emit_event
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

# Include dashboard router
app.include_router(dashboard_router, prefix="/api/dashboard")

# Mount static files if directory exists
static_dir = os.path.join(os.path.dirname(__file__), "ui", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


from dotenv import load_dotenv
load_dotenv()

import os
# Set cache directories to the project folder to save C: drive space
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(project_root, "models")
os.makedirs(models_dir, exist_ok=True)

# Add site-packages to PATH for CUDA DLLs (important for Windows)
import site
import sys
for path in site.getsitepackages():
    bin_path = os.path.join(path, "nvidia", "cublas", "bin")
    if os.path.exists(bin_path):
        os.environ["PATH"] += os.pathsep + bin_path
    bin_path = os.path.join(path, "nvidia", "cudnn", "bin")
    if os.path.exists(bin_path):
        os.environ["PATH"] += os.pathsep + bin_path

os.environ["TORCH_HOME"] = os.path.join(models_dir, "torch")
# HuggingFace models (Parler, etc.)
os.environ["HF_HOME"] = os.path.join(models_dir, "huggingface")
# General cache (often used as fallback)
os.environ["XDG_CACHE_HOME"] = os.path.join(models_dir, "cache")

@app.on_event("startup")
async def startup_event():
    # Primary providers
    registry.register("vad", SileroVADProvider())
    registry.register("stt", FasterWhisperSTTProvider(model_size="base"))
    registry.register("llm", OpenRouterLLMProvider())
    
    # Fallback/Optional providers
    # registry.register("llm", GemmaLocalLLMProvider(model_path="models/gemma.gguf")) 
    # registry.register("tts", ParlerTTSProvider())
    
    logger.info("application_startup_complete")


@app.get("/")
async def get():
    return HTMLResponse(content=HTML_CONTENT)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("websocket_connected")

    state_manager = StateManager()
    engine = PipelineEngine(
        state_manager=state_manager,
        vad_provider=registry.get_vad(),
        stt_provider=registry.get_stt(),
        llm_provider=registry.get_llm(),
        tts_provider=registry.get_tts()
    )

    await engine.start()

    # Task to relay output audio/text back to websocket
    async def relay_output():
        try:
            while True:
                # For Phase 1, we might just have text tokens if TTS is not ready
                # Let's check both
                if not engine.output_audio_q.empty():
                    audio_chunk = await engine.output_audio_q.get()
                    await websocket.send_bytes(audio_chunk.tobytes())
                
                # Also send state/text for UI feedback
                if not engine.llm_token_q.empty():
                    token = await engine.llm_token_q.get()
                    if token:
                        await websocket.send_json({"type": "token", "content": token})
                
                await asyncio.sleep(0.01)
        except Exception as e:
            logger.error("relay_output_error", error=str(e))

    relay_task = asyncio.create_task(relay_output())

    try:
        while True:
            # Receive binary frames (PCM 16kHz 16-bit)
            data = await websocket.receive_bytes()
            
            # Convert bytes to numpy float32
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Push to engine
            await engine.raw_audio_q.put(audio_data)
            
    except WebSocketDisconnect:
        logger.info("websocket_disconnected")
    except Exception as e:
        logger.error("websocket_error", error=str(e))
    finally:
        await engine.stop()
        relay_task.cancel()

HTML_CONTENT = """
<!DOCTYPE html>
<html>
    <head>
        <title>Realtime Voice Assistant</title>
        <style>
            body { font-family: sans-serif; text-align: center; padding-top: 50px; background-color: #f9f9f9; }
            #status { font-weight: bold; margin-bottom: 20px; }
            #transcript { border: 1px solid #ccc; padding: 10px; min-height: 100px; width: 80%; margin: 20px auto; background: white; text-align: left; }
            .Speaking { color: blue; }
            .Listening { color: green; }
            .Thinking { color: orange; }
            
            /* Toast Styles */
            #toast-container { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); z-index: 1000; width: 90%; max-width: 500px; }
            .toast { background: #333; color: white; padding: 15px; border-radius: 8px; margin-top: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); text-align: left; display: flex; flex-direction: column; }
            .toast-content { margin-bottom: 10px; word-break: break-all; font-family: monospace; font-size: 12px; }
            .toast-actions { display: flex; gap: 10px; }
            .toast-btn { background: #555; border: none; color: white; padding: 5px 10px; border-radius: 4px; cursor: pointer; font-size: 11px; }
            .toast-btn:hover { background: #777; }
        </style>
    </head>
    <body>
        <h1>Voice Assistant</h1>
        <div id="toast-container"></div>
        <div id="status">Disconnected</div>
        <div id="secure-notice" style="color: red; display: none; margin-bottom: 10px;">
            ⚠️ Mobile Testing: Microphone requires HTTPS. Use a tunnel like ngrok or access via localhost.
        </div>
        <div id="state" style="margin-top: 10px; font-size: 1.2em;">State: IDLE</div>
        <canvas id="waveform" width="600" height="100" style="border: 1px solid #eee; margin-top: 10px;"></canvas>
        <div style="margin-top: 20px;">
            <button id="startBtn">Start Recording</button>
            <button id="stopBtn" disabled>Stop Recording</button>
            <button id="reconnectDash" style="margin-left: 10px;">Reconnect Dashboard</button>
        </div>
        <div id="transcript" style="margin-top: 20px;"></div>

        <script>
            let ws;
            let audioContext;
            let processor;
            let source;
            let eventSource;

            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const status = document.getElementById('status');
            const stateText = document.getElementById('state');
            const transcript = document.getElementById('transcript');
            const canvas = document.getElementById('waveform');
            const secureNotice = document.getElementById('secure-notice');
            const ctx = canvas.getContext('2d');
            const toastContainer = document.getElementById('toast-container');

            function showToast(message, detail = '') {
                const toast = document.createElement('div');
                toast.className = 'toast';
                
                const content = document.createElement('div');
                content.className = 'toast-content';
                content.innerText = `${message}\n${detail}`;
                
                const actions = document.createElement('div');
                actions.className = 'toast-actions';
                
                const copyBtn = document.createElement('button');
                copyBtn.className = 'toast-btn';
                copyBtn.innerText = 'Copy Error';
                copyBtn.onclick = () => {
                    navigator.clipboard.writeText(`${message}\n${detail}`);
                    copyBtn.innerText = 'Copied!';
                    setTimeout(() => copyBtn.innerText = 'Copy Error', 2000);
                };
                
                const closeBtn = document.createElement('button');
                closeBtn.className = 'toast-btn';
                closeBtn.innerText = 'Close';
                closeBtn.onclick = () => toast.remove();
                
                actions.appendChild(copyBtn);
                actions.appendChild(closeBtn);
                toast.appendChild(content);
                toast.appendChild(actions);
                toastContainer.appendChild(toast);
                
                setTimeout(() => { if(toast.parentElement) toast.remove(); }, 15000);
            }

            // Global error handler for mobile
            window.onerror = function(msg, url, line, col, error) {
                showToast("Global Error", `${msg}\nAt: ${url}:${line}:${col}\nStack: ${error ? error.stack : 'N/A'}`);
                return false;
            };

            window.onunhandledrejection = function(event) {
                showToast("Unhandled Promise Rejection", event.reason);
            };

            // Check for secure context
            if (!window.isSecureContext && window.location.hostname !== 'localhost') {
                secureNotice.style.display = 'block';
                showToast("Insecure Context Detected", "Microphone access will be blocked by the browser because this is not HTTPS or localhost.");
            }

            // Waveform visualization
            let audioBuffer = new Float32Array(canvas.width);
            function drawWaveform() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.beginPath();
                ctx.strokeStyle = '#333';
                ctx.lineWidth = 2;
                const sliceWidth = canvas.width / audioBuffer.length;
                let x = 0;
                for (let i = 0; i < audioBuffer.length; i++) {
                    const v = audioBuffer[i] * 50;
                    const y = (canvas.height / 2) + v;
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                    x += sliceWidth;
                }
                ctx.stroke();
                requestAnimationFrame(drawWaveform);
            }
            drawWaveform();

            function connectDashboard() {
                if (eventSource) eventSource.close();
                console.log('Connecting to dashboard SSE...');
                eventSource = new EventSource('/api/dashboard/events');
                
                eventSource.onopen = () => console.log('Dashboard connected');
                
                eventSource.onmessage = (event) => {
                    // SSE data can sometimes contain multiple JSON objects separated by newlines
                    const lines = event.data.split('\\n');
                    for (const line of lines) {
                        if (!line.trim()) continue;
                        try {
                            const data = JSON.parse(line);
                            if (data.type === 'waveform') {
                                const newAmp = data.data.amplitude;
                                audioBuffer.set(audioBuffer.subarray(newAmp.length));
                                audioBuffer.set(newAmp, audioBuffer.length - newAmp.length);
                            } else if (data.type === 'state') {
                                stateText.innerText = `State: ${data.data.state}`;
                                stateText.className = data.data.state;
                            }
                        } catch (e) {
                            console.error('Error parsing SSE line:', e, line);
                        }
                    }
                };

                eventSource.onerror = (err) => {
                    console.warn('Dashboard connection lost. Retrying in 3s...');
                    eventSource.close();
                    setTimeout(connectDashboard, 3000);
                };
            }

            // Wait for full load before connecting SSE
            window.addEventListener('load', connectDashboard);
            document.getElementById('reconnectDash').onclick = connectDashboard;

            startBtn.onclick = async () => {
                try {
                    status.innerText = 'Connecting...';
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                    ws.binaryType = 'arraybuffer';

                    ws.onopen = () => {
                        console.log('WebSocket connected');
                        status.innerText = 'Connected';
                        startBtn.disabled = true;
                        stopBtn.disabled = false;
                        startRecording();
                    };

                    ws.onmessage = (event) => {
                        try {
                            if (typeof event.data === 'string') {
                                // Some environments might batch multiple JSON messages
                                const lines = event.data.split('\\n');
                                for (const line of lines) {
                                    if (!line.trim()) continue;
                                    try {
                                        const msg = JSON.parse(line);
                                        if (msg.type === 'token') {
                                            transcript.innerText += msg.content;
                                            // Auto-scroll to bottom
                                            transcript.scrollTop = transcript.scrollHeight;
                                        }
                                    } catch (innerErr) {
                                        // If it's not JSON, it might be raw text from a different stage
                                        console.warn('Non-JSON message on WebSocket:', line);
                                    }
                                }
                            }
                        } catch (e) {
                            console.error('Error processing message:', e);
                        }
                    };

                    ws.onclose = (event) => {
                        console.log('WebSocket closed:', event.code, event.reason);
                        status.innerText = 'Disconnected';
                        startBtn.disabled = false;
                        stopBtn.disabled = true;
                        if (event.code !== 1000) {
                            showToast("WebSocket Closed", `Code: ${event.code}\nReason: ${event.reason || 'None'}`);
                        }
                    };

                    ws.onerror = (err) => {
                        console.error('WebSocket error:', err);
                        showToast("WebSocket Error", "Check console or network connection.");
                    };
                } catch (err) {
                    showToast("Failed to initiate connection", err.message);
                }
            };

            stopBtn.onclick = () => {
                try {
                    if (ws) ws.close();
                    if (processor) processor.disconnect();
                    if (source) source.disconnect();
                } catch (err) {
                    showToast("Error stopping", err.message);
                }
            };

            async function startRecording() {
                try {
                    console.log('Starting recording sequence...');
                    // Check if getUserMedia is available
                    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                        const msg = 'Microphone API (getUserMedia) is not available. This usually happens on non-HTTPS connections or in very old browsers.';
                        status.innerText = `Error: ${msg}`;
                        showToast("Microphone API Missing", msg);
                        return;
                    }

                    console.log('Requesting microphone access...');
                    const stream = await navigator.mediaDevices.getUserMedia({ 
                        audio: {
                            echoCancellation: true,
                            noiseSuppression: true,
                            autoGainControl: false
                        }
                    });
                    
                    console.log('Microphone access granted. Setting up audio pipeline...');
                    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                    source = audioContext.createMediaStreamSource(stream);
                    
                    // Simplified buffer handling for demo
                    processor = audioContext.createScriptProcessor(1024, 1, 1);
                    source.connect(processor);
                    processor.connect(audioContext.destination);

                    processor.onaudioprocess = (e) => {
                        if (ws && ws.readyState === WebSocket.OPEN) {
                            const inputData = e.inputBuffer.getChannelData(0);
                            // Convert to Int16
                            const pcmData = new Int16Array(inputData.length);
                            for (let i = 0; i < inputData.length; i++) {
                                pcmData[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7FFF;
                            }
                            ws.send(pcmData.buffer);
                        }
                    };
                    console.log('Recording active.');
                } catch (err) {
                    console.error('Microphone access error:', err);
                    let msg = err.message;
                    if (err.name === 'NotAllowedError') {
                        msg = 'Microphone permission denied. Please allow microphone access in your browser settings.';
                    } else if (err.name === 'NotFoundError') {
                        msg = 'No microphone found on this device.';
                    } else if (err.name === 'SecurityError') {
                        msg = 'HTTPS required for microphone access. Use https:// or localhost.';
                    }
                    
                    status.innerText = `Error: ${msg}`;
                    showToast("Microphone Error", `${err.name}: ${msg}\n${err.stack || ''}`);
                    
                    stopBtn.disabled = true;
                    startBtn.disabled = false;
                    if (ws) ws.close();
                }
            }
        </script>
    </body>
</html>
"""
