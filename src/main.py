import os
import sys
from dotenv import load_dotenv

# 0. Load .env immediately for tokens
load_dotenv()

# 1. IMMEDIATE ENVIRONMENT SETUP (Must happen before any ML imports)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(project_root, "models")
os.makedirs(models_dir, exist_ok=True)

# Torch Hub models (Silero VAD)
os.environ["TORCH_HOME"] = os.path.join(models_dir, "torch")
# HuggingFace / Transformers / Faster-Whisper
os.environ["HF_HOME"] = os.path.join(models_dir, "huggingface")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# General cache
os.environ["XDG_CACHE_HOME"] = os.path.join(models_dir, "cache")

# Add site-packages to PATH for CUDA DLLs (Windows specific)
import site
for path in site.getsitepackages():
    bin_path = os.path.join(path, "nvidia", "cublas", "bin")
    if os.path.exists(bin_path):
        os.environ["PATH"] += os.pathsep + bin_path
    bin_path = os.path.join(path, "nvidia", "cudnn", "bin")
    if os.path.exists(bin_path):
        os.environ["PATH"] += os.pathsep + bin_path

import asyncio
import numpy as np
import structlog
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Configure logging
structlog.configure(
    processors=[structlog.processors.JSONRenderer()]
)
logger = structlog.get_logger(__name__)

# Now we can safely import ML providers
from src.orchestrator.state import StateManager
from src.orchestrator.engine import PipelineEngine
from src.registry.provider_registry import registry
from src.providers.vad.silero import SileroVADProvider
from src.providers.stt.faster_whisper import FasterWhisperSTTProvider
from src.providers.llm.openrouter import OpenRouterLLMProvider
from src.providers.tts.pocket_tts import PocketTTSProvider
from src.ui.dashboard import router as dashboard_router, emit_event

app = FastAPI()
app.include_router(dashboard_router, prefix="/api/dashboard")

static_dir = os.path.join(os.path.dirname(__file__), "ui", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.on_event("startup")
async def startup_event():
    registry.register("vad", SileroVADProvider())
    registry.register("stt", FasterWhisperSTTProvider(model_size="base"))
    registry.register("llm", OpenRouterLLMProvider())
    registry.register("tts", PocketTTSProvider())
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

    async def relay_output():
        try:
            while True:
                if state_manager.barge_in_event.is_set():
                    # Clear queues and skip relaying on barge-in
                    while not engine.output_audio_q.empty(): engine.output_audio_q.get_nowait()
                    while not engine.llm_token_q.empty(): engine.llm_token_q.get_nowait()
                    while not engine.user_transcript_q.empty(): engine.user_transcript_q.get_nowait()
                    await asyncio.sleep(0.1)
                    continue

                # Relay User Transcript
                if not engine.user_transcript_q.empty():
                    text = await engine.user_transcript_q.get()
                    await websocket.send_json({"type": "user_transcript", "content": text})

                if not engine.output_audio_q.empty():
                    audio_chunk = await engine.output_audio_q.get()
                    await websocket.send_bytes(audio_chunk.tobytes())
                
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
            data = await websocket.receive_bytes()
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
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
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; padding: 20px; background-color: #f0f2f5; color: #333; }
            h1 { color: #1a73e8; margin-bottom: 10px; }
            #status { font-weight: bold; margin-bottom: 10px; padding: 5px 10px; border-radius: 20px; display: inline-block; background: #ddd; }
            #status.Connected { background: #d4edda; color: #155724; }
            
            #main-container { display: flex; flex-direction: column; align-items: center; max-width: 1000px; margin: 0 auto; gap: 20px; }
            
            #transcript-container { width: 100%; display: flex; flex-direction: column; gap: 10px; }
            #transcript { border: 1px solid #ccc; padding: 15px; min-height: 200px; background: white; text-align: left; overflow-y: auto; max-height: 400px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); display: flex; flex-direction: column; }
            
            .msg { margin-bottom: 10px; padding: 8px 12px; border-radius: 8px; max-width: 80%; line-height: 1.4; position: relative; }
            .user-msg { background: #e3f2fd; align-self: flex-end; border-left: 4px solid #2196f3; }
            .ai-msg { background: #f1f8e9; align-self: flex-start; border-left: 4px solid #4caf50; }
            .msg-label { font-weight: bold; font-size: 0.8em; margin-bottom: 4px; display: block; color: #666; }

            #log-container { width: 100%; border: 1px solid #ddd; padding: 10px; background: #222; color: #0f0; text-align: left; font-family: monospace; font-size: 11px; height: 120px; overflow-y: auto; border-radius: 8px; }
            
            .Speaking { color: blue; }
            .Listening { color: green; }
            .Thinking { color: orange; }
            
            #toast-container { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); z-index: 1000; width: 90%; max-width: 500px; }
            .toast { background: #333; color: white; padding: 15px; border-radius: 8px; margin-top: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); text-align: left; display: flex; flex-direction: column; }
            .toast-content { margin-bottom: 10px; word-break: break-all; font-family: monospace; font-size: 12px; }
            .toast-actions { display: flex; gap: 10px; }
            .toast-btn { background: #555; border: none; color: white; padding: 5px 10px; border-radius: 4px; cursor: pointer; font-size: 11px; }
            
            canvas { background: white; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.1); }
            .controls { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); display: flex; flex-wrap: wrap; justify-content: center; gap: 15px; width: 100%; box-sizing: border-box; }
            button { padding: 10px 20px; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; transition: transform 0.1s; }
            button:active { transform: scale(0.95); }
            #startBtn { background: #1a73e8; color: white; }
            #stopBtn { background: #d93025; color: white; }
            #enableAudio { background: #34a853; color: white; }
            button:disabled { background: #ccc !important; cursor: not-allowed; }
            
            .setting-item { display: flex; align-items: center; gap: 8px; font-size: 0.9em; }
        </style>
    </head>
    <body>
        <h1>Realtime Voice Assistant</h1>
        <div id="toast-container"></div>
        
        <div id="main-container">
            <div id="status">Disconnected</div>
            
            <div id="secure-notice" style="color: red; display: none; margin-bottom: 10px;">
                ⚠️ Mobile Testing: Microphone requires HTTPS. Use a tunnel like ngrok.
            </div>

            <div id="state" style="font-size: 1.2em; font-weight: bold;">State: IDLE</div>
            
            <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; width: 100%;">
                <div>
                    <div style="margin-bottom: 5px; font-size: 0.9em; color: #666;">User Voice (Input)</div>
                    <canvas id="waveform" width="300" height="80"></canvas>
                </div>
                <div>
                    <div style="margin-bottom: 5px; font-size: 0.9em; color: #666;">Assistant Voice (Output)</div>
                    <canvas id="assistant-waveform" width="300" height="80"></canvas>
                </div>
            </div>

            <div class="controls">
                <button id="enableAudio">1. Enable Sound</button>
                <button id="startBtn" disabled>2. Start Assistant</button>
                <button id="stopBtn" disabled>Stop</button>
                <div class="setting-item">
                    <input type="checkbox" id="echoShield" checked>
                    <label for="echoShield" title="Prevents mic data transmission while assistant is speaking">Echo Shield</label>
                </div>
                <button id="reconnectDash" style="background:none; border: 1px solid #ccc; color:#666; padding: 5px 10px; font-size: 12px;">Reset UI</button>
            </div>

            <div id="transcript-container">
                <div style="text-align: left; font-weight: bold; color: #666; font-size: 0.9em;">Conversation</div>
                <div id="transcript"></div>
            </div>

            <div style="width: 100%;">
                <div style="text-align: left; font-weight: bold; color: #666; font-size: 0.9em;">System Logs</div>
                <div id="log-container"></div>
            </div>
        </div>

        <script>
            let ws, audioContext, processor, source, eventSource, playbackContext, nextStartTime = 0;
            let currentAiMsgDiv = null;
            let isSpeakingState = false;

            const startBtn = document.getElementById('startBtn'), stopBtn = document.getElementById('stopBtn'), enableAudioBtn = document.getElementById('enableAudio');
            const status = document.getElementById('status'), stateText = document.getElementById('state'), transcript = document.getElementById('transcript');
            const logContainer = document.getElementById('log-container');
            const canvas = document.getElementById('waveform'), assistCanvas = document.getElementById('assistant-waveform');
            const ctx = canvas.getContext('2d'), assistCtx = assistCanvas.getContext('2d');
            const toastContainer = document.getElementById('toast-container');
            const echoShield = document.getElementById('echoShield');

            function showToast(message, detail = '') {
                const toast = document.createElement('div');
                toast.className = 'toast';
                toast.innerHTML = `<div class="toast-content">${message}\\n${detail}</div>
                    <div class="toast-actions"><button class="toast-btn" onclick="navigator.clipboard.writeText(\`${message}\\n${detail}\`); this.innerText='Copied!'">Copy</button>
                    <button class="toast-btn" onclick="this.closest('.toast').remove()">Close</button></div>`;
                toastContainer.appendChild(toast);
                setTimeout(() => { if(toast.parentElement) toast.remove(); }, 10000);
            }

            function addLog(msg) {
                const line = document.createElement('div');
                line.innerText = `[${new Date().toLocaleTimeString()}] ${msg}`;
                logContainer.appendChild(line);
                logContainer.scrollTop = logContainer.scrollHeight;
                if (logContainer.children.length > 50) logContainer.removeChild(logContainer.firstChild);
            }

            function appendTranscript(role, text, isPartial = false) {
                if (role === 'ai') {
                    if (!currentAiMsgDiv || isPartial === false) {
                        currentAiMsgDiv = document.createElement('div');
                        currentAiMsgDiv.className = 'msg ai-msg';
                        currentAiMsgDiv.innerHTML = `<span class="msg-label">Assistant</span><span class="content"></span>`;
                        transcript.appendChild(currentAiMsgDiv);
                    }
                    if (text) currentAiMsgDiv.querySelector('.content').innerText += text;
                } else {
                    const div = document.createElement('div');
                    div.className = 'msg user-msg';
                    div.innerHTML = `<span class="msg-label">You</span><span class="content">${text}</span>`;
                    transcript.appendChild(div);
                    currentAiMsgDiv = null;
                }
                transcript.scrollTop = transcript.scrollHeight;
            }

            window.onerror = (msg, url, line) => showToast("Global Error", `${msg} at ${line}`);
            window.onunhandledrejection = (e) => showToast("Promise Rejection", e.reason);

            let audioBuffer = new Float32Array(canvas.width), assistBuffer = new Float32Array(assistCanvas.width);
            function draw(c, x, b, color) {
                x.clearRect(0, 0, c.width, c.height);
                x.beginPath(); x.strokeStyle = color; x.lineWidth = 2;
                const sw = c.width / b.length; let xPos = 0;
                for (let i = 0; i < b.length; i++) {
                    const y = (c.height / 2) + (b[i] * 50);
                    if (i === 0) x.moveTo(xPos, y); else x.lineTo(xPos, y);
                    xPos += sw;
                }
                x.stroke();
            }
            function animate() { draw(canvas, ctx, audioBuffer, '#333'); draw(assistCanvas, assistCtx, assistBuffer, '#007bff'); requestAnimationFrame(animate); }
            animate();

            function connectDashboard() {
                if (eventSource) eventSource.close();
                eventSource = new EventSource('/api/dashboard/events');
                eventSource.onmessage = (e) => {
                    e.data.split('\\n').forEach(line => {
                        if (!line.trim()) return;
                        try {
                            const d = JSON.parse(line);
                            if (d.type === 'waveform') {
                                audioBuffer.set(audioBuffer.subarray(d.data.amplitude.length));
                                audioBuffer.set(d.data.amplitude, audioBuffer.length - d.data.amplitude.length);
                            } else if (d.type === 'assistant_waveform') {
                                assistBuffer.set(assistBuffer.subarray(d.data.amplitude.length));
                                assistBuffer.set(d.data.amplitude, assistBuffer.length - d.data.amplitude.length);
                            } else if (d.type === 'state') {
                                stateText.innerText = `State: ${d.data.state}`;
                                stateText.className = d.data.state;
                                isSpeakingState = (d.data.state === 'SPEAKING');
                                addLog(`State changed to ${d.data.state}`);
                                if (d.data.state === 'LISTENING') {
                                    audioBuffer.fill(0);
                                    assistBuffer.fill(0);
                                }
                            } else if (d.type === 'stop_audio') {
                                addLog("Interruption detected - stopping audio");
                                if (playbackContext) {
                                    playbackContext.close().then(() => {
                                        playbackContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
                                        nextStartTime = playbackContext.currentTime;
                                    });
                                }
                                assistBuffer.fill(0);
                            } else if (d.type === 'stt_final') {
                                addLog(`STT Final: ${d.data.transcript}`);
                            }
                        } catch (err) {}
                    });
                };
                eventSource.onerror = () => { eventSource.close(); setTimeout(connectDashboard, 3000); };
            }
            window.addEventListener('load', connectDashboard);
            document.getElementById('reconnectDash').onclick = () => { transcript.innerHTML = ''; logContainer.innerHTML = ''; connectDashboard(); };

            enableAudioBtn.onclick = () => {
                playbackContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
                nextStartTime = playbackContext.currentTime;
                enableAudioBtn.style.background = '#d0ffd0';
                enableAudioBtn.innerText = 'Audio Ready';
                enableAudioBtn.disabled = true;
                startBtn.disabled = false;
                showToast("Audio context initialized");
            };

            startBtn.onclick = async () => {
                if (!playbackContext) return showToast("Click 'Enable Audio Playback' first!");
                status.innerText = 'Connecting...';
                status.className = '';
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                ws.binaryType = 'arraybuffer';
                ws.onopen = () => { status.innerText = 'Connected'; status.className = 'Connected'; startBtn.disabled = true; stopBtn.disabled = false; startRecording(); };
                ws.onmessage = async (e) => {
                    if (typeof e.data === 'string') {
                        e.data.split('\\n').forEach(line => {
                            if (!line.trim()) return;
                            try {
                                const m = JSON.parse(line);
                                if (m.type === 'token') { appendTranscript('ai', m.content, true); }
                                else if (m.type === 'user_transcript') { appendTranscript('user', m.content); }
                            } catch (err) {}
                        });
                    } else if (playbackContext) {
                        const ad = new Float32Array(e.data);
                        const b = playbackContext.createBuffer(1, ad.length, 24000);
                        b.getChannelData(0).set(ad);
                        const s = playbackContext.createBufferSource();
                        s.buffer = b; s.connect(playbackContext.destination);
                        const st = Math.max(nextStartTime, playbackContext.currentTime);
                        s.start(st); nextStartTime = st + b.duration;
                    }
                };
                ws.onclose = () => { status.innerText = 'Disconnected'; status.className = ''; startBtn.disabled = false; stopBtn.disabled = true; };
            };

            stopBtn.onclick = () => { if (ws) ws.close(); if (processor) processor.disconnect(); if (source) source.disconnect(); };

            async function startRecording() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ 
                        audio: { 
                            echoCancellation: true, 
                            noiseSuppression: true,
                            autoGainControl: true
                        } 
                    });
                    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                    source = audioContext.createMediaStreamSource(stream);
                    processor = audioContext.createScriptProcessor(1024, 1, 1);
                    source.connect(processor); processor.connect(audioContext.destination);
                    processor.onaudioprocess = (e) => {
                        // Echo Shield logic: Don't send mic data if assistant is speaking and shield is on
                        if (echoShield.checked && isSpeakingState) return;
                        
                        if (ws && ws.readyState === 1) {
                            const input = e.inputBuffer.getChannelData(0);
                            const pcm = new Int16Array(input.length);
                            for (let i = 0; i < input.length; i++) pcm[i] = Math.max(-1, Math.min(1, input[i])) * 0x7FFF;
                            ws.send(pcm.buffer);
                        }
                    };
                } catch (err) {
                    showToast("Mic Error", err.message);
                    startBtn.disabled = false; stopBtn.disabled = true;
                }
            }
        </script>
    </body>
</html>
"""
