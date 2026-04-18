import asyncio
import base64
import io
import json
import logging
import os
import subprocess
import tempfile
import wave
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("voice-gateway")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STT_MODEL_SIZE = os.getenv("STT_MODEL_SIZE", "large-v3")
STT_DEVICE = os.getenv("STT_DEVICE", "cuda")
STT_COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "float16")
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "pl")

PIPER_BINARY = os.getenv("PIPER_BINARY", "/usr/local/bin/piper")
PIPER_MODEL = os.getenv("PIPER_MODEL", "/models/tts/pl_PL-darkman-medium.onnx")
PIPER_SPEAKER = int(os.getenv("PIPER_SPEAKER", "0"))
TTS_SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", "22050"))

AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "120"))

_DEFAULT_AGENTS = {
    "scenarzysta": {"url": "http://scenarzysta:8000/api/invoke", "description": "Generator scenariuszy"},
    "skryba": {"url": "http://skryba:8000/api/invoke", "description": "Asystent dokumentów"},
    "sprawdzai": {"url": "http://sprawdzai:8000/api/invoke", "description": "Weryfikator treści"},
    "gacek": {"url": "http://gacek:8000/api/invoke", "description": "Asystent ogólny"},
    "sufler": {"url": "http://sufler:8000/api/invoke", "description": "Asystent informacyjny"},
}

try:
    AGENTS: dict[str, dict] = json.loads(os.getenv("AGENTS_CONFIG", "{}")) or _DEFAULT_AGENTS
except json.JSONDecodeError:
    log.warning("AGENTS_CONFIG is not valid JSON — using defaults")
    AGENTS = _DEFAULT_AGENTS

# ---------------------------------------------------------------------------
# Global model handles
# ---------------------------------------------------------------------------

stt_model = None  # faster_whisper.WhisperModel instance


# ---------------------------------------------------------------------------
# Lifespan: load models at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global stt_model

    # Validate Piper
    if not os.path.isfile(PIPER_BINARY):
        log.warning("Piper binary not found at %s — TTS will fail", PIPER_BINARY)
    if not os.path.isfile(PIPER_MODEL):
        log.warning("Piper model not found at %s — TTS will fail", PIPER_MODEL)

    # Load Whisper
    log.info("Loading STT model '%s' on %s (%s)...", STT_MODEL_SIZE, STT_DEVICE, STT_COMPUTE_TYPE)
    try:
        from faster_whisper import WhisperModel
        stt_model = WhisperModel(
            STT_MODEL_SIZE,
            device=STT_DEVICE,
            compute_type=STT_COMPUTE_TYPE,
        )
        log.info("STT model loaded successfully")
    except Exception as exc:
        log.error("Failed to load STT model: %s", exc)

    yield

    log.info("Shutting down Voice Gateway")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Voice Gateway", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Helpers — STT
# ---------------------------------------------------------------------------

def _transcribe_bytes(audio_bytes: bytes) -> str:
    if stt_model is None:
        raise RuntimeError("STT model not loaded")

    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments, info = stt_model.transcribe(
            tmp_path,
            language=STT_LANGUAGE,
            beam_size=5,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        log.info("STT result (lang=%s, prob=%.2f): %s", info.language, info.language_probability, text)
        return text
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Helpers — TTS
# ---------------------------------------------------------------------------

def _synthesize(text: str) -> bytes:
    cmd = [
        PIPER_BINARY,
        "--model", PIPER_MODEL,
        "--speaker", str(PIPER_SPEAKER),
        "--output-raw",
    ]
    result = subprocess.run(
        cmd,
        input=text.encode("utf-8"),
        capture_output=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Piper error: {result.stderr.decode()}")

    raw_pcm = result.stdout
    return _pcm_to_wav(raw_pcm)


def _pcm_to_wav(raw_pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(TTS_SAMPLE_RATE)
        wf.writeframes(raw_pcm)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Helpers — Agent HTTP
# ---------------------------------------------------------------------------

def _get_agent(agent_name: str) -> dict:
    agent = AGENTS.get(agent_name)
    if agent is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Unknown agent '{agent_name}'",
                "available": list(AGENTS.keys()),
            },
        )
    return agent


async def _call_agent(agent_name: str, query: str) -> dict:
    agent = _get_agent(agent_name)
    url = agent["url"]
    log.info("Calling agent '%s' at %s", agent_name, url)
    try:
        async with httpx.AsyncClient(timeout=AGENT_TIMEOUT) as client:
            resp = await client.post(url, json={"query": query})
            resp.raise_for_status()
            return resp.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail=f"Agent '{agent_name}' timed out after {AGENT_TIMEOUT}s")
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail=f"Agent '{agent_name}' is unreachable at {url}")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Agent '{agent_name}' returned {exc.response.status_code}")


def extract_answer_text(agent_result: Any) -> str:
    if not isinstance(agent_result, dict):
        return str(agent_result)

    for key in ("answer", "response", "output", "result", "content", "message", "text"):
        val = agent_result.get(key)
        if val is None:
            continue
        if isinstance(val, str):
            return val
        if isinstance(val, dict):
            for sub_key in ("text", "content", "answer", "message"):
                sub_val = val.get(sub_key)
                if isinstance(sub_val, str):
                    return sub_val
        return json.dumps(val, ensure_ascii=False)

    return json.dumps(agent_result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TTSRequest(BaseModel):
    text: str


class TextAgentRequest(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "stt_model_loaded": stt_model is not None,
        "stt_model_size": STT_MODEL_SIZE,
        "stt_device": STT_DEVICE,
        "piper_binary_exists": os.path.isfile(PIPER_BINARY),
        "piper_model_exists": os.path.isfile(PIPER_MODEL),
        "piper_model": PIPER_MODEL,
        "agents": list(AGENTS.keys()),
    }


@app.get("/agents")
async def list_agents():
    return {
        name: {"url": cfg["url"], "description": cfg.get("description", "")}
        for name, cfg in AGENTS.items()
    }


@app.post("/stt")
async def stt_endpoint(audio: UploadFile = File(...)):
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        text = await asyncio.get_event_loop().run_in_executor(None, _transcribe_bytes, data)
    except Exception as exc:
        log.error("STT error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    if not text:
        raise HTTPException(status_code=422, detail="No speech detected in audio")

    return {"transcription": text}


@app.post("/tts")
async def tts_endpoint(req: TTSRequest):
    try:
        wav_bytes = await asyncio.get_event_loop().run_in_executor(None, _synthesize, req.text)
    except Exception as exc:
        log.error("TTS error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/voice/{agent_name}")
async def voice_pipeline(agent_name: str, audio: UploadFile = File(...)):
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # STT
    loop = asyncio.get_event_loop()
    try:
        transcription = await loop.run_in_executor(None, _transcribe_bytes, data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"STT failed: {exc}")

    if not transcription:
        raise HTTPException(status_code=422, detail="No speech detected in audio")

    # Agent
    agent_result = await _call_agent(agent_name, transcription)
    answer_text = extract_answer_text(agent_result)
    log.info("Agent '%s' answer: %s", agent_name, answer_text[:200])

    # TTS
    try:
        wav_bytes = await loop.run_in_executor(None, _synthesize, answer_text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}")

    headers = {
        "X-Transcription": transcription[:500],
        "X-Agent-Response": answer_text[:500],
        "X-Agent-Name": agent_name,
    }
    return Response(content=wav_bytes, media_type="audio/wav", headers=headers)


@app.post("/voice/{agent_name}/json")
async def voice_pipeline_json(agent_name: str, audio: UploadFile = File(...)):
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")

    loop = asyncio.get_event_loop()

    try:
        transcription = await loop.run_in_executor(None, _transcribe_bytes, data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"STT failed: {exc}")

    if not transcription:
        raise HTTPException(status_code=422, detail="No speech detected in audio")

    agent_result = await _call_agent(agent_name, transcription)
    answer_text = extract_answer_text(agent_result)

    try:
        wav_bytes = await loop.run_in_executor(None, _synthesize, answer_text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}")

    audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")

    return {
        "transcription": transcription,
        "agent_name": agent_name,
        "agent_response": agent_result,
        "answer_text": answer_text,
        "audio_base64": audio_b64,
        "audio_content_type": "audio/wav",
    }


@app.post("/text/{agent_name}")
async def text_to_agent(agent_name: str, req: TextAgentRequest):
    agent_result = await _call_agent(agent_name, req.text)
    answer_text = extract_answer_text(agent_result)

    loop = asyncio.get_event_loop()
    try:
        wav_bytes = await loop.run_in_executor(None, _synthesize, answer_text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}")

    headers = {
        "X-Agent-Response": answer_text[:500],
        "X-Agent-Name": agent_name,
    }
    return Response(content=wav_bytes, media_type="audio/wav", headers=headers)


@app.websocket("/ws/voice/{agent_name}")
async def ws_voice(websocket: WebSocket, agent_name: str):
    # Validate agent exists before accepting
    if agent_name not in AGENTS:
        await websocket.close(code=4004)
        return

    await websocket.accept()
    log.info("WebSocket connected for agent '%s'", agent_name)

    audio_buffer = bytearray()

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                audio_buffer.extend(message["bytes"])

            elif "text" in message:
                if message["text"].strip().upper() == "END":
                    if not audio_buffer:
                        await websocket.send_text(json.dumps({"type": "error", "message": "No audio received"}))
                        continue

                    loop = asyncio.get_event_loop()

                    # STT
                    try:
                        audio_bytes = bytes(audio_buffer)
                        transcription = await loop.run_in_executor(None, _transcribe_bytes, audio_bytes)
                    except Exception as exc:
                        await websocket.send_text(json.dumps({"type": "error", "message": f"STT error: {exc}"}))
                        audio_buffer.clear()
                        continue

                    if not transcription:
                        await websocket.send_text(json.dumps({"type": "error", "message": "No speech detected"}))
                        audio_buffer.clear()
                        continue

                    await websocket.send_text(json.dumps({"type": "transcription", "text": transcription}))

                    # Agent
                    try:
                        agent_result = await _call_agent(agent_name, transcription)
                        answer_text = extract_answer_text(agent_result)
                    except HTTPException as exc:
                        await websocket.send_text(json.dumps({"type": "error", "message": str(exc.detail)}))
                        audio_buffer.clear()
                        continue

                    await websocket.send_text(json.dumps({
                        "type": "agent_response",
                        "text": answer_text,
                        "full_response": agent_result,
                    }))

                    # TTS
                    try:
                        wav_bytes = await loop.run_in_executor(None, _synthesize, answer_text)
                        await websocket.send_bytes(wav_bytes)
                    except Exception as exc:
                        await websocket.send_text(json.dumps({"type": "error", "message": f"TTS error: {exc}"}))

                    audio_buffer.clear()

    except WebSocketDisconnect:
        log.info("WebSocket disconnected for agent '%s'", agent_name)
