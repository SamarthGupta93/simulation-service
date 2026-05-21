"""
Sierra Mock Voice Agent
=======================
Implements the Twilio Media Streams WebSocket protocol so your simulation
client can develop against it identically to how it will work against a real
Sierra phone agent.

Protocol (Twilio Media Streams bidirectional):
  Caller → Server : connected, start, media (mulaw/8kHz base64), stop
  Server → Caller : media (mulaw/8kHz base64), clear

Pipeline per call:
  mulaw audio → decode → 16-kHz PCM → Deepgram STT (listen.v1)
  transcript   → Gemini LLM
  LLM reply    → Deepgram TTS (speak.v1.audio.generate) → mulaw → caller
"""

import asyncio
import audioop
import base64
import json
import logging
import os
import uuid
from typing import Optional

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.listen.v1.types import ListenV1Results, ListenV1UtteranceEnd
from deepgram.speak.v1.types import SpeakV1Text
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from google import genai
from google.genai import types as genai_types

from dotenv import load_dotenv
load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("sierra-mock")

# ── Config ────────────────────────────────────────────────────────────────────
DEEPGRAM_API_KEY = os.environ["DEEPGRAM_API_KEY"]
GEMINI_API_KEY   = os.environ["GEMINI_API_KEY"]

GEMINI_MODEL     = os.getenv("GEMINI_MODEL",   "gemini-2.5-flash-lite")
DG_STT_MODEL     = os.getenv("DG_STT_MODEL",   "nova-3")
DG_TTS_MODEL     = os.getenv("DG_TTS_MODEL",   "aura-2-thalia-en")

# Audio constants
TWILIO_SAMPLE_RATE   = 8_000   # mulaw, telephone quality
DEEPGRAM_SAMPLE_RATE = 16_000  # Deepgram STT wants 16 kHz

SYSTEM_PROMPT = (
    "You are a helpful and friendly customer service voice agent. "
    "Keep answers to two or three sentences because they will be read aloud. "
    "Do not use bullet points, markdown, or special characters."
)

# ── Shared clients ────────────────────────────────────────────────────────────
dg_client     = AsyncDeepgramClient(api_key=DEEPGRAM_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="Sierra Mock Voice Agent")


# ── Audio helpers ─────────────────────────────────────────────────────────────

def mulaw_to_pcm16(mulaw: bytes) -> bytes:
    return audioop.ulaw2lin(mulaw, 2)

def pcm16_to_mulaw(pcm: bytes) -> bytes:
    return audioop.lin2ulaw(pcm, 2)

def upsample_8k_to_16k(pcm: bytes) -> bytes:
    return audioop.ratecv(pcm, 2, 1, TWILIO_SAMPLE_RATE, DEEPGRAM_SAMPLE_RATE, None)[0]

def downsample_16k_to_8k(pcm: bytes) -> bytes:
    return audioop.ratecv(pcm, 2, 1, DEEPGRAM_SAMPLE_RATE, TWILIO_SAMPLE_RATE, None)[0]


# ── Twilio message builders ───────────────────────────────────────────────────

def media_msg(stream_sid: str, mulaw: bytes) -> str:
    return json.dumps({
        "event":     "media",
        "streamSid": stream_sid,
        "media":     {"payload": base64.b64encode(mulaw).decode()},
    })

def clear_msg(stream_sid: str) -> str:
    return json.dumps({"event": "clear", "streamSid": stream_sid})


# ── Per-call session ──────────────────────────────────────────────────────────

class CallSession:
    def __init__(self, ws: WebSocket, stream_sid: str):
        self.ws         = ws
        self.stream_sid = stream_sid
        self.id         = str(uuid.uuid4())[:8]
        self.history: list[genai_types.Content] = []
        self.transcript_q: asyncio.Queue[str]   = asyncio.Queue()
        self._speaking        = False
        self._stt_conn        = None
        self._stt_ctx         = None
        self._keepalive_task  = None   # sends keepalive while agent is speaking
        self._partial_transcript: list[str] = []  # accumulates finals until UtteranceEnd

    # ── STT ───────────────────────────────────────────────────────────────────

    def _stt_connect_kwargs(self) -> dict:
        return dict(
            model            = DG_STT_MODEL,
            encoding         = "linear16",
            sample_rate      = DEEPGRAM_SAMPLE_RATE,
            channels         = 1,
            punctuate        = True,
            interim_results  = True,
            utterance_end_ms = "1500",  # ms of silence before UtteranceEnd fires
            vad_events       = True,    # enables SpeechStarted / UtteranceEnd events
            smart_format     = True,
            language         = "en-US",
        )

    async def _open_stt(self):
        """Open a fresh Deepgram STT WebSocket and register callbacks."""
        session = self

        async def on_message(message):
            if isinstance(message, ListenV1Results):
                alt  = message.channel.alternatives[0]
                text = alt.transcript.strip()
                if not text:
                    return
                log.info(f"[{session.id}] STT {'FINAL' if message.is_final else 'interim'}: {text}")
                if message.is_final:
                    # Accumulate finals – do NOT dispatch yet; wait for UtteranceEnd
                    session._partial_transcript.append(text)

            elif isinstance(message, ListenV1UtteranceEnd):
                # Speaker has stopped – join all accumulated finals into one turn
                if session._speaking:
                    # Agent is mid-response; discard – barge-in not supported yet
                    session._partial_transcript.clear()
                    return
                full_text = " ".join(session._partial_transcript).strip()
                session._partial_transcript.clear()
                if full_text:
                    log.info(f"[{session.id}] UtteranceEnd → dispatch: {full_text!r}")
                    await session.transcript_q.put(full_text)

        self._stt_ctx  = dg_client.listen.v1.connect(**self._stt_connect_kwargs())
        self._stt_conn = await self._stt_ctx.__aenter__()
        self._stt_conn.on(EventType.MESSAGE, on_message)
        asyncio.create_task(self._stt_conn.start_listening())
        log.info(f"[{self.id}] Deepgram STT connected")

    async def start_stt(self):
        await self._open_stt()
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def _keepalive_loop(self):
        """
        While the agent is speaking (no audio coming from the caller),
        send Deepgram a keepalive every 8 seconds so the STT connection
        does not time out (Deepgram closes after ~10 s of no data).
        """
        try:
            while True:
                await asyncio.sleep(8)
                if self._stt_conn is not None and self._speaking:
                    try:
                        await self._stt_conn.send_keep_alive()
                        log.debug(f"[{self.id}] STT keepalive sent")
                    except Exception:
                        # Connection dropped – try to reconnect
                        log.warning(f"[{self.id}] STT keepalive failed – reconnecting")
                        await self._reconnect_stt()
        except asyncio.CancelledError:
            pass

    async def _reconnect_stt(self):
        """Close the dead STT connection and open a new one."""
        try:
            await self._stt_ctx.__aexit__(None, None, None)
        except Exception:
            pass
        self._stt_conn = None
        self._stt_ctx  = None
        try:
            await self._open_stt()
        except Exception as e:
            log.error(f"[{self.id}] STT reconnect failed: {e}")

    async def feed_audio(self, payload_b64: str):
        """Decode Twilio mulaw frame and push PCM to Deepgram."""
        if self._stt_conn is None:
            return
        mulaw  = base64.b64decode(payload_b64)
        pcm8k  = mulaw_to_pcm16(mulaw)
        pcm16k = upsample_8k_to_16k(pcm8k)
        try:
            await self._stt_conn.send_media(pcm16k)
        except Exception:
            # Connection dropped mid-call – reconnect and drop this frame
            log.warning(f"[{self.id}] STT send failed – reconnecting")
            await self._reconnect_stt()

    async def stop_stt(self):
        if self._keepalive_task:
            self._keepalive_task.cancel()
            self._keepalive_task = None
        if self._stt_conn:
            try:
                await self._stt_conn.send_close_stream()
                await self._stt_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            self._stt_conn = None

    # ── Response loop ─────────────────────────────────────────────────────────

    async def run(self):
        """Greet caller, then loop: transcript → Gemini → TTS → audio out."""
        await self._respond("Hello! How can I help you today?")

        while True:
            try:
                transcript = await asyncio.wait_for(self.transcript_q.get(), timeout=300)
            except asyncio.TimeoutError:
                log.info(f"[{self.id}] Timeout – ending session")
                break
            log.info(f"[{self.id}] User: {transcript}")
            await self._respond(transcript)

    async def _respond(self, user_text: str):
        self._speaking = True
        try:
            self.history.append(genai_types.Content(
                role="user", parts=[genai_types.Part(text=user_text)]
            ))

            # Gemini
            resp = await gemini_client.aio.models.generate_content(
                model    = GEMINI_MODEL,
                contents = self.history,
                config   = genai_types.GenerateContentConfig(
                    system_instruction = SYSTEM_PROMPT,
                    max_output_tokens  = 256,
                    temperature        = 0.7,
                ),
            )
            reply = resp.text.strip()
            log.info(f"[{self.id}] Agent: {reply}")

            self.history.append(genai_types.Content(
                role="model", parts=[genai_types.Part(text=reply)]
            ))

            # TTS → stream audio back to caller
            await self._tts_and_send(reply)

        except Exception as e:
            log.exception(f"[{self.id}] Error in respond cycle: {e}")
        finally:
            self._speaking = False

    async def _tts_and_send(self, text: str):
        """
        Use Deepgram speak.v1.audio.generate (streaming REST) to get linear16
        PCM at 8 kHz, convert to mulaw, and stream to the caller in 20 ms chunks.
        """
        pcm_chunks: list[bytes] = []

        async for chunk in dg_client.speak.v1.audio.generate(
            text        = text,
            model       = DG_TTS_MODEL,
            encoding    = "linear16",
            sample_rate = TWILIO_SAMPLE_RATE,   # request 8 kHz directly
        ):
            pcm_chunks.append(chunk)

        pcm_bytes   = b"".join(pcm_chunks)
        chunk_size  = TWILIO_SAMPLE_RATE * 2 * 20 // 1000   # 160 bytes = 20 ms

        for offset in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[offset : offset + chunk_size]
            mulaw = pcm16_to_mulaw(chunk)
            await self.ws.send_text(media_msg(self.stream_sid, mulaw))
            await asyncio.sleep(20 / 1000)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/incoming-call")
async def incoming_call(request: Request):
    """
    TwiML webhook. In real Sierra: Twilio POSTs here when the call connects.
    Returns TwiML that instructs Twilio to open a bidirectional Media Stream.
    """
    host = request.headers.get("host", "localhost:8000")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://{host}/media-stream" />
    </Connect>
</Response>"""
    return PlainTextResponse(twiml, media_type="text/xml")


@app.websocket("/media-stream")
async def media_stream(ws: WebSocket):
    """
    Bidirectional Twilio Media Streams WebSocket.
    Receives: connected, start, media, stop
    Sends:    media (mulaw audio), clear
    """
    await ws.accept()
    log.info("WebSocket accepted")

    session: Optional[CallSession] = None
    run_task: Optional[asyncio.Task] = None

    try:
        async for raw in ws.iter_text():
            msg   = json.loads(raw)
            event = msg.get("event")

            if event == "connected":
                log.info(f"connected: {msg.get('protocol')} {msg.get('version')}")

            elif event == "start":
                sid = msg.get("streamSid", "")
                log.info(f"start: streamSid={sid} callSid={msg['start'].get('callSid')}")
                session  = CallSession(ws=ws, stream_sid=sid)
                await session.start_stt()
                run_task = asyncio.create_task(session.run())

            elif event == "media":
                if session:
                    await session.feed_audio(msg["media"]["payload"])

            elif event == "stop":
                log.info("stop received – call ended")
                break

    except WebSocketDisconnect:
        log.info("WebSocket disconnected")
    except Exception as e:
        log.exception(f"Unexpected error: {e}")
    finally:
        if run_task:
            run_task.cancel()
        if session:
            await session.stop_stt()
        log.info("Session cleaned up")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "agent:app",
        host  = os.getenv("HOST", "0.0.0.0"),
        port  = int(os.getenv("PORT", "8000")),
        reload = False,
    )