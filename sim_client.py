"""
Sierra Simulation Client
========================
Speaks the exact Twilio Media Streams WebSocket protocol that Sierra (via
Twilio) would use when calling your service.

Modes
-----
  (default) Interactive text mode – you type, Deepgram TTS synthesises real
            speech audio which is streamed as mulaw frames. The agent's
            Deepgram STT transcribes it just like a real phone call.
  --file    Stream a WAV file (mono, mulaw or PCM, 8 kHz)
  --mic     Stream live microphone audio (requires pyaudio)

Usage
-----
  python client/sim_client.py                   # interactive text-via-TTS mode
  python client/sim_client.py --file audio.wav
  python client/sim_client.py --mic
  python client/sim_client.py --url ws://localhost:8000/media-stream

Required env var (for default text mode):
  DEEPGRAM_API_KEY
"""

import argparse
import asyncio
import audioop
import base64
import json
import os
import sys
import uuid
import wave
from typing import Optional

import httpx
import websockets

from dotenv import load_dotenv
load_dotenv()

SERVER_URL         = "ws://localhost:8000/media-stream"
SAMPLE_RATE        = 8_000   # Twilio mulaw is always 8 kHz
CHUNK_MS           = 20      # 20 ms per frame (standard telephony)
CHUNK_BYTES        = SAMPLE_RATE * 2 * CHUNK_MS // 1000   # 160 bytes linear16

# Deepgram TTS – same voice as the agent so the sim sounds consistent
DG_TTS_MODEL       = os.getenv("DG_TTS_MODEL", "aura-2-thalia-en")
DEEPGRAM_API_KEY   = os.getenv("DEEPGRAM_API_KEY", "")

# Silence appended after speech to let Deepgram detect end-of-utterance
TRAILING_SILENCE_MS = 2500  # must exceed utterance_end_ms (1500ms) on the agent


# ── Colour helpers ─────────────────────────────────────────────────────────────

def _c(code, t): return f"\033[{code}m{t}\033[0m"
def cyan(t):    return _c(96, t)
def yellow(t):  return _c(93, t)
def green(t):   return _c(92, t)
def magenta(t): return _c(95, t)
def dim(t):     return _c(2, t)
def bold(t):    return _c(1, t)


# ── Audio helpers ──────────────────────────────────────────────────────────────

def pcm16_to_mulaw(pcm: bytes) -> bytes:
    return audioop.lin2ulaw(pcm, 2)

def silence_pcm16(duration_ms: int) -> bytes:
    return b"\x00\x00" * (SAMPLE_RATE * duration_ms // 1000)


# ── Deepgram TTS ───────────────────────────────────────────────────────────────

async def tts_to_pcm16_8k(text: str) -> bytes:
    """
    Call Deepgram TTS REST endpoint and return raw linear16 PCM at 8 kHz.
    This is the same call the agent makes server-side; here we use it on the
    client so that typed text becomes real speech audio before going over the wire.
    """
    if not DEEPGRAM_API_KEY:
        raise RuntimeError("DEEPGRAM_API_KEY not set – cannot synthesise speech for text mode.")

    url = (
        f"https://api.deepgram.com/v1/speak"
        f"?model={DG_TTS_MODEL}&encoding=linear16&sample_rate={SAMPLE_RATE}"
    )
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=30) as http:
        resp = await http.post(url, headers=headers, json={"text": text})
        resp.raise_for_status()
        return resp.content   # raw linear16 PCM at 8 kHz


# ── Twilio Media Streams message builders ──────────────────────────────────────

def make_connected_msg() -> str:
    return json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"})

def make_start_msg(stream_sid: str, call_sid: str) -> str:
    return json.dumps({
        "event":     "start",
        "streamSid": stream_sid,
        "start": {
            "callSid":          call_sid,
            "streamSid":        stream_sid,
            "accountSid":       "AC_SIMULATOR",
            "from":             "+15550001111",
            "to":               "+15559998888",
            "tracks":           ["inbound"],
            "customParameters": {"source": "sierra-simulation"},
        },
    })

def make_media_msg(stream_sid: str, mulaw_bytes: bytes) -> str:
    return json.dumps({
        "event":     "media",
        "streamSid": stream_sid,
        "media": {
            "track":     "inbound",
            "chunk":     "1",
            "timestamp": "0",
            "payload":   base64.b64encode(mulaw_bytes).decode("ascii"),
        },
    })

def make_stop_msg(stream_sid: str, call_sid: str) -> str:
    return json.dumps({
        "event":     "stop",
        "streamSid": stream_sid,
        "stop": {"callSid": call_sid},
    })


# ── Receive task ───────────────────────────────────────────────────────────────

async def receive_loop(ws):
    """Print incoming events and audio progress from the mock agent."""
    audio_chunks = 0
    async for raw in ws:
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue

        event = msg.get("event", "")
        if event == "media":
            audio_chunks += 1
            if audio_chunks % 10 == 1:
                print(dim(f"  ♪ receiving agent audio ({audio_chunks} chunks)"), end="\r")
        elif event == "clear":
            print(yellow("\n  [agent cleared audio buffer]"))
        else:
            print(dim(f"\n  [agent → {event}]"))


# ── Audio streaming helper ─────────────────────────────────────────────────────

async def stream_pcm16_as_mulaw(ws, stream_sid: str, pcm16: bytes):
    """Chop linear16 PCM into 20 ms mulaw frames and send at real-time pace."""
    for offset in range(0, len(pcm16), CHUNK_BYTES):
        chunk = pcm16[offset : offset + CHUNK_BYTES]
        mulaw = pcm16_to_mulaw(chunk)
        await ws.send(make_media_msg(stream_sid, mulaw))
        await asyncio.sleep(CHUNK_MS / 1000)


# ── Mode: text via Deepgram TTS ────────────────────────────────────────────────

async def text_mode(ws, stream_sid: str):
    """
    You type a message → Deepgram TTS converts it to real speech audio at 8 kHz
    → streamed as mulaw media frames → agent's Deepgram STT transcribes it →
    Gemini responds → TTS audio streams back.

    The full STT → LLM → TTS pipeline runs end-to-end, exactly as it will with
    a real Sierra phone call. The only difference is the audio originates from
    TTS rather than a microphone.
    """
    print(cyan("\n⌨️  Text mode (TTS) – type your message and press Enter."))
    print(dim(f"   Your text → Deepgram TTS ({DG_TTS_MODEL}) → mulaw audio → agent STT pipeline"))
    print(dim("   Blank line to hang up.\n"))

    loop = asyncio.get_event_loop()

    while True:
        text = await loop.run_in_executor(None, input, bold("You: "))
        if not text.strip():
            break

        print(dim("  🔊 synthesising speech…"), end="\r")
        try:
            pcm = await tts_to_pcm16_8k(text.strip())
        except httpx.HTTPStatusError as e:
            print(f"\n  ⚠️  TTS failed ({e.response.status_code}): {e.response.text}")
            continue

        # Speech audio
        audio_duration_ms = len(pcm) // (SAMPLE_RATE * 2 // 1000)
        print(dim(f"  📤 streaming {audio_duration_ms}ms of speech audio…"), end="\r")
        await stream_pcm16_as_mulaw(ws, stream_sid, pcm)

        # Trailing silence so Deepgram fires utterance_end
        await stream_pcm16_as_mulaw(ws, stream_sid, silence_pcm16(TRAILING_SILENCE_MS))

        print(dim("  ⏳ waiting for agent response…                        "), end="\r")


# ── Mode: WAV file ─────────────────────────────────────────────────────────────

async def wav_mode(ws, stream_sid: str, path: str):
    """Stream a mono WAV file (PCM or mulaw) at real-time pace."""
    print(cyan(f"\n📂 Streaming WAV: {path}"))
    with wave.open(path, "rb") as wf:
        encoding   = wf.getcomptype()
        sr         = wf.getframerate()
        n_channels = wf.getnchannels()
        sw         = wf.getsampwidth()

        if n_channels != 1:
            print("⚠️  WAV must be mono – aborting.")
            return

        chunk_frames = sr * CHUNK_MS // 1000
        while True:
            raw = wf.readframes(chunk_frames)
            if not raw:
                break

            if encoding == "ULAW":
                if sr != SAMPLE_RATE:
                    pcm = audioop.ulaw2lin(raw, 2)
                    pcm = audioop.ratecv(pcm, 2, 1, sr, SAMPLE_RATE, None)[0]
                    mulaw = audioop.lin2ulaw(pcm, 2)
                else:
                    mulaw = raw
            else:
                if sw != 2:
                    raw = audioop.lin2lin(raw, sw, 2)
                if sr != SAMPLE_RATE:
                    raw = audioop.ratecv(raw, 2, 1, sr, SAMPLE_RATE, None)[0]
                mulaw = pcm16_to_mulaw(raw)

            await ws.send(make_media_msg(stream_sid, mulaw))
            await asyncio.sleep(CHUNK_MS / 1000)

    print(dim("\n  [end of file – waiting for agent to finish…]"))
    await asyncio.sleep(5)


# ── Mode: microphone ───────────────────────────────────────────────────────────

async def mic_mode(ws, stream_sid: str):
    """Stream live microphone input (requires pyaudio)."""
    try:
        import pyaudio
    except ImportError:
        raise RuntimeError(
            "pyaudio is not installed.\n"
            "  Install it with: pip install pyaudio\n"
            "  On macOS you may first need: brew install portaudio"
        )

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=SAMPLE_RATE * CHUNK_MS // 1000,
    )
    print(cyan("\n🎙  Microphone open – speak now. Ctrl+C to hang up."))
    try:
        while True:
            data  = stream.read(SAMPLE_RATE * CHUNK_MS // 1000, exception_on_overflow=False)
            mulaw = pcm16_to_mulaw(data)
            await ws.send(make_media_msg(stream_sid, mulaw))
            await asyncio.sleep(0)
    except asyncio.CancelledError:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


# ── Main ───────────────────────────────────────────────────────────────────────

async def run(args):
    stream_sid = f"MZ{uuid.uuid4().hex[:30]}"
    call_sid   = f"CA{uuid.uuid4().hex[:30]}"

    url = args.url
    print(cyan(f"\nConnecting to {url} …"))

    async with websockets.connect(url, max_size=10 * 1024 * 1024) as ws:
        print(green("✓ Connected"))

        await ws.send(make_connected_msg())
        await ws.send(make_start_msg(stream_sid, call_sid))
        print(dim(f"  streamSid={stream_sid}"))

        recv_task = asyncio.create_task(receive_loop(ws))

        try:
            if args.file:
                await wav_mode(ws, stream_sid, args.file)
            elif args.mic:
                await mic_mode(ws, stream_sid)
            else:
                await text_mode(ws, stream_sid)
        except (KeyboardInterrupt, RuntimeError) as e:
            if isinstance(e, RuntimeError):
                print(f"\n⚠️  {e}")
        finally:
            # Send stop frame so the agent can clean up gracefully, then
            # cancel the receive task. Guard against an already-closed socket.
            try:
                await ws.send(make_stop_msg(stream_sid, call_sid))
                await asyncio.sleep(0.3)
            except Exception:
                pass
            recv_task.cancel()
            try:
                await recv_task
            except (asyncio.CancelledError, Exception):
                pass

    print(green("\n✓ Call ended"))


def main():
    parser = argparse.ArgumentParser(description="Sierra Simulation Client")
    parser.add_argument("--url", default=SERVER_URL, help="Agent WebSocket URL")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mic",  action="store_true", help="Live microphone")
    group.add_argument("--file", metavar="WAV",       help="Stream a WAV file")
    args = parser.parse_args()

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()