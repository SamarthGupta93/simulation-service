"""
Simulated User
==============
An autonomous AI-powered caller that converses with the Sierra mock voice agent.

The simulated user has:
  - A persona  (who they are, their emotional state, communication style)
  - A goal     (what they want to achieve by the end of the call)
  - A memory   (full conversation history fed back to Gemini each turn)

Pipeline per turn:
  Agent audio (mulaw frames) → Deepgram STT → agent transcript
  Agent transcript + history → Gemini → user's next utterance
  User utterance → Deepgram TTS → mulaw frames → agent

The call ends when Gemini decides the goal is achieved or after MAX_TURNS.

Usage
-----
  python client/sim_user.py                        # default billing persona
  python client/sim_user.py --persona custom.json  # load persona from file
  python client/sim_user.py --url ws://host/media-stream
  python client/sim_user.py --turns 10             # max conversation turns

Persona JSON format
-------------------
{
  "name": "Alex Johnson",
  "background": "...",
  "goal": "...",
  "style": "...",
  "emotional_state": "..."
}
"""

import argparse
import asyncio
import audioop
import base64
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Optional

import datetime
import httpx
import websockets
from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.listen.v1.types import ListenV1Results, ListenV1UtteranceEnd
from google import genai
from google.genai import types as genai_types

from dotenv import load_dotenv
load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("sim-user")

# ── Config ─────────────────────────────────────────────────────────────────────
SERVER_URL          = "ws://localhost:8000/media-stream"
DEEPGRAM_API_KEY    = os.environ["DEEPGRAM_API_KEY"]
GEMINI_API_KEY      = os.environ["GEMINI_API_KEY"]

GEMINI_MODEL        = os.getenv("GEMINI_MODEL",       "gemini-2.0-flash")
DG_STT_MODEL        = os.getenv("DG_STT_MODEL",       "nova-3")
DG_USER_TTS_MODEL   = os.getenv("DG_USER_TTS_MODEL",  "aura-2-orpheus-en")  # different voice from agent

SAMPLE_RATE         = 8_000
DEEPGRAM_STT_RATE   = 16_000
CHUNK_MS            = 20
CHUNK_BYTES         = SAMPLE_RATE * 2 * CHUNK_MS // 1000   # 160 bytes
TRAILING_SILENCE_MS = 2500
MAX_TURNS           = int(os.getenv("MAX_TURNS", "5"))

# ── Default persona ────────────────────────────────────────────────────────────
DEFAULT_PERSONA = {
    "persona": (
        "Sarah Mitchell, a 34-year-old marketing manager. "
        "Polite but firm. Mildly frustrated but not rude. "
        "Asks follow-up questions if the agent's answer is vague. "
        "Does not accept vague answers without pushing for specifics or escalation. "
        "Emotional state: mildly annoyed, wants a quick resolution."
    ),
    "goal": (
        "Get a clear explanation of what the $47.99 charge is for, and if it "
        "is an error, have it reversed and receive confirmation. "
        "Consider the goal achieved once the agent has either explained the "
        "charge satisfactorily OR confirmed a reversal or credit."
    ),
    "context": (
        "Sarah has been a customer for 3 years. She received an unexpected "
        "charge of $47.99 on her latest bill that she does not recognise. "
        "She checked her account online but could not find an explanation."
    ),
    "guidelines": (
        "Keep each utterance to 1-3 sentences – this is a phone call. "
        "Stay in character at all times. "
        "Do not use markdown, bullet points, or special characters. "
        "Respond naturally to what the agent just said. "
        "If the agent asks a clarifying question, answer it. "
        "If the agent is vague, push for specifics. "
        "Decline hold requests politely but firmly if offered more than once."
    ),
    "max_turns": 5,
}

# ── Colour helpers ─────────────────────────────────────────────────────────────
def _c(code, t): return f"\033[{code}m{t}\033[0m"
def cyan(t):    return _c(96, t)
def yellow(t):  return _c(93, t)
def green(t):   return _c(92, t)
def magenta(t): return _c(95, t)
def dim(t):     return _c(2, t)
def bold(t):    return _c(1, t)
def red(t):     return _c(91, t)

# ── Audio helpers ──────────────────────────────────────────────────────────────

def mulaw_to_pcm16(mulaw: bytes) -> bytes:
    return audioop.ulaw2lin(mulaw, 2)

def pcm16_to_mulaw(pcm: bytes) -> bytes:
    return audioop.lin2ulaw(pcm, 2)

def upsample_8k_to_16k(pcm: bytes) -> bytes:
    return audioop.ratecv(pcm, 2, 1, SAMPLE_RATE, DEEPGRAM_STT_RATE, None)[0]

def silence_pcm16(duration_ms: int) -> bytes:
    return b"\x00\x00" * (SAMPLE_RATE * duration_ms // 1000)

# ── TTS ────────────────────────────────────────────────────────────────────────

async def tts_to_pcm16_8k(text: str, model: str = DG_USER_TTS_MODEL) -> bytes:
    """Synthesise speech via Deepgram TTS REST → raw linear16 PCM at 8 kHz."""
    url = (
        f"https://api.deepgram.com/v1/speak"
        f"?model={model}&encoding=linear16&sample_rate={SAMPLE_RATE}"
    )
    async with httpx.AsyncClient(timeout=30) as http:
        resp = await http.post(
            url,
            headers={
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"text": text},
        )
        resp.raise_for_status()
        return resp.content

# ── Twilio protocol builders ───────────────────────────────────────────────────

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
            "customParameters": {"source": "sierra-sim-user"},
        },
    })

def make_media_msg(stream_sid: str, mulaw_bytes: bytes) -> str:
    return json.dumps({
        "event":     "media",
        "streamSid": stream_sid,
        "media": {
            "track":   "inbound",
            "chunk":   "1",
            "payload": base64.b64encode(mulaw_bytes).decode("ascii"),
        },
    })

def make_stop_msg(stream_sid: str, call_sid: str) -> str:
    return json.dumps({
        "event": "stop", "streamSid": stream_sid,
        "stop":  {"callSid": call_sid},
    })

# ── Gemini user brain ──────────────────────────────────────────────────────────

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "utterance":      {"type": "string"},
        "goal_achieved":  {"type": "boolean"},
        "reasoning":      {"type": "string"},
    },
    "required": ["utterance", "goal_achieved", "reasoning"],
}

def build_system_prompt(persona: dict) -> str:
    return f"""You are roleplaying as a customer calling a customer service voice agent.

PERSONA
-------
{persona['persona']}

GOAL
----
{persona['goal']}

CONTEXT
-------
{persona['context']}

GUIDELINES
----------
{persona['guidelines']}
- Decide after each agent response whether your goal has been achieved.

OUTPUT FORMAT (strict JSON, no other text)
------------------------------------------
{{
  "utterance":     "<what you say next as the customer>",
  "goal_achieved": <true | false>,
  "reasoning":     "<brief internal note on why goal is or is not yet achieved>"
}}"""


async def generate_user_response(
    gemini_client: genai.Client,
    persona: dict,
    history: list[genai_types.Content],
    agent_utterance: str,
) -> tuple[str, bool]:
    """
    Ask Gemini to generate the user's next utterance given the agent's response.
    Returns (utterance_text, goal_achieved).
    """
    history.append(genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=f"[Agent said]: {agent_utterance}")]
    ))

    resp = await gemini_client.aio.models.generate_content(
        model    = GEMINI_MODEL,
        contents = history,
        config   = genai_types.GenerateContentConfig(
            system_instruction = build_system_prompt(persona),
            max_output_tokens  = 200,
            temperature        = 0.8,
            response_mime_type = "application/json",
            response_schema    = DECISION_SCHEMA,
        ),
    )

    try:
        data = json.loads(resp.text)
        utterance     = data.get("utterance", "").strip()
        goal_achieved = bool(data.get("goal_achieved", False))
        reasoning     = data.get("reasoning", "")
    except (json.JSONDecodeError, AttributeError):
        # Fallback if Gemini returns plain text
        utterance     = resp.text.strip()
        goal_achieved = False
        reasoning     = "JSON parse failed"

    history.append(genai_types.Content(
        role="model",
        parts=[genai_types.Part(text=utterance)]
    ))

    return utterance, goal_achieved, reasoning


# ── Agent audio listener (STT on inbound agent audio) ─────────────────────────

@dataclass
class AgentListener:
    """
    Receives mulaw audio frames from the agent and transcribes them via
    Deepgram STT, accumulating finals until UtteranceEnd fires.
    """
    dg_client:         AsyncDeepgramClient
    agent_speech_q:    asyncio.Queue = field(default_factory=asyncio.Queue)
    _stt_conn:         object = field(default=None, init=False)
    _stt_ctx:          object = field(default=None, init=False)
    _partial:          list   = field(default_factory=list, init=False)
    _keepalive_task:   object = field(default=None, init=False)

    async def start(self):
        listener = self
        # Dispatch once audio frames stop arriving for this long.
        # Tied to media frame gaps (true end-of-TTS), not is_final gaps.
        AUDIO_GAP_SEC = 1.0

        async def dispatch_after_silence():
            """Poll until no audio frame has arrived for AUDIO_GAP_SEC, then dispatch."""
            while True:
                await asyncio.sleep(0.1)
                if not listener._partial:
                    continue
                elapsed = asyncio.get_event_loop().time() - listener._last_audio_at
                if elapsed >= AUDIO_GAP_SEC:
                    full = " ".join(listener._partial).strip()
                    listener._partial.clear()
                    if full:
                        log.info(f"Agent STT: {full!r}")
                        await listener.agent_speech_q.put(full)

        listener._last_audio_at: float = 0.0

        async def on_message(message):
            if isinstance(message, ListenV1Results):
                alt  = message.channel.alternatives[0]
                text = alt.transcript.strip()
                if text and message.is_final:
                    listener._partial.append(text)

        self._stt_ctx  = self.dg_client.listen.v1.connect(
            model            = DG_STT_MODEL,
            encoding         = "linear16",
            sample_rate      = DEEPGRAM_STT_RATE,
            channels         = 1,
            punctuate        = True,
            interim_results  = True,
            smart_format     = True,
            language         = "en-US",
        )
        self._stt_conn = await self._stt_ctx.__aenter__()
        self._stt_conn.on(EventType.MESSAGE, on_message)
        asyncio.create_task(self._stt_conn.start_listening())
        asyncio.create_task(dispatch_after_silence())
        self._keepalive_task = asyncio.create_task(self._keepalive())
        log.info("AgentListener STT connected")

    async def feed(self, mulaw_b64: str):
        """Decode agent mulaw audio, stamp arrival time, and push to Deepgram."""
        if self._stt_conn is None:
            return
        # Update the audio-arrival timestamp so dispatch_after_silence waits
        # until the full TTS stream has finished before dispatching.
        self._last_audio_at = asyncio.get_event_loop().time()
        mulaw  = base64.b64decode(mulaw_b64)
        pcm8k  = mulaw_to_pcm16(mulaw)
        pcm16k = upsample_8k_to_16k(pcm8k)
        await self.feed_pcm(pcm16k)

    async def feed_pcm(self, pcm16k: bytes):
        """Push raw linear16 PCM at 16 kHz directly to Deepgram (used by silence injector)."""
        if self._stt_conn is None:
            return
        try:
            await self._stt_conn.send_media(pcm16k)
        except Exception:
            log.warning("AgentListener STT send failed – reconnecting")
            await self._reconnect()

    async def _keepalive(self):
        try:
            while True:
                await asyncio.sleep(8)
                if self._stt_conn:
                    try:
                        await self._stt_conn.send_keep_alive()
                    except Exception:
                        await self._reconnect()
        except asyncio.CancelledError:
            pass

    async def _reconnect(self):
        try:
            await self._stt_ctx.__aexit__(None, None, None)
        except Exception:
            pass
        self._stt_conn = None
        try:
            await self.start()
        except Exception as e:
            log.error(f"AgentListener reconnect failed: {e}")

    async def stop(self):
        if self._keepalive_task:
            self._keepalive_task.cancel()
        if self._stt_conn:
            try:
                await self._stt_conn.send_close_stream()
                await self._stt_ctx.__aexit__(None, None, None)
            except Exception:
                pass


# ── Main simulation loop ───────────────────────────────────────────────────────

def _save_transcript(transcript: list[dict], persona: dict, goal_achieved: bool):
    """Save the conversation transcript to a JSON file in ./transcripts/."""
    import os
    os.makedirs("transcripts", exist_ok=True)
    ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"transcripts/transcript_{ts}.json"
    payload   = {
        "timestamp":     datetime.datetime.now().isoformat(),
        "goal_achieved": goal_achieved,
        "turns":         len([t for t in transcript if t["role"] == "user"]),
        "persona":       persona,
        "transcript":    transcript,
    }
    with open(filename, "w") as f:
        json.dump(payload, f, indent=2)
    print(green(f"\n💾 Transcript saved → {filename}"))


async def simulate(persona: dict, url: str, max_turns: int):
    # Persona-level max_turns overrides the CLI argument
    max_turns = int(persona.get("max_turns", max_turns))
    stream_sid = f"MZ{uuid.uuid4().hex[:30]}"
    call_sid   = f"CA{uuid.uuid4().hex[:30]}"

    dg_client     = AsyncDeepgramClient(api_key=DEEPGRAM_API_KEY)
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    listener = AgentListener(dg_client=dg_client)
    history: list[genai_types.Content] = []
    transcript: list[dict] = []  # accumulated turns for final save

    print(cyan(f"\n{'━'*60}"))
    print(bold(f"  Simulated user: {persona['persona'][:60]}…"))
    print(dim(f"  Goal: {persona['goal'][:80]}…" if len(persona['goal']) > 80 else f"  Goal: {persona['goal']}"))
    print(cyan(f"{'━'*60}\n"))
    print(cyan(f"Connecting to {url} …"))

    async with websockets.connect(url, max_size=10 * 1024 * 1024) as ws:
        print(green("✓ Connected\n"))

        # ── Handshake ─────────────────────────────────────────────────────────
        await ws.send(make_connected_msg())
        await ws.send(make_start_msg(stream_sid, call_sid))
        await listener.start()

        # ── Receive task: route agent audio to AgentListener ──────────────────
        # After agent audio stops arriving, we inject trailing silence into the
        # AgentListener so Deepgram can measure the required utterance_end_ms
        # gap and fire UtteranceEnd (it only counts silence in received audio).
        async def recv_loop():
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                event = msg.get("event", "")
                if event == "media":
                    await listener.feed(msg["media"]["payload"])
        recv_task = asyncio.create_task(recv_loop())

        try:
            turn = 0
            goal_achieved = False

            # ── User speaks first ──────────────────────────────────────────────
            # Generate and send the opening line before waiting for the agent.
            # Pass an empty opener so Gemini produces a natural call-opening line.
            log.info("Generating opening user utterance…")
            opening, _, _ = await generate_user_response(
                gemini_client, persona, history,
                agent_utterance="[The call just connected. The line is open. Start the conversation.]"
            )
            print(f"{magenta(persona['persona'][:40] + ':')} {opening}")
            transcript.append({"role": "user", "text": opening, "turn": 0, "reasoning": "opening line", "goal_achieved": False})
            pcm = await tts_to_pcm16_8k(opening)
            await _stream_audio(ws, stream_sid, pcm)
            await _stream_audio(ws, stream_sid, silence_pcm16(TRAILING_SILENCE_MS))

            while turn < max_turns and not goal_achieved:
                turn += 1
                log.info(f"Turn {turn}/{max_turns} – waiting for agent…")

                # Wait for the agent to finish speaking (UtteranceEnd from STT)
                try:
                    agent_text = await asyncio.wait_for(
                        listener.agent_speech_q.get(), timeout=30
                    )
                except asyncio.TimeoutError:
                    log.warning("Timed out waiting for agent – ending call")
                    break

                print(f"\n{yellow('Agent:')} {agent_text}")
                transcript.append({"role": "agent", "text": agent_text, "turn": turn})

                # Generate user's response via Gemini
                utterance, goal_achieved, reasoning = await generate_user_response(
                    gemini_client, persona, history, agent_text
                )

                print(f"{magenta(persona['persona'][:40] + ':')} {utterance}")
                print(dim(f"  [internal: {reasoning}]"))
                transcript.append({"role": "user", "text": utterance, "turn": turn, "reasoning": reasoning, "goal_achieved": goal_achieved})

                if goal_achieved:
                    print(green(f"\n✓ Goal achieved on turn {turn}"))
                    # Still speak the final utterance before hanging up
                    pcm = await tts_to_pcm16_8k(utterance)
                    await _stream_audio(ws, stream_sid, pcm)
                    await _stream_audio(ws, stream_sid, silence_pcm16(TRAILING_SILENCE_MS))
                    await asyncio.sleep(1)
                    break

                # Synthesise and stream user speech
                print(dim("  🔊 synthesising…"), end="\r")
                pcm = await tts_to_pcm16_8k(utterance)
                audio_ms = len(pcm) // (SAMPLE_RATE * 2 // 1000)
                print(dim(f"  📤 speaking ({audio_ms}ms)…"), end="\r")
                await _stream_audio(ws, stream_sid, pcm)
                await _stream_audio(ws, stream_sid, silence_pcm16(TRAILING_SILENCE_MS))

            if not goal_achieved and turn >= max_turns:
                print(red(f"\n✗ Max turns ({max_turns}) reached without achieving goal"))

        except KeyboardInterrupt:
            pass
        finally:
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
            await listener.stop()
            _save_transcript(transcript, persona, goal_achieved)

    print(cyan(f"\n{'━'*60}"))
    print(bold("  Call ended"))
    print(cyan(f"{'━'*60}\n"))


async def _stream_audio(ws, stream_sid: str, pcm16: bytes):
    """Stream linear16 PCM as mulaw media frames at real-time pace."""
    for offset in range(0, len(pcm16), CHUNK_BYTES):
        chunk = pcm16[offset : offset + CHUNK_BYTES]
        mulaw = pcm16_to_mulaw(chunk)
        await ws.send(make_media_msg(stream_sid, mulaw))
        await asyncio.sleep(CHUNK_MS / 1000)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Simulated User")
    parser.add_argument("--url",     default=SERVER_URL,  help="Agent WebSocket URL")
    parser.add_argument("--turns",   type=int, default=MAX_TURNS, help="Max conversation turns")
    parser.add_argument("--persona", metavar="JSON_FILE",
                        help="Path to persona JSON file (see module docstring for format)")
    args = parser.parse_args()

    if args.persona:
        with open(args.persona) as f:
            persona = json.load(f)
    else:
        persona = DEFAULT_PERSONA

    try:
        asyncio.run(simulate(persona, args.url, args.turns))
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()