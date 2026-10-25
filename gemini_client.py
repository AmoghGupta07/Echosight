"""
Step 3a: Gemini client.

  GeminiAssistant.ask_stream(query, jpegs) yields the answer one sentence at a time,
  so the first sentence can be spoken while the rest is still being generated.

Setup:
  pip install google-genai
  export GEMINI_API_KEY="your-key"        # put this line in ~/.bashrc too

Test (no voice):
  python gemini_client.py --models         # list models your key can use
  python gemini_client.py                  # camera frames -> Gemini -> printed sentences
Change models without editing code (first = preferred, rest = fallbacks):
  export GEMINI_MODEL="gemini-3.5-flash,gemini-3.1-flash-lite,gemini-2.5-flash"
"""
import os
import re
import time

from google import genai
from google.genai import errors, types

# Comma-separated list, tried in order. If one is overloaded (503) or rate-limited (429),
# the next one is used automatically, so a busy model never leaves the user in silence.
MODELS = [m.strip() for m in os.environ.get(
    "GEMINI_MODEL", "gemini-3.5-flash,gemini-3.1-flash-lite,gemini-2.5-flash").split(",") if m.strip()]
RETRY_CODES = (404, 429, 500, 503, 504)
TIMEOUT_MS = 20_000

SYSTEM_PROMPT = """You are the voice of smart glasses worn by a blind user. Your replies are spoken aloud through earphones.

Images, when present, are the latest frames from a camera on the user's glasses, oldest first. They show what is in front of the user. Left and right always mean the user's left and right.

How to answer:
- Speak naturally in short sentences. Plain text only: no lists, markdown, asterisks or emoji.
- Be brief: usually one to three sentences. Only go longer when asked to read something or describe in detail.
- Never say "image", "photo" or "picture". Say "I can see" or "in front of you".
- Mention anything dangerous first (vehicles, stairs, drops, obstacles in the path, hot or sharp objects).
- Give locations as left, right, ahead, or clock positions (for example "at your two o'clock"), with approximate distance in steps or metres.
- If it is too dark, blurry or partly cut off to be sure, say so plainly and tell the user what to do ("hold the paper a little further back", "turn your head slowly to the right"). Never guess when it could matter.
- Never say it is safe to cross a road or step forward. You may describe what you see, but the user decides.
- When reading labels, signs or documents, read the exact text. For medicine, read the label only and give no dosing advice.
- If the user asks something that does not need the camera, just answer it."""

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_MARKDOWN = re.compile(r"[*_#`>]")


def _clean(text: str) -> str:
    return _MARKDOWN.sub("", text).strip()


def _thinking_config(model: str):
    """
    Minimise silent 'thinking' time, which otherwise adds 10+ seconds to every answer.
    Gemini 3.x is controlled with a level; Gemini 2.5 with a token budget (0 = off).
    Mixing them up returns an error, so the setting must match the model.
    """
    m = model.lower().replace("models/", "")
    try:
        if m.startswith("gemini-2.5"):
            return types.ThinkingConfig(thinking_budget=0)
        if m.startswith("gemini-3"):
            # 3.7 / 3.8 Flash don't offer "minimal"; their lowest level is "low".
            level = "low" if m.startswith(("gemini-3.7", "gemini-3.8")) else "minimal"
            return types.ThinkingConfig(thinking_level=level)
    except Exception as e:   # old SDK without thinking_level
        print(f"[gemini] can't set thinking level ({e}); run: pip install -U google-genai")
    return None


class GeminiAssistant:
    def __init__(self, models: list[str] | None = None, max_turns: int = 4):
        # Client() reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment.
        self.client = genai.Client(http_options=types.HttpOptions(timeout=TIMEOUT_MS))
        self.models = models or MODELS
        self.model = self.models[0]   # updated to whichever model actually answers
        self.max_turns = max_turns
        self.history: list[tuple[str, str]] = []   # (user text, assistant text); images not kept

    def reset(self):
        self.history.clear()

    def _build_contents(self, query, jpegs, note):
        contents = []
        for user_text, model_text in self.history:
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_text)]))
            contents.append(types.Content(role="model", parts=[types.Part.from_text(text=model_text)]))

        parts = [types.Part.from_bytes(data=j, mime_type="image/jpeg") for j in (jpegs or [])]
        text = query if not note else f"{query}\n\n(System note: {note})"
        parts.append(types.Part.from_text(text=text))
        contents.append(types.Content(role="user", parts=parts))
        return contents

    def ask_stream(self, query: str, jpegs: list[bytes] | None = None, note: str = ""):
        """Yield the reply sentence by sentence."""
        contents = self._build_contents(query, jpegs, note)
        # Temperature left at the default: Google advises against lowering it on Gemini 3.
        config = dict(system_instruction=SYSTEM_PROMPT, max_output_tokens=1024)

        buffer, spoken = "", []
        for chunk in self._stream_with_fallback(contents, config):
            piece = getattr(chunk, "text", None)
            if not piece:
                continue
            buffer += piece
            *done, buffer = _SENTENCE_END.split(buffer)
            for sentence in done:
                sentence = _clean(sentence)
                if sentence:
                    spoken.append(sentence)
                    yield sentence

        tail = _clean(buffer)
        if tail:
            spoken.append(tail)
            yield tail

        if spoken:
            self._remember(query, " ".join(spoken))

    def _stream_with_fallback(self, contents, config_kwargs):
        """
        Try each model in turn with minimal thinking. Fails over only if nothing has
        been received yet (never mid-answer, which could produce a garbled reply).
        """
        last_err = None
        for model in self.models:
            tc = _thinking_config(model)
            for thinking in ([tc, None] if tc else [None]):
                config = types.GenerateContentConfig(thinking_config=thinking, **config_kwargs)
                started = False
                try:
                    for chunk in self.client.models.generate_content_stream(
                        model=model, contents=contents, config=config
                    ):
                        if not started:
                            started = True
                            self.model = model
                        yield chunk
                    return
                except errors.APIError as e:
                    last_err = e
                    if started:
                        raise
                    if e.code == 400 and thinking is not None:
                        print(f"[gemini] {model}: thinking setting rejected, retrying without it")
                        continue
                    if e.code not in RETRY_CODES:
                        raise
                    print(f"[gemini] {model} unavailable ({e.code}), trying next model...")
                    break
        raise last_err

    def ask_structured(self, prompt: str, jpegs: list[bytes], schema):
        """
        One non-streaming call that returns a parsed `schema` object (a pydantic model).
        Same minimal-thinking and model-fallback behaviour as ask_stream. Raises on failure.
        """
        parts = [types.Part.from_bytes(data=j, mime_type="image/jpeg") for j in jpegs]
        parts.append(types.Part.from_text(text=prompt))
        contents = [types.Content(role="user", parts=parts)]

        last_err = None
        for model in self.models:
            tc = _thinking_config(model)
            for thinking in ([tc, None] if tc else [None]):
                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    max_output_tokens=1024,
                    thinking_config=thinking,
                )
                try:
                    resp = self.client.models.generate_content(
                        model=model, contents=contents, config=config)
                    self.model = model
                    if getattr(resp, "parsed", None) is not None:
                        return resp.parsed
                    return schema.model_validate_json(resp.text)
                except errors.APIError as e:
                    last_err = e
                    if e.code == 400 and thinking is not None:
                        continue
                    if e.code not in RETRY_CODES:
                        raise
                    print(f"[gemini] {model} unavailable ({e.code}), trying next model...")
                    break
        raise last_err

    def ask(self, query, jpegs=None, note="") -> str:
        return " ".join(self.ask_stream(query, jpegs, note))

    def _remember(self, query: str, answer: str):
        self.history.append((query, answer))
        self.history = self.history[-self.max_turns:]


# ---------------- Self-test ----------------
if __name__ == "__main__":
    import sys

    g = GeminiAssistant()

    if "--models" in sys.argv:
        for m in g.client.models.list():
            if "flash" in m.name.lower():
                print(m.name)
        sys.exit()

    from camera import CameraBuffer

    cam = CameraBuffer().start()
    time.sleep(2)
    jpegs = cam.get_query_jpegs(n=3)
    print(f"Model: {g.model} | sending {len(jpegs)} frames ({sum(map(len, jpegs)) // 1024} KB)")
    t0 = time.time()
    for sentence in g.ask_stream("Describe what is in front of me.", jpegs):
        print(f"[{time.time() - t0:.1f}s] {sentence}")
    print(f"Total: {time.time() - t0:.1f}s")
    cam.stop()
