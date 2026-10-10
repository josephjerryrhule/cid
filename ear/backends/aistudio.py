"""Gemini through a Google AI Studio API key. One structured request per call."""
from __future__ import annotations

import os
import time
from pathlib import Path

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from backends import BackendError, RateLimited

DEFAULT_MODEL = "gemini-flash-latest"


def model_name() -> str:
    return os.environ.get("LISTEN_MODEL", DEFAULT_MODEL)


def hear(audio: Path, prompt: str, schema_model: type[BaseModel], *, sleep=time.sleep, retries: int = 4) -> dict:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise BackendError("GEMINI_API_KEY not set in ~/.config/listen/.env")
    client = genai.Client(api_key=key)
    part = types.Part.from_bytes(data=Path(audio).read_bytes(), mime_type="audio/ogg")
    config = types.GenerateContentConfig(
        response_mime_type="application/json", response_schema=schema_model, temperature=0.2
    )
    delay = 5.0
    for attempt in range(retries + 1):
        try:
            resp = client.models.generate_content(model=model_name(), contents=[part, prompt], config=config)
            return schema_model.model_validate_json(resp.text).model_dump()
        except errors.APIError as e:
            if e.code == 429 and attempt < retries:
                sleep(delay)
                delay = min(delay * 2, 60.0)
                continue
            if e.code == 429:
                raise RateLimited(str(e))
            raise BackendError(f"{e.code}: {e.message}")
    raise RateLimited("retries exhausted")
