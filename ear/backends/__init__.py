"""Hearing backends. Each module exposes hear(audio, prompt, schema_model) -> dict."""


class BackendError(Exception):
    """The backend could not produce a hearing result."""


class RateLimited(BackendError):
    """The backend refused because of a rate or quota limit."""
