"""Minimal client for a local Ollama server (stdlib only).

Talks to Ollama's native HTTP API at $OLLAMA_HOST (default localhost:11434).
Exposes just what the app needs: a health check, model listing, a streaming
chat generator, and a schema-constrained JSON chat for structured output.

No third-party HTTP dependency on purpose -- the pipeline's other modules pull
in torch/ase already, and the assistant should stay lightweight and offline.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List, Optional

def _resolve_host() -> str:
    # Ollama's OLLAMA_HOST convention is a bare host:port (no scheme); urllib
    # needs a scheme, so add http:// when one is missing.
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").strip()
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host.rstrip("/")


OLLAMA_HOST = _resolve_host()
DEFAULT_MODEL = os.environ.get("ASALD_LLM_MODEL", "qwen2.5:7b-instruct-q4_K_M")

# Deterministic-leaning defaults; the surface only ever asks for factual
# analysis of a report, never creative text.
DEFAULT_OPTIONS = {"temperature": 0.2, "num_ctx": 8192}


class OllamaError(RuntimeError):
    """Raised when the local server is unreachable or returns an error."""


def _post(path: str, payload: Dict[str, Any], timeout: float):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def is_up(timeout: float = 2.0) -> bool:
    """True if the Ollama server answers on OLLAMA_HOST."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/version",
                                    timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def list_models(timeout: float = 5.0) -> List[str]:
    """Names of locally pulled models (empty list if the server is down)."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=timeout) as r:
            tags = json.loads(r.read())
        return sorted(m["name"] for m in tags.get("models", []))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError):
        return []


def chat_stream(model: str, messages: List[Dict[str, str]],
                options: Optional[Dict[str, Any]] = None,
                timeout: float = 600.0) -> Iterator[str]:
    """Yield assistant text chunks as the model generates them.

    `messages` is the usual [{"role", "content"}, ...] list. Streams Ollama's
    newline-delimited JSON and yields each `message.content` fragment so the UI
    can render tokens live.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {**DEFAULT_OPTIONS, **(options or {})},
    }
    try:
        resp = _post("/api/chat", payload, timeout=timeout)
    except urllib.error.URLError as e:
        raise OllamaError(f"cannot reach Ollama at {OLLAMA_HOST}: {e}") from e
    with resp:
        for raw in resp:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if obj.get("error"):
                raise OllamaError(obj["error"])
            chunk = obj.get("message", {}).get("content", "")
            if chunk:
                yield chunk
            if obj.get("done"):
                break


def chat_tools(model: str, messages: List[Dict[str, Any]],
               tools: List[Dict[str, Any]],
               options: Optional[Dict[str, Any]] = None,
               timeout: float = 600.0) -> Dict[str, Any]:
    """Return the assistant *message* for a tool-enabled turn.

    Passes `tools` (OpenAI-style function schemas) to Ollama's /api/chat. The
    returned dict is the raw ``message`` object: it has ``content`` and, when
    the model decides to act, a ``tool_calls`` list of
    ``{"function": {"name", "arguments"}}``. The agent loop dispatches those,
    feeds results back as role="tool" messages, and calls this again. Kept
    non-streaming because tool_calls only arrive complete.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "tools": tools,
        "options": {**DEFAULT_OPTIONS, **(options or {})},
    }
    try:
        resp = _post("/api/chat", payload, timeout=timeout)
    except urllib.error.URLError as e:
        raise OllamaError(f"cannot reach Ollama at {OLLAMA_HOST}: {e}") from e
    with resp:
        obj = json.loads(resp.read())
    if obj.get("error"):
        raise OllamaError(obj["error"])
    return obj.get("message", {}) or {}


def chat_json(model: str, messages: List[Dict[str, str]],
              schema: Dict[str, Any],
              options: Optional[Dict[str, Any]] = None,
              timeout: float = 600.0) -> Dict[str, Any]:
    """Return a dict validated against `schema` via Ollama structured output.

    Passing a JSON schema as `format` constrains decoding so small models
    reliably emit parseable JSON -- the mechanism Job 3 relies on for its
    suggestion payload.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": schema,
        "options": {**DEFAULT_OPTIONS, **(options or {})},
    }
    try:
        resp = _post("/api/chat", payload, timeout=timeout)
    except urllib.error.URLError as e:
        raise OllamaError(f"cannot reach Ollama at {OLLAMA_HOST}: {e}") from e
    with resp:
        obj = json.loads(resp.read())
    if obj.get("error"):
        raise OllamaError(obj["error"])
    content = obj.get("message", {}).get("content", "").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise OllamaError(f"model did not return valid JSON: {content[:200]}") from e
