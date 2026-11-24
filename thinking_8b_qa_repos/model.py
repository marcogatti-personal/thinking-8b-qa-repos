from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from .common import byte_len, dump


def obj(properties, required=None):
    return {"type": "object", "properties": properties, "required": required or list(properties), "additionalProperties": False}


def array(items, maximum):
    return {"type": "array", "items": items, "maxItems": maximum}


STRING = {"type": "string"}


class InvalidModelResponse(ValueError):
    """A model response failed its output contract; distinct from transport errors."""


PLAN_SCHEMA = obj({"queries": array(STRING, 5), "unknowns": array(STRING, 4)})
READ_SCHEMA = obj({"findings": array(obj({"text": STRING, "source": STRING, "quote": STRING}), 6),
                   "queries": array(STRING, 4), "unknowns": array(STRING, 4)})
ANSWER_SCHEMA = obj({"claims": array(obj({"text": STRING, "sources": array(STRING, 4)}), 8),
                     "unknowns": array(STRING, 4)})
AUDIT_SCHEMA = obj({"supported": {"type": "boolean"}, "reason": STRING})
AUDIT_BATCH_SCHEMA = obj({"checks": array(obj({"id": STRING, "supported": {"type": "boolean"}, "reason": STRING}), 4)})


def validate(value, schema):
    """Small validator for the schema subset used here (no third-party runtime)."""
    kind = schema["type"]
    if kind == "object":
        if not isinstance(value, dict) or any(k not in value for k in schema["required"]):
            raise InvalidModelResponse("JSON object: required fields missing")
        if any(k not in schema["properties"] for k in value):
            raise InvalidModelResponse("JSON object: unexpected field")
        for key, item in value.items():
            validate(item, schema["properties"][key])
    elif kind == "array":
        if not isinstance(value, list) or len(value) > schema["maxItems"]:
            raise InvalidModelResponse("JSON array: invalid length/type")
        for item in value:
            validate(item, schema["items"])
    elif kind == "string" and not isinstance(value, str):
        raise InvalidModelResponse("Expected string")
    elif kind == "boolean" and type(value) is not bool:
        raise InvalidModelResponse("Expected boolean")


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class LocalModel:
    """Stateless, single-task calls. No native tool calling or shell execution."""
    def __init__(self, *, provider="openai", base_url="http://127.0.0.1:1234/v1", model,
                 prompt_bytes=12000, output_tokens=1536, context_window=16384,
                 timeout=120, retries=2, structured=True, allow_remote=False):
        parts = urllib.parse.urlsplit(base_url)
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
            raise ValueError("Invalid server URL")
        if parts.query or parts.fragment:
            raise ValueError("Base URL must not contain a query or fragment")
        if parts.hostname not in {"localhost", "127.0.0.1", "::1"} and not allow_remote:
            raise ValueError("Remote server: use --allow-remote to authorize sending repository code")
        if provider not in {"openai", "ollama"}:
            raise ValueError("Unknown provider")
        if prompt_bytes < 2048 or output_tokens < 128 or timeout <= 0 or retries < 0:
            raise ValueError("Invalid model limits")
        if prompt_bytes + output_tokens + 512 > context_window:
            raise ValueError("Increase --context-window or decrease --prompt-bytes/--output-tokens")
        self.provider, self.base_url, self.model = provider, base_url.rstrip("/"), model
        self.prompt_bytes, self.output_tokens = prompt_bytes, output_tokens
        self.context_window, self.timeout, self.retries = context_window, timeout, retries
        self.structured = structured
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        self.calls = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0, "max_request_bytes": 0}

    def cache_identity(self):
        return {"provider": self.provider, "base_url": self.base_url, "model": self.model,
                "structured": self.structured, "output_tokens": self.output_tokens,
                "context_window": self.context_window, "generation_policy": "task_caps_v2"}

    def complete(self, task, system, payload, schema):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": dump(payload)}]
        request_bytes = byte_len(dump(messages)) + (byte_len(dump(schema)) if self.structured else 0)
        if request_bytes > self.prompt_bytes:
            raise ValueError(f"Prompt exceeds budget: {request_bytes} > {self.prompt_bytes} bytes")
        self.usage["max_request_bytes"] = max(self.usage["max_request_bytes"], request_bytes)
        cap = {"plan": 384, "audit_batch": 512}.get(task, self.output_tokens)
        # A larger user budget must still allow recovery from a truncated reply.
        output_limit = min(self.output_tokens, int(cap * max(1, self.output_tokens / 1536)))
        if self.provider == "ollama":
            body = {"model": self.model, "messages": messages, "stream": False,
                    "options": {"temperature": 0, "num_predict": output_limit, "num_ctx": self.context_window}}
            if self.structured:
                body["format"] = schema
            endpoint = self.base_url + "/api/chat"
        else:
            body = {"model": self.model, "messages": messages, "stream": False,
                    "temperature": 0, "max_tokens": output_limit}
            if self.structured:
                body["response_format"] = {"type": "json_schema", "json_schema": {"name": task, "strict": True, "schema": schema}}
            endpoint = self.base_url + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("THINKING_8B_QA_REPOS_API_KEY") or os.environ.get("REPO_MEMORY_API_KEY")
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        last_error = None
        for attempt in range(self.retries + 1):
            self.calls += 1
            try:
                request = urllib.request.Request(endpoint, data=dump(body).encode(), headers=headers, method="POST")
                with self.opener.open(request, timeout=self.timeout) as response:
                    raw = response.read(2 * 1024 * 1024 + 1)
                if len(raw) > 2 * 1024 * 1024:
                    raise ValueError("HTTP response is too large")
                result = json.loads(raw)
                if self.provider == "ollama":
                    content = result["message"]["content"]
                    self.usage["input_tokens"] += int(result.get("prompt_eval_count", 0))
                    self.usage["output_tokens"] += int(result.get("eval_count", 0))
                    truncated = result.get("done_reason") == "length"
                else:
                    content = result["choices"][0]["message"]["content"]
                    usage = result.get("usage", {})
                    self.usage["input_tokens"] += int(usage.get("prompt_tokens", 0))
                    self.usage["output_tokens"] += int(usage.get("completion_tokens", 0))
                    truncated = result["choices"][0].get("finish_reason") == "length"
                if truncated:
                    raise ValueError("Response truncated: increase --output-tokens")
                parsed = parse_json(content)
                validate(parsed, schema)
                return parsed
            except urllib.error.HTTPError as exc:
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise RuntimeError(f"LLM server: HTTP {exc.code}; check model, URL and JSON Schema support. "
                                       "For servers without schema support, use --no-structured.") from exc
                last_error = f"HTTP {exc.code}"
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = f"{type(exc).__name__}: connection to the LLM server failed"
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                last_error = str(exc)
                # Synthesis owns corrective retries with a smaller task or feedback.
                # Replaying identical malformed reductions here wastes local inference.
                if task == "reduce":
                    raise InvalidModelResponse(f"Invalid reduction response: {last_error}") from exc
            if attempt < self.retries:
                time.sleep(min(2 ** attempt, 4))
        raise RuntimeError(f"Call {task} failed after {self.retries + 1} attempts: {last_error}")
