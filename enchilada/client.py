"""Minimal Enchilada REST client (stdlib only).

Enchilada exposes a knowledge base of documents plus a graph-reasoned query
endpoint. This wraps the handful of endpoints the memory provider needs, with
hard timeouts so a slow/unreachable instance can never stall a turn.

Auth: ``Authorization: Bearer ench_*`` and ``X-Workspace: <workspace UUID>``.
Note the workspace header wants the workspace **UUID**, not the ``ragWorkspace``
slug — passing the slug yields a 403 "not found or not accessible".
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_URL = "https://getenchilada.com"


class EnchiladaError(RuntimeError):
    """An API call failed. ``status`` is the HTTP code when there was one."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status

    @property
    def needs_llm_key(self) -> bool:
        """True for the instance-side "no LLM key configured" refusal, which blocks
        ingest and search until the user adds a key in the Enchilada portal."""
        return "LLM API key not configured" in str(self)


class EnchiladaClient:
    def __init__(self, api_key: str, base_url: str = DEFAULT_URL,
                 workspace: str = "", timeout: float = 10.0):
        self.api_key = api_key
        self.base_url = (base_url or DEFAULT_URL).rstrip("/")
        self.workspace = workspace or ""
        self.timeout = timeout

    # -- transport ---------------------------------------------------------

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None,
                 *, timeout: Optional[float] = None) -> Any:
        url = f"{self.base_url}/api/v1{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.api_key}")
        req.add_header("Accept", "application/json")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if self.workspace:
            req.add_header("X-Workspace", self.workspace)

        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                body = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            try:
                detail = json.loads(detail).get("error", detail)
            except Exception:
                pass
            raise EnchiladaError(detail or exc.reason, status=exc.code) from None
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise EnchiladaError(f"unreachable: {exc}") from None

        if not body:
            return None
        try:
            return json.loads(body)
        except ValueError:
            raise EnchiladaError(f"non-JSON response: {body[:200]}") from None

    # -- endpoints ---------------------------------------------------------

    def list_workspaces(self) -> List[Dict[str, Any]]:
        result = self._request("GET", "/workspaces")
        return result if isinstance(result, list) else []

    def recent(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Chronological — genuinely recent, not a search proxy."""
        result = self._request("GET", f"/documents/recent?limit={int(limit)}")
        return (result or {}).get("documents", []) if isinstance(result, dict) else []

    def search(self, query: str, top_k: int = 5,
               *, timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        result = self._request("POST", "/documents/search",
                               {"query": query, "top_k": int(top_k)}, timeout=timeout)
        if isinstance(result, dict):
            for key in ("results", "documents", "matches"):
                if isinstance(result.get(key), list):
                    return result[key]
            return []
        return result if isinstance(result, list) else []

    def query(self, question: str, *, timeout: Optional[float] = None) -> str:
        """Graph-reasoned answer. Slower than search — give it a longer timeout."""
        result = self._request("POST", "/query", {"query": question},
                               timeout=timeout or max(self.timeout, 30.0))
        if isinstance(result, dict):
            for key in ("answer", "response", "result", "text"):
                value = result.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        return result if isinstance(result, str) else ""

    def insert_text(self, text: str, title: str = "",
                    *, timeout: Optional[float] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"text": text}
        if title:
            payload["title"] = title
        result = self._request("POST", "/documents/text", payload, timeout=timeout)
        return result if isinstance(result, dict) else {}

    def ping(self) -> bool:
        """Cheap reachability + auth probe that does not need an LLM key."""
        try:
            self.recent(limit=1)
            return True
        except EnchiladaError:
            return False
