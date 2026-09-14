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

# Hermes bounds an external provider's prefetch at 8s (_EXTERNAL_PREFETCH_TIMEOUT_S)
# and then skips the provider until the stuck call returns. Staying at or below that
# keeps the client's own timeout the one that fires, so we can still report back.
DEFAULT_TIMEOUT = 8.0


class EnchiladaError(RuntimeError):
    """An API call failed. ``status`` is the HTTP code when there was one."""

    def __init__(self, message: str, status: Optional[int] = None,
                 *, timed_out: bool = False):
        super().__init__(message)
        self.status = status
        self.timed_out = timed_out

    @property
    def needs_llm_key(self) -> bool:
        """True for the instance-side "no LLM key configured" refusal, which blocks
        ingest and search until the user adds a key in the Enchilada portal."""
        return "LLM API key not configured" in str(self)


class EnchiladaClient:
    def __init__(self, api_key: str, base_url: str = DEFAULT_URL,
                 workspace: str = "", timeout: float = DEFAULT_TIMEOUT):
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
            # socket.timeout is a TimeoutError subclass; URLError wraps it too.
            reason = getattr(exc, "reason", exc)
            timed_out = isinstance(exc, TimeoutError) or isinstance(reason, TimeoutError)
            raise EnchiladaError(
                f"timed out after {timeout or self.timeout:g}s" if timed_out
                else f"unreachable: {exc}",
                timed_out=timed_out,
            ) from None

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

    def documents(self, *, limit: int = 100, review_status: str = "",
                  timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """``/documents`` (not ``/documents/recent``) — carries ``review_status`` and
        ``rag_synced_at``. ``review_status="unreviewed"`` is server-side filtered, which
        is what lets an undo find autonomously-written documents from PAST sessions."""
        path = f"/documents?limit={int(limit)}"
        if review_status:
            path += f"&review_status={review_status}"
        result = self._request("GET", path, timeout=timeout)
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
        """Graph-reasoned answer. Measured ~3-4s on a cold query versus ~0.3s for
        search, so it gets its own budget and is never used for automatic recall."""
        result = self._request("POST", "/query", {"query": question},
                               timeout=timeout or max(self.timeout, 30.0))
        if isinstance(result, dict):
            for key in ("answer", "response", "result", "text"):
                value = result.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        return result if isinstance(result, str) else ""

    def insert_text(self, text: str, title: str = "",
                    *, review_status: str = "", metadata: Optional[Dict[str, Any]] = None,
                    timeout: Optional[float] = None) -> Dict[str, Any]:
        """``review_status="unreviewed"`` flags a document the user never approved,
        so autonomously-learned material stays auditable and prunable in the portal."""
        payload: Dict[str, Any] = {"text": text}
        if title:
            payload["title"] = title
        if review_status:
            payload["review_status"] = review_status
        if metadata:
            payload["metadata"] = metadata
        result = self._request("POST", "/documents/text", payload, timeout=timeout)
        return result if isinstance(result, dict) else {}

    def delete_document(self, document_id: str,
                        *, timeout: Optional[float] = None) -> bool:
        """Irreversible. Used only to undo autonomously-learned facts the user rejects."""
        try:
            self._request("DELETE", f"/documents/{document_id}", timeout=timeout)
            return True
        except EnchiladaError as exc:
            logger.debug("Deleting document %s failed: %s", document_id, exc)
            return False

    def ping(self) -> bool:
        """Cheap reachability + auth probe that does not need an LLM key."""
        try:
            self.recent(limit=1)
            return True
        except EnchiladaError:
            return False
