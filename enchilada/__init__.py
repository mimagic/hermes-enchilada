"""Enchilada memory provider for Hermes.

Enchilada (getenchilada.com) is a graph-based knowledge platform: documents are
ingested, entities and relations extracted, and queries answered over the graph.
This provider wires it into Hermes as a memory backend:

* **Recall** — before each turn, a background thread searches the knowledge base
  with the user's message; results are injected into the *user* message by the
  memory manager (never the system prompt, to keep prompt caching intact).
* **Writes are explicit.** Turns are NOT auto-ingested. Enchilada is a curated
  knowledge base, not a chat log: every document costs LLM extraction work and
  pollutes the graph if it is conversational noise. The model writes only when
  the user asks, via ``enchilada_remember``.

Config (env, all optional except the key):
  ENCHILADA_API_KEY    required, ``ench_*``
  ENCHILADA_URL        default https://getenchilada.com
  ENCHILADA_WORKSPACE  workspace UUID (omit for the account default)
  ENCHILADA_TIMEOUT    per-call seconds, default 10
  ENCHILADA_TOP_K      recall hits per turn, default 5
  ENCHILADA_RECALL     ``off`` disables automatic recall (tools still work)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional

from agent.memory_provider import (
    MemoryProvider,
    RecallStatus,
    is_trivial_prompt,
    spawn_context_thread,
)

from .client import DEFAULT_TIMEOUT, DEFAULT_URL, EnchiladaClient, EnchiladaError

logger = logging.getLogger(__name__)

_MAX_SNIPPET = 600
_MAX_CONTEXT = 4000

# Recall always reports back, even when it found nothing usable, so the model can
# tell "the knowledge base was consulted and is quiet" from "memory never ran".
_NOTE_TIMEOUT = ("Knowledge base did not answer in time — it may hold relevant "
                 "material that is missing here. Retry with enchilada_search if it matters.")
_NOTE_UNREACHABLE = ("Knowledge base unreachable — treat this as no information, "
                     "not as an empty knowledge base.")
_NOTE_NO_LLM_KEY = ("Knowledge base cannot answer: the instance has no LLM key "
                    "configured (/app/settings).")
_NOTE_MORE = ("More matches exist beyond those shown — use enchilada_search with a "
              "higher top_k, or enchilada_ask for a graph-reasoned synthesis.")


def _truthy(value: Optional[str], default: bool = True) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() not in ("0", "false", "off", "no")


class EnchiladaMemoryProvider(MemoryProvider):
    """Graph-based knowledge recall backed by an Enchilada instance."""

    @property
    def name(self) -> str:
        return "enchilada"

    def __init__(self) -> None:
        self._client: Optional[EnchiladaClient] = None
        self._session_id = ""
        self._lock = threading.Lock()
        self._pending: Dict[str, tuple] = {}     # session_id -> (context, hit count)
        self._inflight: Optional[str] = None     # session_id currently being fetched
        self._last_status: Optional[RecallStatus] = None
        self._llm_key_warned = False
        self._recall_enabled = True
        self._top_k = 5

    # -- availability ------------------------------------------------------

    def is_available(self) -> bool:
        return bool(os.environ.get("ENCHILADA_API_KEY", "").strip())

    def unavailable_reason(self) -> str:
        return ("ENCHILADA_API_KEY is not set. Get a key at https://getenchilada.com "
                "and add it to your .env, or run `hermes memory setup enchilada`.")

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id or ""
        try:
            timeout = float(os.environ.get("ENCHILADA_TIMEOUT", "") or DEFAULT_TIMEOUT)
        except ValueError:
            timeout = DEFAULT_TIMEOUT
        try:
            self._top_k = max(1, int(os.environ.get("ENCHILADA_TOP_K", "") or 5))
        except ValueError:
            self._top_k = 5
        self._recall_enabled = _truthy(os.environ.get("ENCHILADA_RECALL"), True)

        self._client = EnchiladaClient(
            api_key=os.environ.get("ENCHILADA_API_KEY", "").strip(),
            base_url=os.environ.get("ENCHILADA_URL", "").strip() or DEFAULT_URL,
            workspace=os.environ.get("ENCHILADA_WORKSPACE", "").strip(),
            timeout=timeout,
        )
        # Writes are explicit, so a non-primary agent context is no reason to
        # disable anything — recall is just as useful for a subagent.

    def system_prompt_block(self) -> str:
        if not self._recall_enabled:
            return ""
        return (
            "Enchilada knowledge base: relevant documents are recalled automatically "
            "and appear under <enchilada-memory>. Cite them when you use them. "
            "Use enchilada_search for a targeted lookup, enchilada_ask for a "
            "graph-reasoned answer across documents, and enchilada_remember ONLY "
            "when the user explicitly asks to store something."
        )

    # -- recall ------------------------------------------------------------

    def _format_hits(self, hits: List[Dict[str, Any]], *, note: str = "") -> str:
        lines: List[str] = []
        for index, hit in enumerate(hits, 1):
            if not isinstance(hit, dict):
                continue
            title = (hit.get("title") or hit.get("name")
                     or hit.get("documentTitle") or f"document {index}")
            body = ""
            for key in ("content", "text", "snippet", "chunk", "excerpt", "summary"):
                value = hit.get(key)
                if isinstance(value, str) and value.strip():
                    body = value.strip()
                    break
            if len(body) > _MAX_SNIPPET:
                body = body[:_MAX_SNIPPET].rsplit(" ", 1)[0] + "…"
            lines.append(f"[{index}] {title}\n{body}" if body else f"[{index}] {title}")

        block = "\n\n".join(lines)
        if len(block) > _MAX_CONTEXT:
            block = block[:_MAX_CONTEXT].rsplit("\n\n", 1)[0] + "\n\n…(truncated)"
            note = note or _NOTE_MORE
        if not block and not note:
            return ""
        parts = [part for part in (block, f"NOTE: {note}" if note else "") if part]
        return "<enchilada-memory>\n" + "\n\n".join(parts) + "\n</enchilada-memory>"

    def _fetch(self, query: str, session_id: str) -> None:
        """Always records an outcome: hits, or a note explaining the silence."""
        try:
            hits = self._client.search(query, top_k=self._top_k) if self._client else []
            count = sum(1 for hit in hits if isinstance(hit, dict))
            # No total in the API response, so a full page is the only "more exists"
            # signal available. Upstream could expose a real total later.
            note = _NOTE_MORE if count >= self._top_k else ""
            context = self._format_hits(hits, note=note)
            with self._lock:
                if context:
                    self._pending[session_id] = (context, count)
                self._inflight = None
        except EnchiladaError as exc:
            if exc.needs_llm_key:
                note = _NOTE_NO_LLM_KEY
                if not self._llm_key_warned:
                    self._llm_key_warned = True
                    logger.warning("Enchilada recall unavailable: no LLM key configured "
                                   "on the instance (add one at /app/settings).")
            elif exc.timed_out:
                note = _NOTE_TIMEOUT
                logger.warning("Enchilada recall timed out: %s", exc)
            else:
                note = _NOTE_UNREACHABLE
                logger.debug("Enchilada recall failed: %s", exc)
            with self._lock:
                self._pending[session_id] = (self._format_hits([], note=note), 0)
                self._inflight = None
        except Exception as exc:  # noqa: BLE001 - a recall bug must not kill the turn
            logger.debug("Enchilada recall error: %s", exc)
            with self._lock:
                self._inflight = None

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not (self._recall_enabled and self._client) or is_trivial_prompt(query):
            return
        session_id = session_id or self._session_id
        with self._lock:
            if self._inflight is not None:
                return  # one lookup at a time; a stale answer is worse than none
            self._inflight = session_id
        thread = spawn_context_thread(lambda: self._fetch(query, session_id),
                                      name="enchilada-recall")
        thread.start()  # spawn_context_thread returns an UNSTARTED thread

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        session_id = session_id or self._session_id
        if not (self._recall_enabled and self._client):
            self._last_status = None
            return ""
        # A trivial prompt gets no context AND drops anything buffered: reusing
        # another query's hits on "thanks" is worse than injecting nothing.
        if is_trivial_prompt(query):
            with self._lock:
                self._pending.pop(session_id, None)
            self._last_status = None
            return ""

        with self._lock:
            entry = self._pending.pop(session_id, None)
        if entry is None:
            # Nothing queued yet (first turn): fetch inline, bounded by the client timeout.
            self._fetch(query, session_id)
            with self._lock:
                entry = self._pending.pop(session_id, None)
        if not entry:
            self._last_status = None
            return ""

        context, count = entry
        # count 0 means "a note, no documents" — the core renders count<=0 generically,
        # so the label carries the distinction instead of claiming a recall happened.
        label = "enchilada" if count else "enchilada (no hits)"
        self._last_status = RecallStatus(provider_label=label, count=count)
        return context

    def recall_status(self) -> Optional[RecallStatus]:
        return self._last_status

    # -- tools -------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "enchilada_search",
                "description": ("Search the Enchilada knowledge base and return matching "
                                "documents. Use for a targeted lookup of stored material."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to look for."},
                        "top_k": {"type": "integer",
                                  "description": "How many documents (default 5)."},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "enchilada_ask",
                "description": ("Ask a question answered by reasoning over the knowledge "
                                "GRAPH — connects entities and relations across documents. "
                                "Slower than enchilada_search; use for synthesis questions."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The question."},
                    },
                    "required": ["question"],
                },
            },
            {
                "name": "enchilada_remember",
                "description": ("Store a note as a NEW document in the knowledge base. "
                                "Call ONLY when the user explicitly asks to remember or "
                                "store something — this is a curated knowledge base, not "
                                "a chat log."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The content to store."},
                        "title": {"type": "string", "description": "Short title."},
                    },
                    "required": ["text"],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if not self._client:
            return json.dumps({"error": "Enchilada is not initialized."})
        try:
            if tool_name == "enchilada_search":
                top_k = args.get("top_k") or self._top_k
                hits = self._client.search(str(args.get("query", "")), top_k=int(top_k))
                return json.dumps({"count": len(hits), "results": hits[:int(top_k)]},
                                  ensure_ascii=False, default=str)[:8000]
            if tool_name == "enchilada_ask":
                answer = self._client.query(str(args.get("question", "")))
                return json.dumps({"answer": answer}, ensure_ascii=False)[:8000]
            if tool_name == "enchilada_remember":
                text = str(args.get("text", "")).strip()
                if not text:
                    return json.dumps({"error": "text is required"})
                result = self._client.insert_text(text, str(args.get("title", "")).strip())
                return json.dumps({"stored": True, "document": result},
                                  ensure_ascii=False, default=str)[:4000]
        except EnchiladaError as exc:
            payload: Dict[str, Any] = {"error": str(exc), "status": exc.status}
            if exc.needs_llm_key:
                payload["hint"] = ("The Enchilada instance has no LLM key configured. "
                                   "Add one at /app/settings — ingest and search need it.")
            return json.dumps(payload, ensure_ascii=False)
        return json.dumps({"error": f"unknown tool {tool_name}"})

    # -- lifecycle ---------------------------------------------------------

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "",
                          reset: bool = False, rewound: bool = False, **kwargs) -> None:
        with self._lock:
            self._pending.pop(self._session_id, None)
            self._inflight = None
        self._session_id = new_session_id or ""
        self._last_status = None

    def shutdown(self) -> None:
        with self._lock:
            self._pending.clear()
            self._inflight = None

    # -- setup -------------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "api_key", "description": "Enchilada API key (ench_*)",
             "secret": True, "required": True, "env_var": "ENCHILADA_API_KEY",
             "url": "https://getenchilada.com/app/settings"},
            {"key": "url", "description": "Instance base URL",
             "default": DEFAULT_URL, "env_var": "ENCHILADA_URL"},
            {"key": "workspace", "description": "Workspace UUID (blank = account default)",
             "required": False, "env_var": "ENCHILADA_WORKSPACE"},
            {"key": "top_k", "description": "Documents recalled per turn",
             "type": "integer", "default": 5, "minimum": 1, "maximum": 20,
             "env_var": "ENCHILADA_TOP_K"},
            {"key": "recall", "description": "Automatic recall before each turn",
             "type": "boolean", "default": True, "env_var": "ENCHILADA_RECALL"},
        ]
