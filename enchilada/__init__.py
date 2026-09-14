"""Enchilada memory provider for Hermes.

Enchilada (getenchilada.com) is a graph-based knowledge platform: documents are
ingested, entities and relations extracted, and queries answered over the graph.
This provider wires it into Hermes as a memory backend:

* **Recall** — before each turn, a background thread searches the knowledge base
  with the user's message; results are injected into the *user* message by the
  memory manager (never the system prompt, to keep prompt caching intact).
* **Writes are explicit by default.** Turns are NOT auto-ingested. Enchilada is a
  curated knowledge base, not a chat log: every document costs LLM extraction
  work and pollutes the graph if it is conversational noise. The model writes
  when the user asks, via ``enchilada_remember``.
* **Reflection** — the provider distils *durable facts* from conversations with
  an auxiliary model. Two modes (``ENCHILADA_REFLECT``):

  ``auto``  learn continuously: facts are stored immediately, flagged
            ``unreviewed`` so everything learned without explicit approval stays
            visible and prunable in the portal. The user is told what was kept.
  ``ask``   propose only: the suggestion rides the next memory block and nothing
            is written until the user agrees.

  Either way the extraction is conservative — session events, task progress and
  low-confidence guesses are dropped, and most conversations yield nothing.

Config (env, all optional except the key):
  ENCHILADA_API_KEY    required, ``ench_*``
  ENCHILADA_URL        default https://getenchilada.com
  ENCHILADA_WORKSPACE  workspace UUID (omit for the account default)
  ENCHILADA_TIMEOUT    per-call seconds, default 8 (must stay <= the core's 8s cap)
  ENCHILADA_TOP_K      recall hits per turn, default 5
  ENCHILADA_RECALL     ``off`` disables automatic recall (tools still work)
  ENCHILADA_REFLECT    ``auto`` | ``ask`` | ``off`` (default off)
  ENCHILADA_REFLECT_EVERY  turns between reflection passes, default 6
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
    spawn_context_thread,
)

from .client import DEFAULT_TIMEOUT, DEFAULT_URL, EnchiladaClient, EnchiladaError
from .trivial import is_trivial_prompt

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

# Reflection proposes; the user disposes. The wording has to make the model ASK
# rather than store, because the tool to store is sitting right there.
_PROPOSAL_HEADER = (
    "REFLECTION — these look like durable facts from this conversation. They are "
    "NOT stored. Ask the user, in your own words and only if it fits the moment, "
    "whether to keep them; call enchilada_remember ONLY after they agree. If they "
    "decline or ignore it, drop the subject and do not ask again."
)

# Autonomous mode still tells the user — silent background writes to their own
# knowledge base are the kind of thing people discover months later and resent.
_LEARNED_HEADER = (
    "REFLECTION — learned and stored from this conversation, flagged unreviewed. "
    "Mention this briefly and naturally if there is a good moment; do not derail "
    "the conversation for it. If the user objects, call enchilada_forget_learned "
    "to remove them and stop learning for this session."
)


def _truthy(value: Optional[str], default: bool = True) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() not in ("0", "false", "off", "no")


def _reflect_mode(value: Optional[str]) -> str:
    """``auto`` learns autonomously, ``ask`` proposes, anything else is off.
    Bare truthy values map to ``ask`` — the safer reading of an ambiguous ``on``."""
    normalized = (value or "").strip().lower()
    if normalized in ("auto", "always", "autonomous"):
        return "auto"
    if normalized in ("ask", "propose", "suggest"):
        return "ask"
    return "ask" if normalized in ("1", "true", "on", "yes") else "off"


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
        # Reflection state: proposals wait here until a turn picks them up.
        self._reflect_mode = "off"           # off | ask | auto
        self._reflect_every = 6
        self._reflect_turn = 0
        self._reflect_running = False
        self._proposal: str = ""
        self._proposed_titles: set = set()   # never propose/store the same fact twice
        self._declined = False               # one refusal silences reflection for the session
        self._learned_ids: List[str] = []    # documents written autonomously this session

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
        self._reflect_mode = _reflect_mode(os.environ.get("ENCHILADA_REFLECT"))
        try:
            self._reflect_every = max(2, int(os.environ.get("ENCHILADA_REFLECT_EVERY", "") or 6))
        except ValueError:
            self._reflect_every = 6

        self._client = EnchiladaClient(
            api_key=os.environ.get("ENCHILADA_API_KEY", "").strip(),
            base_url=os.environ.get("ENCHILADA_URL", "").strip() or DEFAULT_URL,
            workspace=os.environ.get("ENCHILADA_WORKSPACE", "").strip(),
            timeout=timeout,
        )
        # Writes are explicit, so a non-primary agent context is no reason to
        # disable anything — recall is just as useful for a subagent.

    def system_prompt_block(self) -> str:
        if not self._recall_enabled and self._reflect_mode == "off":
            return ""
        block = (
            "Enchilada knowledge base: relevant documents are recalled automatically "
            "and appear under <enchilada-memory>. Cite them when you use them. "
            "Use enchilada_search for a targeted lookup, enchilada_ask for a "
            "graph-reasoned answer across documents, and enchilada_remember ONLY "
            "when the user explicitly asks to store something."
        )
        if self._reflect_mode == "auto":
            block += (
                " Reflection also learns durable facts from conversations on its own "
                "and stores them flagged unreviewed; a REFLECTION note lists what was "
                "kept. Do not repeat that list verbatim — mention it naturally when it "
                "fits, and use enchilada_forget_learned if the user objects."
            )
        return block

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
        # A pending proposal is delivered even when recall is off or the prompt is
        # trivial — "ok" is exactly when the user has room to answer a question.
        with self._lock:
            proposal, self._proposal = self._proposal, ""

        if not (self._recall_enabled and self._client):
            self._last_status = None
            return self._wrap_proposal(proposal)
        # A trivial prompt gets no context AND drops anything buffered: reusing
        # another query's hits on "thanks" is worse than injecting nothing.
        if is_trivial_prompt(query):
            with self._lock:
                self._pending.pop(session_id, None)
            self._last_status = None
            return self._wrap_proposal(proposal)

        with self._lock:
            entry = self._pending.pop(session_id, None)
        if entry is None:
            # Nothing queued yet (first turn): fetch inline, bounded by the client timeout.
            self._fetch(query, session_id)
            with self._lock:
                entry = self._pending.pop(session_id, None)
        if not entry:
            self._last_status = None
            return self._wrap_proposal(proposal)

        context, count = entry
        # count 0 means "a note, no documents" — the core renders count<=0 generically,
        # so the label carries the distinction instead of claiming a recall happened.
        label = "enchilada" if count else "enchilada (no hits)"
        self._last_status = RecallStatus(provider_label=label, count=count)
        if proposal:
            context = context.replace(
                "</enchilada-memory>", f"\n{proposal}\n</enchilada-memory>")
        return context

    @staticmethod
    def _wrap_proposal(proposal: str) -> str:
        return f"<enchilada-memory>\n{proposal}\n</enchilada-memory>" if proposal else ""

    def recall_status(self) -> Optional[RecallStatus]:
        return self._last_status

    # -- reflection --------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *,
                  session_id: str = "", messages: Optional[List[Dict[str, Any]]] = None,
                  turn_author: Optional[Dict[str, Any]] = None) -> None:
        """Raw turns are never written. This only advances the reflection clock and,
        when due, runs the extraction pass. Runs on the manager's background worker,
        so the LLM call and any writes are free to take their time."""
        if self._reflect_mode == "off" or not self._client or self._declined:
            return
        self._reflect_turn += 1
        if self._reflect_turn % self._reflect_every:
            return
        with self._lock:
            if self._reflect_running or self._proposal:
                return  # one pass at a time; don't stack proposals
            self._reflect_running = True
        try:
            self._reflect(list(messages or []))
        finally:
            with self._lock:
                self._reflect_running = False

    def _reflect(self, messages: List[Dict[str, Any]]) -> None:
        from .reflect import reflect

        facts = reflect(messages)
        fresh = [f for f in facts if f["title"].lower() not in self._proposed_titles]
        if not fresh:
            return
        for fact in fresh:
            self._proposed_titles.add(fact["title"].lower())

        if self._reflect_mode == "ask":
            body = "\n".join(f"- {f['title']}: {f['text']}" for f in fresh)
            with self._lock:
                self._proposal = f"{_PROPOSAL_HEADER}\n\n{body}"
            return

        stored = self._store_learned(fresh)
        if stored:
            body = "\n".join(f"- {title}" for title in stored)
            with self._lock:
                self._proposal = f"{_LEARNED_HEADER}\n\n{body}"

    def _store_learned(self, facts: List[Dict[str, str]]) -> List[str]:
        """Write autonomously-learned facts, flagged ``unreviewed`` so they are
        auditable and prunable rather than indistinguishable from curated material."""
        stored: List[str] = []
        for fact in facts:
            try:
                result = self._client.insert_text(
                    fact["text"], fact["title"],
                    review_status="unreviewed",
                    metadata={"source": "hermes-reflection", "session": self._session_id},
                )
            except EnchiladaError as exc:
                logger.debug("Storing learned fact failed: %s", exc)
                continue
            document_id = str(result.get("rag_doc_id") or result.get("id") or "")
            if document_id:
                self._learned_ids.append(document_id)
            stored.append(fact["title"])
        return stored

    def _learned_document_ids(self) -> List[str]:
        """Documents this provider wrote autonomously, session-local list FIRST and the
        server's ``unreviewed`` set as the durable fallback.

        ``_learned_ids`` is cleared on every session switch, so relying on it alone made
        the undo a lie the moment a user opened a new session: facts reflection wrote
        yesterday were unreachable. ``review_status`` is the server-side record of the
        same thing, which is exactly why autonomous writes carry it.
        """
        ids: List[str] = list(self._learned_ids)
        if not self._client:
            return ids
        seen = set(ids)
        try:
            for document in self._client.documents(review_status="unreviewed"):
                if not isinstance(document, dict):
                    continue
                document_id = str(document.get("rag_doc_id") or document.get("id") or "")
                if document_id and document_id not in seen:
                    seen.add(document_id)
                    ids.append(document_id)
        except EnchiladaError as exc:
            # The session-local ids still get deleted; a partial undo beats none.
            logger.debug("Listing unreviewed documents failed: %s", exc)
        return ids

    def decline_reflection(self) -> None:
        """Stop learning for this session (the user said no)."""
        self._declined = True
        with self._lock:
            self._proposal = ""

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
            {
                "name": "enchilada_forget_learned",
                "description": ("Delete facts reflection stored autonomously (this "
                                "session and earlier ones) and stop learning for the "
                                "rest of this session. Call when the user objects to "
                                "something reflection stored. Curated documents the "
                                "user asked for are never touched."),
                "parameters": {"type": "object", "properties": {}},
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
            if tool_name == "enchilada_forget_learned":
                removed = [doc_id for doc_id in self._learned_document_ids()
                           if self._client.delete_document(doc_id)]
                self._learned_ids.clear()
                self.decline_reflection()
                return json.dumps({"removed": len(removed),
                                   "learning_disabled_for_session": True},
                                  ensure_ascii=False)
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
            self._proposal = ""
        self._session_id = new_session_id or ""
        self._last_status = None
        if reset:
            # A genuinely new conversation: an old refusal and old proposals no
            # longer apply, and the reflection clock starts over.
            self._declined = False
            self._proposed_titles.clear()
            self._reflect_turn = 0
            self._learned_ids.clear()

    def shutdown(self) -> None:
        with self._lock:
            self._pending.clear()
            self._inflight = None
            self._proposal = ""

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
            {"key": "reflect", "description":
             "Learn durable facts from conversations: auto (store, flagged "
             "unreviewed) | ask (propose first) | off",
             "default": "off", "choices": ["off", "auto", "ask"],
             "env_var": "ENCHILADA_REFLECT"},
            {"key": "reflect_every", "description": "Turns between reflection passes",
             "type": "integer", "default": 6, "minimum": 2, "maximum": 50,
             "env_var": "ENCHILADA_REFLECT_EVERY"},
        ]
