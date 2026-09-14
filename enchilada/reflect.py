"""Session reflection: distil durable facts from a conversation.

Honcho's value is not that it stores transcripts — it is that it *derives* a
model of the user and revises it over time. This module does the derivation half
for Enchilada: at a real session boundary, an auxiliary LLM reads the transcript
and returns only facts worth keeping, or nothing at all.

Design constraints, each learned the expensive way:

* **Never ingest raw turns.** A knowledge graph fed chat logs fills with
  conversational noise and pays LLM extraction for every line of it.
* **Opt-in.** Writing to someone's knowledge base without asking is a surprise;
  ``ENCHILADA_REFLECT`` must be set truthy.
* **Cheap and bounded.** Runs on the auxiliary model (the one Hermes already uses
  for summaries), not the main model, and only at session end — never per turn.
* **Empty is the common answer.** Most sessions produce no durable fact. The
  prompt says so explicitly, and an empty reply is a success, not a failure.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Reflection reads the tail of a session; enough for the substance, bounded so a
# long session cannot blow up the auxiliary call.
MAX_TRANSCRIPT_CHARS = 12000
MAX_FACTS = 5

_SYSTEM_PROMPT = """\
You extract durable facts from a conversation for a long-term knowledge base.

Return ONLY facts that stay true after this conversation ends and that would
genuinely help in a future, unrelated session:

* stable preferences and working conventions ("deploys go through staging first")
* decisions and their reasons ("chose Postgres over Mongo because of the join load")
* durable project/domain facts ("the billing service owns retry logic")
* corrections of something previously believed

Never return:

* what happened in this session ("we debugged the timeout", "the tests passed")
* task progress, todos, or transient state
* generic knowledge the model already has
* anything you are not confident about

MOST CONVERSATIONS CONTAIN NOTHING DURABLE. Returning an empty list is the
normal, correct outcome — do not invent facts to fill the quota.

Reply with JSON only: {"facts": [{"title": "...", "text": "..."}]}
`title` is a short noun phrase. `text` is one or two self-contained sentences
that make sense to someone who never saw this conversation. Maximum %d facts.\
""" % MAX_FACTS


def _render_transcript(messages: List[Dict[str, Any]]) -> str:
    """Flatten to `role: text`, tail-truncated. Tool calls and their results are
    dropped: they are execution detail, and their content is what produced the
    assistant's prose anyway."""
    lines: List[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        lines.append(f"{role}: {content.strip()}")

    transcript = "\n\n".join(lines)
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        transcript = "…(earlier turns omitted)…\n\n" + transcript[-MAX_TRANSCRIPT_CHARS:]
    return transcript


def _parse_facts(reply: str) -> List[Dict[str, str]]:
    """Tolerant JSON extraction — models wrap JSON in prose or fences."""
    if not reply or not reply.strip():
        return []
    text = reply.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            text = brace.group(0)

    try:
        payload = json.loads(text)
    except ValueError:
        logger.debug("Reflection reply was not JSON: %s", reply[:200])
        return []

    raw = payload.get("facts") if isinstance(payload, dict) else payload
    if not isinstance(raw, list):
        return []

    facts: List[Dict[str, str]] = []
    for item in raw[:MAX_FACTS]:
        if not isinstance(item, dict):
            continue
        body = str(item.get("text") or "").strip()
        if not body:
            continue
        title = str(item.get("title") or "").strip() or body[:60]
        facts.append({"title": title, "text": body})
    return facts


def reflect(messages: List[Dict[str, Any]], *, timeout: float = 60.0) -> List[Dict[str, str]]:
    """Ask the auxiliary model for durable facts. Returns [] on anything unusual —
    an empty result is the expected outcome for most sessions, and a reflection
    failure must never surface to the user."""
    transcript = _render_transcript(messages)
    if len(transcript) < 200:  # nothing of substance was said
        return []

    try:
        from agent.auxiliary_client import resolve_provider_client
    except ImportError:
        logger.debug("Reflection skipped: auxiliary client unavailable")
        return []

    try:
        client, model = resolve_provider_client("auto", task="memory")
        if client is None or not model:
            logger.debug("Reflection skipped: no auxiliary model resolved")
            return []

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content":
                 f"Conversation:\n\n{transcript}\n\nExtract durable facts as JSON."},
            ],
            timeout=timeout,
        )
        reply = (response.choices[0].message.content or "") if response.choices else ""
    except Exception as exc:  # noqa: BLE001 - reflection is best-effort by design
        logger.debug("Reflection call failed: %s", exc)
        return []

    facts = _parse_facts(reply)
    logger.debug("Reflection produced %d fact(s)", len(facts))
    return facts
