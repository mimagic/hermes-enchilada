"""Provider-side trivial-prompt gate.

Hermes core gates recall with ``agent.memory_provider.is_trivial_prompt``, whose
word list is English. A German-speaking user's "passt", "danke" or "ja, passt!"
carries exactly as much signal as "ok", so every acknowledgement slips through
the core gate and costs a pointless search round-trip against the knowledge base.

This module widens the classifier for THIS plugin only: core stays untouched, and
the provider simply declines to spend a request on a prompt it knows is empty.
That direction is safe — the core gate runs first and only decides whether the
provider is consulted at all, so a provider may be stricter but never looser.
Anything core already calls trivial never reaches us.
"""

from __future__ import annotations

import re
from typing import Optional

from agent.memory_provider import is_trivial_prompt as core_is_trivial_prompt

# One acknowledgement token. Multi-word phrases come first so the alternation
# prefers the long reading ("alles klar" over "alles"). Kept to genuine
# contentless replies: a word that commonly OPENS a real request (e.g. "zeig",
# "kannst") does not belong here.
_WORD = (
    r'(?:alles klar|alles gut|guten morgen|guten tag|guten abend|guter punkt|'
    r'sehr gut|danke schön|dankeschön|mach das|passt so|na klar|'
    r'ja|nein|jo|joa|klar|logo|gerne|gern|bitte|danke|'
    r'passt|stimmt|genau|richtig|korrekt|verstanden|'
    r'weiter|okey|oki|jupp|jup|nee|nö|doch|'
    r'super|prima|perfekt|top)'
)
_PUNCT = (r'[\s!?.:;,"' + "'"
          + r'~\u2018\u2019\u201c\u201d\u2014\u2013\u2026()\[\]{}<>*&^%$#@!+=`\u00a0]')

# A RUN of acknowledgements ("ja, passt!", "alles klar danke") is still an
# acknowledgement; a single-token regex misses exactly these. Separators between
# them are whitespace/punctuation only, so "ja beides fixen" stays substantial.
_GERMAN_TRIVIAL_RE = re.compile(
    r'^' + _WORD + r'(?:' + _PUNCT + r'+' + _WORD + r')*' + _PUNCT + r'*$',
    re.IGNORECASE,
)


def is_trivial_prompt(text: Optional[str]) -> bool:
    """True when core says so, or when the prompt is a German acknowledgement."""
    if core_is_trivial_prompt(text):
        return True
    return bool(_GERMAN_TRIVIAL_RE.match((text or "").strip()))
