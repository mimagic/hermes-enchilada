"""Behaviour tests for the Enchilada memory provider.

No network and no API key required: the client is replaced by a fake. These
assert the contracts that matter to a Hermes turn — recall never blocks, trivial
prompts inject nothing, failures stay silent — not the shape of the source.

    python -m pytest tests/test_provider.py -v
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

# Import the plugin package by path, the way Hermes loads it.
PLUGIN_DIR = Path(__file__).resolve().parent.parent / "enchilada"
sys.path.insert(0, str(PLUGIN_DIR.parent))

pytest.importorskip("agent.memory_provider",
                    reason="needs a Hermes checkout on sys.path")

import importlib.util


def _load_plugin():
    """Load the plugin as a package so its relative imports resolve, mirroring
    how Hermes registers it under a synthetic namespace."""
    package = "enchilada_plugin"
    spec = importlib.util.spec_from_file_location(
        package, PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)])
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package
    sys.modules[package] = module  # must be registered BEFORE exec for `from .client`
    spec.loader.exec_module(module)
    return module


plugin = _load_plugin()
EnchiladaMemoryProvider = plugin.EnchiladaMemoryProvider
EnchiladaError = plugin.EnchiladaError


class FakeClient:
    """Stands in for EnchiladaClient; records calls, fakes hits and failures."""

    def __init__(self, hits=None, fail=None, delay=0.0):
        self.hits = hits if hits is not None else []
        self.fail = fail
        self.delay = delay
        self.searches = []
        self.inserts = []

    def search(self, query, top_k=5, timeout=None):
        self.searches.append(query)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise self.fail
        return self.hits[:top_k]

    def query(self, question, timeout=None):
        if self.fail:
            raise self.fail
        return f"answer to {question}"

    def insert_text(self, text, title="", timeout=None):
        if self.fail:
            raise self.fail
        self.inserts.append((text, title))
        return {"status": "queued", "job_id": "job_1"}


HIT = {"title": "doc one", "snippet": "the user is marsch"}
HIT2 = {"title": "doc two", "snippet": "enchilada is a graph platform"}


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("ENCHILADA_API_KEY", "ench_test")
    monkeypatch.delenv("ENCHILADA_RECALL", raising=False)
    monkeypatch.delenv("ENCHILADA_TOP_K", raising=False)
    p = EnchiladaMemoryProvider()
    p.initialize("s1", hermes_home="/tmp", platform="cli")
    return p


# -- availability ---------------------------------------------------------

def test_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("ENCHILADA_API_KEY", raising=False)
    p = EnchiladaMemoryProvider()
    assert p.is_available() is False
    assert "ENCHILADA_API_KEY" in p.unavailable_reason()


def test_available_with_key(monkeypatch):
    monkeypatch.setenv("ENCHILADA_API_KEY", "ench_test")
    assert EnchiladaMemoryProvider().is_available() is True


# -- recall ---------------------------------------------------------------

def test_recall_injects_hits_and_counts_them(provider):
    provider._client = FakeClient(hits=[HIT, HIT2])
    context = provider.prefetch("who is marsch?", session_id="s1")
    assert "<enchilada-memory>" in context
    assert "the user is marsch" in context
    assert provider.recall_status().count == 2


def test_recall_empty_when_no_hits(provider):
    provider._client = FakeClient(hits=[])
    assert provider.prefetch("nothing here", session_id="s1") == ""
    assert provider.recall_status() is None


@pytest.mark.parametrize("prompt", ["ok", "thanks", "hi", "/help", ""])
def test_trivial_prompts_never_query(provider, prompt):
    fake = FakeClient(hits=[HIT])
    provider._client = fake
    assert provider.prefetch(prompt, session_id="s1") == ""
    assert fake.searches == []
    assert provider.recall_status() is None


def test_trivial_prompt_discards_buffered_context(provider):
    """A leftover lookup must not land on an unrelated reply."""
    provider._client = FakeClient(hits=[HIT])
    provider.queue_prefetch("who is marsch?", session_id="s1")
    for _ in range(50):
        if provider._pending.get("s1"):
            break
        time.sleep(0.02)
    assert provider._pending.get("s1"), "background prefetch never landed"

    assert provider.prefetch("ok", session_id="s1") == ""
    assert provider._pending.get("s1") is None


def test_background_prefetch_is_consumed_next_turn(provider):
    fake = FakeClient(hits=[HIT])
    provider._client = fake
    provider.queue_prefetch("who is marsch?", session_id="s1")
    for _ in range(50):
        if provider._pending.get("s1"):
            break
        time.sleep(0.02)
    context = provider.prefetch("tell me about marsch", session_id="s1")
    assert "the user is marsch" in context
    assert len(fake.searches) == 1, "consumed turn must not search again"


def test_recall_can_be_disabled(monkeypatch):
    monkeypatch.setenv("ENCHILADA_API_KEY", "ench_test")
    monkeypatch.setenv("ENCHILADA_RECALL", "off")
    p = EnchiladaMemoryProvider()
    p.initialize("s1", hermes_home="/tmp", platform="cli")
    fake = FakeClient(hits=[HIT])
    p._client = fake
    assert p.prefetch("who is marsch?", session_id="s1") == ""
    assert fake.searches == []
    assert p.system_prompt_block() == ""


def test_top_k_is_honoured(monkeypatch):
    monkeypatch.setenv("ENCHILADA_API_KEY", "ench_test")
    monkeypatch.setenv("ENCHILADA_TOP_K", "1")
    p = EnchiladaMemoryProvider()
    p.initialize("s1", hermes_home="/tmp", platform="cli")
    p._client = FakeClient(hits=[HIT, HIT2])
    p.prefetch("query", session_id="s1")
    assert p.recall_status().count == 1


# -- fail-open ------------------------------------------------------------

def test_recall_fails_open_on_error(provider):
    """Unreachable: no documents, but the model is TOLD memory was consulted."""
    provider._client = FakeClient(fail=EnchiladaError("unreachable"))
    context = provider.prefetch("who is marsch?", session_id="s1")
    assert "NOTE:" in context and "unreachable" in context.lower()
    assert provider.recall_status().count == 0


def test_recall_reports_timeout_as_possibly_incomplete(provider):
    """A timeout must never read as 'the knowledge base is empty'."""
    provider._client = FakeClient(
        fail=EnchiladaError("timed out after 8s", timed_out=True))
    context = provider.prefetch("who is marsch?", session_id="s1")
    assert "NOTE:" in context
    assert "in time" in context
    assert "enchilada_search" in context, "must offer a retry path"


def test_recall_fails_open_on_missing_llm_key(provider):
    provider._client = FakeClient(
        fail=EnchiladaError("LLM API key not configured. Visit /app/settings", status=400))
    context = provider.prefetch("who is marsch?", session_id="s1")
    assert "NOTE:" in context and "LLM key" in context


def test_full_page_of_hits_signals_more_exist(monkeypatch):
    """The API returns no total, so a full page is the 'there is more' signal."""
    monkeypatch.setenv("ENCHILADA_API_KEY", "ench_test")
    monkeypatch.setenv("ENCHILADA_TOP_K", "2")
    p = EnchiladaMemoryProvider()
    p.initialize("s1", hermes_home="/tmp", platform="cli")
    p._client = FakeClient(hits=[HIT, HIT2, HIT])
    context = p.prefetch("query", session_id="s1")
    assert "More matches exist" in context


def test_partial_page_does_not_claim_more(provider):
    provider._client = FakeClient(hits=[HIT])
    context = provider.prefetch("query", session_id="s1")
    assert "More matches exist" not in context


def test_unexpected_exception_does_not_escape(provider):
    provider._client = FakeClient(fail=ValueError("boom"))
    assert provider.prefetch("who is marsch?", session_id="s1") == ""


# -- tools ----------------------------------------------------------------

def test_tool_schemas_are_well_formed(provider):
    names = set()
    for schema in provider.get_tool_schemas():
        assert schema["name"] and schema["description"]
        assert schema["parameters"]["type"] == "object"
        names.add(schema["name"])
    assert names == {"enchilada_search", "enchilada_ask", "enchilada_remember"}


def test_search_tool_returns_json(provider):
    import json
    provider._client = FakeClient(hits=[HIT])
    result = json.loads(provider.handle_tool_call("enchilada_search", {"query": "marsch"}))
    assert result["count"] == 1


def test_remember_tool_stores(provider):
    import json
    fake = FakeClient()
    provider._client = fake
    result = json.loads(provider.handle_tool_call(
        "enchilada_remember", {"text": "a fact", "title": "t"}))
    assert result["stored"] is True
    assert fake.inserts == [("a fact", "t")]


def test_remember_rejects_empty_text(provider):
    import json
    provider._client = FakeClient()
    result = json.loads(provider.handle_tool_call("enchilada_remember", {"text": "  "}))
    assert "error" in result


def test_tool_error_includes_llm_key_hint(provider):
    import json
    provider._client = FakeClient(
        fail=EnchiladaError("LLM API key not configured. Visit /app/settings", status=400))
    result = json.loads(provider.handle_tool_call("enchilada_search", {"query": "x"}))
    assert "error" in result and "hint" in result


def test_unknown_tool_is_reported(provider):
    import json
    provider._client = FakeClient()
    assert "error" in json.loads(provider.handle_tool_call("nope", {}))


# -- lifecycle ------------------------------------------------------------

def test_session_switch_drops_stale_context(provider):
    provider._client = FakeClient(hits=[HIT])
    provider.queue_prefetch("who is marsch?", session_id="s1")
    time.sleep(0.2)
    provider.on_session_switch("s2", reset=True)
    assert provider._pending.get("s1") is None
    assert provider.recall_status() is None


def test_shutdown_clears_state(provider):
    provider._client = FakeClient(hits=[HIT])
    provider.queue_prefetch("query", session_id="s1")
    time.sleep(0.2)
    provider.shutdown()
    assert provider._pending == {}


def test_config_schema_marks_key_secret(provider):
    fields = {f["key"]: f for f in provider.get_config_schema()}
    assert fields["api_key"]["secret"] is True
    assert fields["api_key"]["env_var"] == "ENCHILADA_API_KEY"


def test_reflection_defaults_to_off(provider):
    """Writing to someone's knowledge base uninvited is a surprise."""
    assert provider._reflect_enabled is False


# -- reflection -----------------------------------------------------------

FACTS = [{"title": "deploy flow", "text": "Deploys go through staging first."}]


@pytest.fixture
def reflecting(monkeypatch):
    monkeypatch.setenv("ENCHILADA_API_KEY", "ench_test")
    monkeypatch.setenv("ENCHILADA_REFLECT", "on")
    monkeypatch.setenv("ENCHILADA_REFLECT_EVERY", "2")
    p = EnchiladaMemoryProvider()
    p.initialize("s1", hermes_home="/tmp", platform="cli")
    p._client = FakeClient(hits=[])
    return p


def _drive_turns(provider, count, facts=FACTS):
    import enchilada_plugin.reflect as reflect_mod
    original = reflect_mod.reflect
    reflect_mod.reflect = lambda messages, **kw: list(facts)
    try:
        for _ in range(count):
            provider.sync_turn("u", "a", session_id="s1", messages=[{"role": "user", "content": "x"}])
    finally:
        reflect_mod.reflect = original


def test_reflection_proposes_but_never_stores(reflecting):
    """The whole point: a fact is offered, not written."""
    fake = FakeClient(hits=[])
    reflecting._client = fake
    _drive_turns(reflecting, 2)
    context = reflecting.prefetch("what now?", session_id="s1")
    assert "REFLECTION" in context
    assert "Deploys go through staging first" in context
    assert fake.inserts == [], "reflection must not write on its own"


def test_proposal_instructs_the_model_to_ask(reflecting):
    _drive_turns(reflecting, 2)
    context = reflecting.prefetch("what now?", session_id="s1")
    assert "NOT stored" in context
    assert "enchilada_remember ONLY after they agree" in context


def test_proposal_is_delivered_even_on_a_trivial_prompt(reflecting):
    """'ok' is exactly when the user has room to answer a question."""
    _drive_turns(reflecting, 2)
    context = reflecting.prefetch("ok", session_id="s1")
    assert "REFLECTION" in context


def test_proposal_is_delivered_once(reflecting):
    _drive_turns(reflecting, 2)
    assert "REFLECTION" in reflecting.prefetch("q", session_id="s1")
    assert "REFLECTION" not in reflecting.prefetch("q", session_id="s1")


def test_same_fact_is_never_proposed_twice(reflecting):
    _drive_turns(reflecting, 2)
    reflecting.prefetch("q", session_id="s1")      # consume
    _drive_turns(reflecting, 2)                     # same fact again
    assert reflecting._proposal == ""


def test_declining_silences_reflection_for_the_session(reflecting):
    reflecting.decline_reflection()
    _drive_turns(reflecting, 4)
    assert reflecting._proposal == ""
    assert "REFLECTION" not in reflecting.prefetch("q", session_id="s1")


def test_reflection_respects_its_cadence(reflecting):
    _drive_turns(reflecting, 1)          # cadence is 2
    assert reflecting._proposal == ""
    _drive_turns(reflecting, 1)
    assert "REFLECTION" in reflecting._proposal


def test_reflection_is_skipped_when_disabled(provider):
    _drive_turns(provider, 10)
    assert provider._proposal == ""


def test_reset_clears_refusal_and_history(reflecting):
    reflecting.decline_reflection()
    reflecting.on_session_switch("s2", reset=True)
    assert reflecting._declined is False
    assert reflecting._proposed_titles == set()


def test_empty_reflection_proposes_nothing(reflecting):
    _drive_turns(reflecting, 2, facts=[])
    assert reflecting._proposal == ""
    assert reflecting.prefetch("q", session_id="s1") == ""


def test_reflect_parses_fenced_json():
    from enchilada_plugin.reflect import _parse_facts

    facts = _parse_facts('```json\n{"facts": [{"title": "t", "text": "body"}]}\n```')
    assert facts == [{"title": "t", "text": "body"}]


def test_reflect_tolerates_garbage():
    from enchilada_plugin.reflect import _parse_facts

    assert _parse_facts("I could not find any durable facts.") == []
    assert _parse_facts("") == []


def test_reflect_drops_factless_entries():
    from enchilada_plugin.reflect import _parse_facts

    facts = _parse_facts('{"facts": [{"title": "t"}, {"text": "keeps"}]}')
    assert len(facts) == 1 and facts[0]["text"] == "keeps"


def test_client_timeout_stays_within_the_core_prefetch_budget():
    """Hermes aborts an external prefetch at _EXTERNAL_PREFETCH_TIMEOUT_S and then
    skips the provider until the stuck call returns. A client timeout above that
    never fires, so the provider can no longer report why it went quiet."""
    from agent.memory_manager import _EXTERNAL_PREFETCH_TIMEOUT_S

    assert plugin.DEFAULT_TIMEOUT <= _EXTERNAL_PREFETCH_TIMEOUT_S


def test_default_timeout_is_used_when_env_is_unset(monkeypatch):
    monkeypatch.setenv("ENCHILADA_API_KEY", "ench_test")
    monkeypatch.delenv("ENCHILADA_TIMEOUT", raising=False)
    p = EnchiladaMemoryProvider()
    p.initialize("s1", hermes_home="/tmp", platform="cli")
    assert p._client.timeout == plugin.DEFAULT_TIMEOUT
