# Enchilada Memory Provider

Graph-based knowledge memory for Hermes, backed by [Enchilada](https://getenchilada.com) —
documents are ingested, entities and relations extracted, and queries answered over the graph.

Works against the hosted cloud or a self-hosted instance
([`mimagic/enchilada-compose`](https://github.com/mimagic/enchilada-compose)).

## Install

The plugin lives in `$HERMES_HOME/plugins/enchilada/` — note the path is
`plugins/<name>/`, **not** `plugins/memory/<name>/`; the loader's user-plugin root is
`$HERMES_HOME/plugins` and a `memory/` subdirectory is never scanned.

```bash
hermes config set memory.provider enchilada
hermes memory status          # verify: installed ✓ / available ✓
```

## Configuration

Secrets belong in `~/.hermes/.env` (chmod 600), never in `config.yaml`:

| Variable | Required | Default | Purpose |
| :--- | :--- | :--- | :--- |
| `ENCHILADA_API_KEY` | yes | — | Bearer key (`ench_*`) |
| `ENCHILADA_URL` | no | `https://getenchilada.com` | Instance base URL |
| `ENCHILADA_WORKSPACE` | no | account default | Workspace **UUID** |
| `ENCHILADA_TIMEOUT` | no | `10` | Per-call seconds |
| `ENCHILADA_TOP_K` | no | `5` | Documents recalled per turn |
| `ENCHILADA_RECALL` | no | `on` | `off` disables automatic recall (tools stay) |

**The workspace header wants the UUID, not the `ragWorkspace` slug.** Passing the
slug (`ws_…`) returns `403 Workspace not found or not accessible`. List them with:

```bash
curl -sS -H "Authorization: Bearer $ENCHILADA_API_KEY" \
  https://getenchilada.com/api/v1/workspaces
```

## How it behaves

**Recall (automatic).** Each turn's user message searches the knowledge base in a
background thread; hits are injected into the *user* message as an
`<enchilada-memory>` block — never the system prompt, which would break prompt
caching. Trivial prompts ("ok", "thanks", slash commands) are skipped and drop any
buffered context, so a leftover lookup never lands on an unrelated reply.

**Reads fail open.** A slow, unreachable or unauthenticated instance yields no
context and never blocks a turn. Absence of a memory block therefore does *not*
mean "nothing is known".

**Writes are explicit.** Turns are not auto-ingested. Enchilada is a curated
knowledge base, not a chat log: every document costs LLM extraction and pollutes
the graph with conversational noise. The model writes only via
`enchilada_remember`, when the user asks.

## Tools

| Tool | Use |
| :--- | :--- |
| `enchilada_search` | Targeted document lookup (ranked snippets) |
| `enchilada_ask` | Graph-reasoned answer connecting entities across documents; slower |
| `enchilada_remember` | Store a note as a new document |

## Requirements on the instance side

The Enchilada instance needs an **LLM key configured in its own settings**
(`/app/settings`) — ingest needs it for entity extraction, search for embeddings.
Without one, both return:

```
400 {"error":"LLM API key not configured. Visit /app/settings to add your key."}
```

Recall degrades silently in that case (logged once at WARNING); the tools return
the error with a hint.

## Ingest is asynchronous

`enchilada_remember` returns `{"status":"queued","job_id":…}` immediately. A
document only becomes searchable once processing completes — roughly 30 seconds for
a short note. Check with:

```bash
curl -sS -H "Authorization: Bearer $ENCHILADA_API_KEY" \
     -H "X-Workspace: $ENCHILADA_WORKSPACE" \
     https://getenchilada.com/api/v1/documents | jq '.documents[] | {filename, status, rag_synced_at}'
```

Searching before `rag_synced_at` is set returns zero hits — that is timing, not failure.

## Relation to other memory providers

Enchilada models **your material** (documents, entities, relations). It does not
model *you* — it only knows what was explicitly stored. Hermes's built-in
`MEMORY.md` / `USER.md` layer keeps running alongside it and still holds durable
user facts; the two are complementary.

## Development

```bash
PYTHONPATH=/path/to/hermes-agent python -m pytest tests/ -q
```

The suite runs offline against a fake client — no API key, no network.

## Repository layout

```
enchilada/        the plugin itself (copy to $HERMES_HOME/plugins/enchilada/)
  __init__.py     EnchiladaMemoryProvider
  client.py       stdlib-only REST client
  plugin.yaml     manifest
tests/            offline behaviour tests
```
