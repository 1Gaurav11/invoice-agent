# A2A Invoice Action Agent

A2A 1.0 (HTTP+JSON) agent that reads invoice batches, picks one typed action per
package with cited evidence, and only records receipts for proposals the grader
accepts. Python standard library only: no `pip install`.

```
server.py   protocol layer: routes, auth, version/media checks, lifecycle, race handling
store.py    SQLite: tasks, (principal, messageId) idempotency records, decision cache
decide.py   AI layer: one batched model call, strict validation, repair round, cache
inspect_db.py   debug helper: print what was received/decided
tests/      protocol + model-client tests (no network, no key needed)
```

## 1. Run the tests (offline)

```
python3 -m unittest discover -s tests -v
```

## 2. Configure

| Variable | Meaning |
|---|---|
| `BASE_URL` | **Exactly** the base URL you paste into the grader, e.g. `https://my-app.onrender.com/a2a/` (trailing slash included if you include it there). The agent card's `supportedInterfaces[0].url` is this string, and routes are served under its path. |
| `LLM_API_KEY`, `LLM_MODEL` | Your model provider key and model name. |
| `LLM_PROVIDER` | `openai` (default; any OpenAI-compatible `/chat/completions` endpoint) or `anthropic`. |
| `LLM_BASE_URL` | Endpoint root for OpenAI-compatible providers (default `https://api.openai.com/v1`). Gemini, Groq, OpenRouter, Together, vLLM and Ollama all expose compatible endpoints. |
| `LLM_EXTRA_BODY` | Optional JSON merged into the request, e.g. to turn off "thinking" on a reasoning model so calls stay fast. |
| `LLM_MAX_TOKENS` | Output cap per call (default 8000). |
| `EVIDENCE_REF_STYLE` | `verbatim` (default: refs returned exactly as written in the documents, brackets included) or `bare` (brackets stripped). See "Calibrating". |
| `DB_PATH` | SQLite file (default `data/agent.sqlite`). Put it on a persistent disk if you can. |
| `PORT` | Listen port (default 8080). |
| `CACHE_SALT` | Change it to discard cached decisions (do this after switching model or prompt). |
| `ALLOWED_TOKENS` | Optional comma list. Leave unset for the grader (see assumptions). |

Pick a small, fast, non-reasoning model (or disable thinking). Each request has
45 s and the whole verification 160 s. The first core task is the only expensive
one; later ones hit the cache.

Never put real keys or finance data in prompts; the case files are synthetic.

## 3. Deploy (needs a public HTTPS URL, single instance)

Run **one** process: the race/idempotency locks are in-process.

* **Fly.io / Railway / Render / Cloud Run (min instances = 1)**: deploy this folder with the
  `Dockerfile` (or start command `python3 server.py`, no build step). Set the variables above.
  The platform terminates HTTPS and forwards to `$PORT`.
* Attach a volume at `/data` if offered. Without one the DB resets on redeploy, which is fine
  for grading but do not redeploy between a Check and its continuation.
* Avoid free tiers that sleep: a cold start can exceed the 45 s request limit.
* Quick test from your own machine: `BASE_URL=https://<tunnel-host>/a2a/ python3 server.py`
  plus a Cloudflare/ngrok tunnel to port 8080.

Sanity check after deploy:

```
curl -s https://<host>/.well-known/agent-card.json | python3 -m json.tool | head -20
curl -si -H 'A2A-Version: 1.0' https://<host>/a2a/tasks     # expect 401
```

Submit `BASE_URL` in the grader. No redirects are issued; the card is at the origin root.

## 4. Calibrating with your Checks (you have 9 before the reserved Save)

1. Run **Check** once and read the failure codes.
2. `python3 inspect_db.py data/agent.sqlite --refs` shows each package's action, facts, refs and rationale.
3. If the codes point at **evidence**: try `EVIDENCE_REF_STYLE=bare` (or back to `verbatim`), and set a new
   `CACHE_SALT`. Evidence format is the one thing I could not verify without a sample package.
4. If the codes point at **decision quality**: look at the wrong packages in `inspect_db.py` and
   adjust the rules in `SYSTEM_PROMPT` in `decide.py`; bump `PROMPT_VERSION`/`CACHE_SALT`.
5. Keep the last check for **Save**. Only your most recent Save counts.

## 5. Behaviour summary and assumptions

* Any non-empty `Authorization: Bearer <token>` is accepted; each distinct token is a separate
  principal (SHA-256 of the token). The grader's tokens are not known in advance, so a fixed
  allow-list would reject it. Task reads/cancel/continue, listing and messageId dedupe are all
  scoped to the principal; another user's task id returns the same generic 404 as a missing one.
* Dedupe key `(principal, messageId)`; hash is recursively key-sorted compact JSON of the message
  only. Equivalent replays (including concurrent ones) return the stored Task with no model call;
  changed content returns 409 `IDEMPOTENCY_CONFLICT`. A task is saved before any model work, and
  unfinished tasks are completed after a restart.
* Decisions are cached by canonical package content (package/batch/message ids removed) **plus
  `policyRevision`**, in SQLite. Fallback decisions (model down) never settle and are never cached.
* Results are checked against the stored proposal (task, context, batch, package, actionId, action);
  a bad continuation changes nothing. Receipt-vs-cancel is one atomic state transition: one 200,
  the other 409. Terminal tasks never change; replays return the stored task.
* The receipts artifact is added as a second artifact; the proposals artifact stays.
* Error bodies follow the `google.rpc.Status` shape (`error.code/status/message/details`), generic text only.
