"""AI decision layer: one batched model call per task, strict validation, and a
persistent cache keyed by canonical package content.

Nothing in here knows about HTTP or A2A. Input: a batch dict. Output: one
validated decision per package.
"""
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
import hashlib

log = logging.getLogger("decide")

ACTIONS = ("settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception")

# Bump to invalidate cached decisions when the prompt/validation logic changes.
PROMPT_VERSION = "v1"

# Keys that name a *namespace* rather than commercial content. They are removed
# before hashing so Check/Save re-runs of the same package hit the cache.
NAMESPACE_KEYS = {"packageId", "batchId", "namespace", "taskId", "messageId", "contextId"}

BRACKET_RE = re.compile(r"\[[^\[\]\n]{1,120}\]")
CHUNK = 12

SYSTEM_PROMPT = """You are a careful accounts-payable controls analyst. You receive invoice packages. Each package is a bundle of documents (cover sheet, invoice, purchase order, receipts, ledger/payment records, policy text, approval records, archive examples, training material). For EVERY package you choose exactly ONE action and cite the decisive evidence.

ACTIONS (use these exact strings):
- settle_invoice: the invoice is valid, reconciled to the supporting records, and within autonomous (delegated) payment authority. Pay it.
- request_approval: the invoice is commercially valid and reconciled, but the amount or circumstances are OUTSIDE the delegated authority/limit, so a named approver must sign off before payment.
- hold_invoice: payment must pause until a specifically stated verification completes (for example a bank-detail callback, vendor identity check, missing receipt or confirmation).
- reject_duplicate: the SAME commercial invoice was already paid (same vendor, same invoice number/reference, same amount and currency). A payment for a DIFFERENT invoice (different invoice number, period or amount) from the same vendor is NOT a duplicate.
- open_exception: material records conflict (vendor, amount, currency, PO, quantities or dates disagree between the invoice and the supporting records) and the case needs an exception workflow rather than a payment decision.

HOW TO DECIDE
1. The documents deliberately mix useful facts with noise. Ignore: the cover-sheet summary/reference, archived or historical examples, old-policy examples, training or sample material, hypotheticals, and action words that merely appear in passing.
2. Watch negation: "not a duplicate", "no hold is required", "does not need approval", "was not settled" mean the opposite of the bare action word.
3. Find the ONE paragraph that states the current, controlling finding for THIS invoice (the live reconciliation / payment-history / authority / verification determination). Base the action on that paragraph, read in the context of the current policy limits it relies on.
4. Evidence: return EXACTLY THREE references, the three bracketed references that appear in that deciding paragraph, copied character-for-character from the documents including the square brackets. Never use the cover-sheet reference, archive/example references or training decoys.
5. facts: vendorName and invoiceNumber exactly as printed on the invoice being decided; amountMinor is an INTEGER in minor currency units (for two-decimal currencies multiply the printed amount by 100, e.g. 1,234.50 INR -> 123450; zero-decimal currencies such as JPY are used as-is); currency is the 3-letter ISO code.
6. rationale: 80-400 characters, plain sentences. State the action name and quote at least two of the three evidence references. Do not mention any other bracketed reference.

OUTPUT: respond with ONE JSON object and nothing else:
{"decisions":[{"packageId":"...","action":"<one of the five>","facts":{"vendorName":"...","invoiceNumber":"...","amountMinor":12345,"currency":"INR"},"evidenceRefs":["[..]","[..]","[..]"],"rationale":"..."}]}
Include exactly one decision per package, in the same order, using the packageId given."""


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def strip_namespace(obj):
    if isinstance(obj, dict):
        return {k: strip_namespace(v) for k, v in obj.items() if k not in NAMESPACE_KEYS}
    if isinstance(obj, list):
        return [strip_namespace(v) for v in obj]
    return obj


def cache_key(policy_revision, package, salt=""):
    blob = canonical({"v": PROMPT_VERSION, "salt": salt, "policy": policy_revision, "pkg": strip_namespace(package)})
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def leaves(obj, out):
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            leaves(v, out)
    elif isinstance(obj, list):
        for v in obj:
            leaves(v, out)
    return out


def find_key(obj, names):
    """First scalar value stored under any key in `names` (case-insensitive)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in names and isinstance(v, (str, int, float)) and not isinstance(v, bool):
                return v
        for v in obj.values():
            r = find_key(v, names)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, names)
            if r is not None:
                return r
    return None


def parse_json_loose(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        return json.loads(text)
    except Exception:
        pass
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b > a:
        try:
            return json.loads(text[a : b + 1])
        except Exception:
            pass
    raise ValueError("model did not return parseable JSON")


def to_minor(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v) if v == int(v) else None
    if isinstance(v, str):
        s = v.strip().replace(",", "")
        if re.fullmatch(r"-?\d+", s):
            return int(s)
    return None


def canon_ref(ref, candidates, flat):
    """Map a model-returned reference to the form that literally appears in the text."""
    if not isinstance(ref, str):
        return None
    r = ref.strip()
    if not r:
        return None
    if r in candidates:
        return r
    if "[" + r + "]" in candidates:
        return "[" + r + "]"
    if r in flat:
        return r
    return None


def fix_rationale(text, action, refs):
    r = re.sub(r"\s+", " ", text or "").strip()
    keep = set(refs)
    # drop any other bracketed reference so the rationale never cites a decoy
    r = BRACKET_RE.sub(lambda m: m.group(0) if m.group(0) in keep else "", r)
    r = re.sub(r"\s+", " ", r).strip()
    if action not in r:
        r = f"Action {action}: {r}".strip()
    if sum(1 for x in refs if x in r) < min(2, len(refs)) or len(r) > 1500:
        base = r[:900] if len(r) > 900 else r
        r = f"{base} Evidence: {', '.join(refs)}."
    if len(r) < 60:
        r = f"{r} Decision {action} is supported by the cited evidence {', '.join(refs)}.".strip()
    return r[:1500]


# ----------------------------------------------------------------------------
# model clients
# ----------------------------------------------------------------------------
class LLMError(Exception):
    pass


class OpenAICompatLLM:
    """Any OpenAI-style /chat/completions endpoint (OpenAI, Gemini's compat
    endpoint, Groq, OpenRouter, Together, Ollama, vLLM, ...)."""

    def __init__(self, base_url, api_key, model, max_tokens=8000, extra=None, token_param="max_tokens", json_mode=True):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.key, self.model, self.max_tokens = api_key, model, max_tokens
        self.extra, self.token_param, self.json_mode = extra or {}, token_param, json_mode

    def complete(self, system, user, deadline):
        for attempt in range(3):
            remaining = deadline - time.monotonic()
            if remaining < 3:
                raise LLMError("no time left for model call")
            body = {
                "model": self.model,
                "temperature": 0,
                self.token_param: self.max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            }
            if self.json_mode:
                body["response_format"] = {"type": "json_object"}
            body.update(self.extra)
            req = urllib.request.Request(
                self.url,
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=max(3, remaining - 1)) as resp:
                    data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:300]
                if (e.code in (400, 404, 422) and self.json_mode
                        and any(w in detail.lower() for w in ("response_format", "json_object", "json mode", "structured", "parameter"))):
                    log.warning("provider rejected response_format; retrying without JSON mode")
                    self.json_mode = False
                    continue
                if e.code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(min(2 ** attempt + 0.5, max(0, deadline - time.monotonic() - 4)))
                    continue
                raise LLMError(f"model HTTP {e.code}: {detail}")
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt < 2 and deadline - time.monotonic() > 8:
                    continue
                raise LLMError(f"model request failed: {e}")
        raise LLMError("model call failed")


class AnthropicLLM:
    def __init__(self, api_key, model, base_url="https://api.anthropic.com", max_tokens=8000):
        self.url = base_url.rstrip("/") + "/v1/messages"
        self.key, self.model, self.max_tokens = api_key, model, max_tokens

    def complete(self, system, user, deadline):
        for attempt in range(3):
            remaining = deadline - time.monotonic()
            if remaining < 3:
                raise LLMError("no time left for model call")
            body = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": 0,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
            req = urllib.request.Request(
                self.url,
                data=json.dumps(body).encode(),
                headers={"content-type": "application/json", "x-api-key": self.key, "anthropic-version": "2023-06-01"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=max(3, remaining - 1)) as resp:
                    data = json.loads(resp.read())
                return "".join(b.get("text", "") for b in data.get("content", []))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:300]
                if e.code in (429, 500, 502, 503, 529) and attempt < 2:
                    time.sleep(min(2 ** attempt + 0.5, max(0, deadline - time.monotonic() - 4)))
                    continue
                raise LLMError(f"model HTTP {e.code}: {detail}")
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt < 2 and deadline - time.monotonic() > 8:
                    continue
                raise LLMError(f"model request failed: {e}")
        raise LLMError("model call failed")


class MockLLM:
    """Deterministic stand-in used by the test-suite (LLM_PROVIDER=mock)."""

    def __init__(self):
        self.calls = 0
        self._lock = threading.Lock()
        self.delay = 0.0

    def complete(self, system, user, deadline):
        with self._lock:
            self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        pkgs = json.loads(user.split("PACKAGES_JSON:\n", 1)[1])
        out = []
        for p in pkgs:
            decisive = None
            for t in leaves(p, []):
                refs = BRACKET_RE.findall(t)
                if len(refs) == 3 and not re.search(r"archive|example|decoy|cover", t, re.I):
                    decisive = (t, refs)
            text, refs = decisive if decisive else ("", [])
            m = re.search(r"Determination: (\w+)", text)
            inv = p.get("invoice", {})
            out.append({
                "packageId": p["packageId"],
                "action": m.group(1) if m else "open_exception",
                "facts": {k: inv.get(k) for k in ("vendorName", "invoiceNumber", "amountMinor", "currency")},
                "evidenceRefs": refs,
                "rationale": "short",
            })
        return json.dumps({"decisions": out})


def llm_from_env(env=os.environ):
    prov = env.get("LLM_PROVIDER", "openai").lower()
    if prov == "mock":
        return MockLLM()
    key = env.get("LLM_API_KEY", "")
    model = env.get("LLM_MODEL", "")
    if not key or not model:
        return None
    max_tokens = int(env.get("LLM_MAX_TOKENS", "8000"))
    if prov == "anthropic":
        return AnthropicLLM(key, model, env.get("LLM_BASE_URL", "https://api.anthropic.com"), max_tokens)
    extra = json.loads(env.get("LLM_EXTRA_BODY", "{}") or "{}")
    return OpenAICompatLLM(
        env.get("LLM_BASE_URL", "https://api.openai.com/v1"),
        key,
        model,
        max_tokens,
        extra,
        env.get("LLM_TOKEN_PARAM", "max_tokens"),
        env.get("LLM_JSON_MODE", "1") != "0",
    )


# ----------------------------------------------------------------------------
# Decider
# ----------------------------------------------------------------------------
class Decider:
    def __init__(self, store, llm, ref_style="verbatim", salt=""):
        self.store, self.llm, self.ref_style, self.salt = store, llm, ref_style, salt

    # public ------------------------------------------------------------
    def decide(self, batch, deadline):
        """Return a list of decisions aligned with batch['packages']."""
        packages = batch["packages"]
        policy = batch.get("policyRevision")
        keys = [cache_key(policy, p, self.salt) for p in packages]
        by_key = {}  # key -> decision (canonical, bracketed refs)
        todo = []  # (key, representative package)
        for k, p in zip(keys, packages):
            if k in by_key or any(k == t[0] for t in todo):
                continue
            hit = self.store.cache_get(k)
            if hit:
                by_key[k] = hit
            else:
                todo.append((k, p))
        stats = {"cached": len(packages) - sum(1 for _ in todo), "model": len(todo)}

        fresh_ok = {}
        if todo and self.llm is not None:
            for i in range(0, len(todo), CHUNK):
                chunk = todo[i : i + CHUNK]
                try:
                    got = self._ask(policy, [p for _, p in chunk], deadline)
                except LLMError as e:
                    log.error("model failure: %s", e)
                    continue
                for (k, p), d in zip(chunk, got):
                    if d is not None:
                        fresh_ok[k] = d
        elif todo:
            log.error("no LLM configured (set LLM_API_KEY and LLM_MODEL); using conservative fallback")

        out = []
        for k, p in zip(keys, packages):
            d = by_key.get(k)
            if d is None and k in fresh_ok:
                d = fresh_ok[k]
                if d.get("strong"):
                    self.store.cache_put(k, d)
            if d is None:
                d = self._fallback(p)
                log.warning("fallback decision used for package %s", p.get("packageId"))
            out.append(self._render(d))
        log.info("batch %s: %s", batch.get("batchId"), stats)
        return out

    # internals -----------------------------------------------------------
    def _render(self, d):
        refs = list(d["evidenceRefs"])
        if self.ref_style == "bare":
            refs = [r[1:-1] if r.startswith("[") and r.endswith("]") else r for r in refs]
        rationale = d["rationale"]
        if self.ref_style == "bare":
            rationale = fix_rationale(
                re.sub(r"\[([^\[\]\n]{1,120})\]", r"\1", rationale), d["action"], refs
            )
        return {
            "action": d["action"],
            "facts": dict(d["facts"]),
            "evidenceRefs": refs,
            "rationale": rationale,
        }

    def _fallback(self, pkg):
        """Used only when the model is unavailable. Never settles (a wrong
        settlement is the costliest error); never cached."""
        vendor = find_key(pkg, {"vendorname", "vendor", "supplier"}) or "Unknown vendor"
        number = find_key(pkg, {"invoicenumber", "invoice_number", "invoiceno", "invoiceid"}) or "UNKNOWN"
        amount = to_minor(find_key(pkg, {"amountminor"})) or 0
        currency = str(find_key(pkg, {"currency"}) or "INR").upper()[:3]
        flat = "\n".join(leaves(pkg, []))
        cands = []
        for r in BRACKET_RE.findall(flat):
            if r not in cands:
                cands.append(r)
        refs = cands[-3:] if len(cands) >= 3 else cands
        action = "open_exception"
        return {
            "action": action,
            "facts": {"vendorName": str(vendor), "invoiceNumber": str(number), "amountMinor": amount, "currency": currency},
            "evidenceRefs": refs,
            "rationale": fix_rationale(
                "Automatic analysis was unavailable, so the package is routed to the exception workflow for manual review.",
                action, refs),
        }

    def _validate(self, pkg, raw):
        problems = []
        if not isinstance(raw, dict):
            return None, ["decision is not an object"]
        action = raw.get("action")
        if action not in ACTIONS:
            problems.append(f"action must be one of {list(ACTIONS)}")
        f = raw.get("facts") if isinstance(raw.get("facts"), dict) else {}
        vendor = f.get("vendorName")
        number = f.get("invoiceNumber")
        if isinstance(number, (int, float)) and not isinstance(number, bool):
            number = str(number)
        amount = to_minor(f.get("amountMinor"))
        currency = f.get("currency")
        if not (isinstance(vendor, str) and vendor.strip()):
            problems.append("facts.vendorName missing")
        if not (isinstance(number, str) and number.strip()):
            problems.append("facts.invoiceNumber missing")
        if amount is None:
            problems.append("facts.amountMinor must be an integer in minor units")
        if not (isinstance(currency, str) and len(currency.strip()) == 3):
            problems.append("facts.currency must be a 3-letter code")

        flat = "\n".join(leaves(pkg, []))
        cands = set(BRACKET_RE.findall(flat))
        refs = []
        for r in raw.get("evidenceRefs") or []:
            c = canon_ref(r, cands, flat)
            if c is None:
                problems.append(f"evidence reference not found verbatim in the documents: {str(r)[:60]}")
            elif c not in refs:
                refs.append(c)
        if len(refs) != 3:
            problems.append(f"need exactly 3 distinct evidence references from the deciding paragraph (got {len(refs)})")

        if action in ACTIONS and refs:
            if not any(all(r in para for r in refs) for para in re.split(r"\n\s*\n", flat)):
                log.info("note: refs for %s are not within a single paragraph", pkg.get("packageId"))
        strong = not problems
        if action not in ACTIONS or not refs or amount is None or not isinstance(vendor, str) or not isinstance(number, str) or not isinstance(currency, str):
            return None, problems
        refs = refs[:3]
        d = {
            "action": action,
            "facts": {
                "vendorName": vendor.strip(),
                "invoiceNumber": number.strip(),
                "amountMinor": amount,
                "currency": currency.strip().upper(),
            },
            "evidenceRefs": refs,
            "rationale": fix_rationale(raw.get("rationale"), action, refs),
            "strong": strong,
        }
        return d, problems

    def _ask(self, policy, pkgs, deadline):
        """One batched call for `pkgs`; one repair round for invalid items."""
        n = len(pkgs)
        result = [None] * n  # validated decisions (strong or weak)
        pending = list(range(n))
        feedback = {}
        for attempt in range(2):
            if not pending or deadline - time.monotonic() < 6:
                break
            subset = [pkgs[i] for i in pending]
            user = self._prompt(policy, subset, [feedback.get(i) for i in pending])
            text = self.llm.complete(SYSTEM_PROMPT, user, deadline)
            try:
                parsed = parse_json_loose(text)
            except ValueError as e:
                log.error("%s", e)
                continue
            items = parsed.get("decisions") if isinstance(parsed, dict) else parsed
            if not isinstance(items, list):
                continue
            by_id = {str(it.get("packageId")): it for it in items if isinstance(it, dict)}
            still = []
            for pos, i in enumerate(pending):
                pid = str(pkgs[i].get("packageId"))
                raw = by_id.get(pid)
                if raw is None and len(items) == len(pending) and isinstance(items[pos], dict):
                    raw = items[pos]
                d, problems = self._validate(pkgs[i], raw)
                if d is not None and (result[i] is None or d["strong"]):
                    result[i] = d
                if d is None or not d["strong"]:
                    feedback[i] = "; ".join(problems) or "no decision returned"
                    still.append(i)
            pending = still
        return result

    def _prompt(self, policy, pkgs, feedback):
        payload = [{"packageId": p.get("packageId"), **{k: v for k, v in p.items() if k != "packageId"}} for p in pkgs]
        head = f"Policy revision: {policy}\nNumber of packages: {len(pkgs)}\n"
        fb = [(p.get("packageId"), f) for p, f in zip(pkgs, feedback) if f]
        if fb:
            head += "Your previous answer for these packages was invalid; fix exactly these problems:\n"
            head += "".join(f"- {pid}: {f}\n" for pid, f in fb)
        return head + "PACKAGES_JSON:\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
