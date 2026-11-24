"""Deterministic integration fixture. This is NOT an LLM or a quality benchmark."""
from __future__ import annotations

from .common import byte_len, dump


FILES = {
    "api.py": '''from orders import create_order

def checkout(request):
    key = request.headers.get("Idempotency-Key")
    if not key:
        return {"error": "missing key"}, 400
    return create_order(request.body, key)
''',
    "orders.py": '''from payments import authorize_payment

def create_order(body, idempotency_key):
    payment = authorize_payment(body["amount"], idempotency_key)
    if not payment["ok"]:
        return {"error": "payment declined"}, 402
    # Persistence is intentionally omitted in this minimal example.
    return {"order": "created", "payment_id": payment["id"]}, 201
''',
    "payments.py": '''# This in-memory cache is an example, not a production payment implementation.
_attempts = {}

def authorize_payment(amount, key):
    if key in _attempts:
        return _attempts[key]
    result = {"ok": amount > 0, "id": "example-payment"}
    _attempts[key] = result
    return result
''',
}

FACTS = {
    "api.py": ("Checkout requires Idempotency-Key and returns 400 when it is missing.", 'key = request.headers.get("Idempotency-Key")\n    if not key:\n        return {"error": "missing key"}, 400', "create_order"),
    "orders.py": ("create_order calls authorize_payment and returns 402 when payment is declined.", 'payment = authorize_payment(body["amount"], idempotency_key)\n    if not payment["ok"]:\n        return {"error": "payment declined"}, 402', "authorize_payment"),
    "payments.py": ("authorize_payment reuses the result already cached in memory for the same key.", 'if key in _attempts:\n        return _attempts[key]', "_attempts"),
}


class DemoModel:
    model = "SIMULATED: deterministic fixture, not an LLM"

    def __init__(self):
        self.calls = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0, "max_request_bytes": 0}

    def complete(self, task, system, payload, schema):
        self.calls += 1
        self.usage["max_request_bytes"] = max(self.usage["max_request_bytes"], byte_len(dump(payload)))
        if task == "plan":
            return {"queries": ["checkout", "create_order", "authorize_payment"], "unknowns": ["How is the key handled?"]}
        if task == "read":
            findings, queries = [], []
            for source in payload["evidence"]:
                fact = FACTS.get(source["path"].split("/")[-1])
                if fact and fact[1] in source["text"]:
                    findings.append({"text": fact[0], "source": source["id"], "quote": fact[1]})
                    queries.append(fact[2])
            return {"findings": findings[:6], "queries": queries[:4], "unknowns": ["Persistence and concurrency behavior have not been established."]}
        if task == "answer":
            return {"claims": [{"text": f["text"], "sources": [f["source"]]} for f in payload["findings"][:8]],
                    "unknowns": ["The demo does not implement order persistence or a distributed cache."]}
        if task == "reduce":
            # Deterministic fixture only: exercises contracts, not semantic quality.
            return {"summaries": {g["id"]: g["findings"][0]["text"] for g in payload["groups"]}}
        if task == "audit_batch":
            checks = []
            for claim in payload["claims"]:
                sources = [e for e in payload["evidence"] if e["id"] in claim["sources"]]
                supported = any(claim["text"] == text and any(quote in e["text"] for e in sources)
                                for text, quote, _ in FACTS.values())
                checks.append({"id": claim["id"], "supported": supported, "reason": "Deterministic fixture verification."})
            return {"checks": checks}
        if task == "audit":
            supported = any(payload["claim"] == text and any(quote in e["text"] for e in payload["evidence"])
                            for text, quote, _ in FACTS.values())
            return {"supported": supported, "reason": "Deterministic fixture comparison, not LLM verification."}
        raise ValueError(task)


def make_demo(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for filename, content in FILES.items():
        (directory / filename).write_text(content, encoding="utf-8")
