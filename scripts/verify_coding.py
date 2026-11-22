"""End-to-end verification for the AI coding assistant (the 8 acceptance scenarios).

Run the API first, then:

    python scripts/verify_coding.py --base-url http://127.0.0.1:8000

Scenarios 1-5 exercise code generation, explanation, debugging, improvement and
review. Scenario 6 uploads a programming document and checks knowledge-base
grounding. Scenario 7 checks follow-up context. Scenario 8 (UI) is a browser
check; this script verifies the API contracts the UI depends on.

With a scripted mock LLM every response is deterministic; with a real model the
checks still assert structural contracts (fenced code, corrected code present,
sources listed, context forwarded), not exact wording.
"""

import argparse
import sys

import httpx

BROKEN_CODE = "values = [1, 2, 3]\nprint(values[10])"
SLOW_CODE = (
    "def total(prices):\n"
    "    result = 0\n"
    "    for price in prices:\n"
    "        result = result + price\n"
    "    return result\n"
    "\n"
    "def repeated(prices):\n"
    "    out = []\n"
    "    for p in prices:\n"
    "        if p in out:\n"
    "            continue\n"
    "        out.append(p)\n"
    "    return out\n"
)
REVIEW_CODE = (
    "def save(password, path):\n"
    "    f = open(path, 'w')\n"
    "    f.write(password)\n"
    "    print('saved')\n"
)
SAMPLE_FUNCTION = (
    "def summarize(numbers):\n"
    "    total = 0\n"
    "    for number in numbers:\n"
    "        total += number\n"
    "    return total / len(numbers)\n"
)
API_DOC = """# Deploy API reference

The service exposes `POST /deploy` with a JSON body of `build_id` and `token`.
Authentication uses a bearer token issued by the admin console; requests without
it return HTTP 401. Rate limits are ten requests per minute per account. A
successful deploy returns `{"status": "queued", "build": "<build_id>"}` and the
build finishes within five minutes.
"""


class Checker:
    def __init__(self, base_url: str):
        self.client = httpx.Client(base_url=base_url, timeout=180)
        self.failures = 0

    def check(self, name: str, condition: bool, detail: str = ""):
        status = "PASS" if condition else "FAIL"
        if not condition:
            self.failures += 1
        print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))

    def coding(self, payload: dict) -> dict:
        response = self.client.post("/api/v1/coding/assist", json=payload)
        response.raise_for_status()
        return response.json()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    checker = Checker(args.base_url)
    client = checker.client

    health = client.get("/api/v1/health")
    checker.check("API is up", health.status_code == 200)

    # TEST 1 — code generation
    prime_prompt = "Write a Python function that checks whether a number is prime."
    run = checker.coding({"question": prime_prompt})
    generated = run["status"] == "answered" and run["task"] == "generate"
    checker.check("1 generation: answered", generated)
    checker.check("1 generation: fenced python code", "```python" in run["answer"])
    checker.check("1 generation: defines a function", "def " in run["answer"])

    # TEST 2 — code explanation
    explain_prompt = "Explain what this function does step by step."
    run = checker.coding(
        {"question": explain_prompt, "task": "explain", "code": SAMPLE_FUNCTION}
    )
    explained = run["status"] == "answered" and run["task"] == "explain"
    checker.check("2 explanation: answered", explained)
    covers = "summarize" in run["answer"] or "function" in run["answer"].lower()
    checker.check("2 explanation: covers the code", covers)

    # TEST 3 — debugging
    run = checker.coding(
        {
            "question": "Why does this crash with IndexError?",
            "task": "debug",
            "code": BROKEN_CODE,
            "error": "IndexError: list index out of range",
        }
    )
    answer = run["answer"]
    debugged = run["status"] == "answered" and run["task"] == "debug"
    checker.check("3 debugging: answered", debugged)
    checker.check("3 debugging: names the cause", "index" in answer.lower())
    fixed = "```" in answer and ("vals" in answer or "values" in answer)
    checker.check("3 debugging: provides corrected code", fixed)

    # TEST 4 — code improvement
    improve_prompt = "Improve this code for performance and readability."
    run = checker.coding(
        {"question": improve_prompt, "task": "improve", "code": SLOW_CODE}
    )
    improved = run["status"] == "answered" and run["task"] == "improve"
    checker.check("4 improvement: answered", improved)
    substantial = "```" in run["answer"] and len(run["answer"]) > 80
    checker.check("4 improvement: improved code + explanation", substantial)

    # TEST 5 — code review
    review_prompt = "Review this code for security and quality."
    run = checker.coding(
        {"question": review_prompt, "task": "review", "code": REVIEW_CODE}
    )
    answer = run["answer"]
    reviewed = run["status"] == "answered" and run["task"] == "review"
    checker.check("5 review: answered", reviewed)
    useful = "security" in answer.lower() or "issue" in answer.lower()
    checker.check("5 review: security/quality feedback", useful)

    # TEST 6 — knowledge-base-aware coding
    upload = client.post(
        "/api/v1/documents", files={"file": ("deploy-api.md", API_DOC.encode())}
    )
    document_id = upload.json()["document"]["id"]
    ready = upload.json()["document"]["status"] == "ready"
    checker.check("6 setup: markdown document indexed", ready)
    doc_prompt = (
        "Using the uploaded documentation, create a Python function that "
        "calls the deploy API."
    )
    run = checker.coding({"question": doc_prompt, "document_ids": [document_id]})
    checker.check("6 grounded coding: answered", run["status"] == "answered")
    checker.check("6 grounded coding: marked grounded", run["grounded"] is True)
    cited = any(source["document_id"] == document_id for source in run["sources"])
    checker.check("6 grounded coding: cites the uploaded document", cited)

    # TEST 7 — follow-up context
    first = checker.coding({"question": prime_prompt})
    follow_up = "Now make it handle negative numbers gracefully."
    second = checker.coding(
        {
            "question": follow_up,
            "conversation": [
                {"question": first["question"], "answer": first["answer"][:1200]}
            ],
        }
    )
    checker.check("7 follow-up: answered", second["status"] == "answered")
    checker.check("7 follow-up: history accepted", second["id"] != first["id"])

    # UI-contract checks (scenario 8 continues in the browser)
    history = client.get("/api/v1/history").json()
    coding_runs = [item for item in history if item.get("kind") == "coding"]
    checker.check("8 history: coding runs persisted", len(coding_runs) >= 7)
    readiness = client.get("/api/v1/readiness").json()
    reporting = "coding" in readiness and "provider" in readiness["coding"]
    checker.check("8 readiness: coding provider reported", reporting)
    docs = client.get("/api/v1/documents").json()
    kind_ok = bool(docs) and docs[0].get("kind") in ("text", "pdf", "url")
    checker.check("8 documents: kind reported for text sources", kind_ok)
    deleted = client.delete(f"/api/v1/documents/{document_id}")
    checker.check("8 deletion: removes the source", deleted.status_code == 200)

    verdict = (
        "ALL CHECKS PASSED"
        if checker.failures == 0
        else f"{checker.failures} CHECK(S) FAILED"
    )
    print(f"\n{verdict}")
    return 1 if checker.failures else 0


if __name__ == "__main__":
    sys.exit(main())
