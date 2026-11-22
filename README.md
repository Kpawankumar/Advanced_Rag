# OmniRAG AI — private Mac workspace

A local document workspace with durable PDF ingestion, BM25 and dense retrieval, rank fusion, local reranking, grounded Ollama answers, document overviews and summaries, and page-linked citations.

**Open the installed app at [127.0.0.1:8000](http://127.0.0.1:8000)** or double-click [Open OmniRAG.command](Open%20OmniRAG.command). The launcher starts the built frontend and API together with owner authentication, reuses the local worker/Ollama, and binds only to loopback. The first password is in the private, ignored file `data/local/auth/owner-password.txt`; use **Change password** after signing in. See [Mac operation and recovery](docs/PHASE10.md).

Phase 12 repairs the reported product failure — ordinary questions such as *“What is in this PDF?”, “Summarize this.”, “Which internships are listed?”* returned *insufficient evidence* or unrelated excerpts. Answers are now **written from cited passages with server-verified exact quotes**: quote provenance, lexical claim support, number conservation, subject substitution and record/policy distinction are checked before an answer is shown, and every claim is labeled *Quoted* or *Explanation*. The phase adds document **overviews and summaries** (with suggested questions, key points and section notes), **“this document” resolution** and a clarification flow, **follow-up questions** with conversation memory, honest **status messages** (no relevant passage / not in the document / model unavailable / choose a document), plural-variant keyword matching, and a local **backup/restore** tool.

Phase 13 closes the four failures documented by Phase 12's validation: the dessert false answer (the app now withholds any answer that never addresses the asked attribute), the refused receipt-requirement question (requirement questions always get one bounded reconsideration), the rewritten-quote discard (claim-level quote handling plus an exact-quote prompt rule), and the scorer's overly literal phrase matching (justified subsequence fact matching with positive/negative unit tests). It also adds **rasterized page previews** (the source panel now shows the actual page rendered locally from the PDF — verified by pixel-level browser assertions), **follow-up phrasings** ("Explain that.", "What else?", "Give more detail…"), a **context reset on document switching**, plural-variant keyword matching for topic-only queries, and conflict-surfacing instructions. Phase 12's acceptance fixture became development evidence after informing these repairs; a new frozen `acceptance_v4` (8 new documents, 26 new queries) ran once. See [Phase 13 behavior](docs/PHASE13.md), [validation](docs/PHASE13_VALIDATION.md) and the [model assessment](docs/MODEL_ASSESSMENT.md).

**Measured results (authored fixtures, real local model — not independent human review):** the frozen Phase 12 acceptance fixture passed **26/26 contracts in both Hybrid+reranking and Keyword modes with zero errors**; six regression fixtures passed **61/65** with the four misses documented per case; **171 backend tests** and **12/12 real-model browser workflows** passed. All 30 documents, 50 chunks, 43 saved queries and every original PDF were hash-verified unchanged through activation and the schema-4 migration. See [Phase 12 behavior](docs/PHASE12.md), [validation and limitations](docs/PHASE12_VALIDATION.md) and the [reviewer checklist](docs/REVIEW_CHECKLIST.md).

Phase 11 remains the reference for the excerpt-selection repair history and its frozen acceptance evidence: [repair behavior](docs/PHASE11.md), [validation](docs/PHASE11_VALIDATION.md).

Phase 10 completes owner authentication, Mac-only deployment and a new frozen quality evaluation. **136 backend tests and 10 browser regressions passed**, and live access-control/data-preservation checks passed. **The new answer-quality gate did not pass:** hybrid scored 19/24 and BM25 17/24, with three unsupported questions incorrectly answered in each mode. These results supersede any impression that passing Phase 9 development cases established general accuracy. [Full validation and limitations](docs/PHASE10_VALIDATION.md).

Phase 9 adds **one guarded retry after a valid policy abstention**, using the same source excerpts and the remaining time/output-token budget. Conservative record, event and missing-name checks restrict the retry; answers still copy validated source text. The change is active locally. See the [retry guide](docs/PHASE9.md) and [observed results](docs/PHASE9_VALIDATION.md): 164/164 replay contracts, 41/41 fresh hybrid cases, 120 backend tests and 10 browser tests passed. Three BM25 no-match cases remain; these authored development cases do not establish general answer accuracy.

Phase 8 added a **saved-evidence robustness audit** for source selection. It replays fixed questions with original and reversed passage order, distinguishes changed model inputs from repeated identical inputs, and reports answer stability separately from correctness. See the [audit guide](docs/PHASE8.md) and [historical results](docs/PHASE8_VALIDATION.md): 152/164 contracts passed before the Phase 9 repair, with the same three policy abstentions in every repetition.

Phase 7 adds a **bounded evidence assessment before source selection** and a check that withholds individual-status answers when the model identifies no supporting record. It targets policy paraphrases while retaining answers copied from cited excerpts, keeps short linked conditions together, and builds the selector’s candidates in document page order. The assessment is discarded; existing documents and history remain unchanged. See the [selection guide](docs/PHASE7.md) and [observed validation](docs/PHASE7_VALIDATION.md). Keyword search remains literal; use Hybrid + reranking for semantic paraphrases.

Phase 6 adds **bounded OCR for sparse native text over a large raster image**, recovering scanned bodies that were previously hidden by a native header. It records the trigger and extraction version, retains OCR citations and existing resource limits, and applies to new ingestion or failed-document retries. Existing ready documents are preserved. See the [mixed-content OCR guide](docs/PHASE6.md) and [observed validation](docs/PHASE6_VALIDATION.md).

Phase 5 adds **answers built from selected source excerpts**. The local model chooses up to two excerpts; the server uses the same text for the answer and citation, preventing a separately written claim from being attached to an unrelated quote. New answers are labeled **Source excerpts**. Relevance, missing context, and OCR accuracy still require review. See the [answer guide](docs/PHASE5.md) and [observed comparison](docs/PHASE5_VALIDATION.md). Existing history and documents are preserved.

Phase 4 added **reproducible local answer and OCR evaluation**: authored reference text, character/word error rates, retrieval coverage, expected-answer facts, citation-page checks, unsupported questions, and separately reported challenge cases. It runs the existing worker and query endpoint in temporary storage. See the [evaluation guide](docs/PHASE4.md) and [historical baseline](docs/PHASE4_VALIDATION.md). The database schema remains version 3.

Phase 3 adds **local English OCR for scanned PDFs and PDFs mixing native-text and scanned pages**, with page-level extraction metadata and visible OCR citations. It retains Phase 2 local Qdrant, CPU embedding/reranking and the retrieval comparison. Image interpretation, graph retrieval, numerical tools and agents remain outside the implemented scope. Phase 10 adds single-owner authentication and local Mac deployment; public hosting and separate user workspaces are not implemented.

See the [Phase 3 guide](docs/PHASE3.md) and [observed Mac validation](docs/PHASE3_VALIDATION.md). Existing ready documents and saved citations are preserved; previously failed scans can be retried from the document list.

See the [Phase 2 guide](docs/PHASE2.md) for indexing, model pinning, configuration and failure behavior, and [Phase 2 validation](docs/PHASE2_VALIDATION.md) for observed results. Existing documents are automatically backfilled; source IDs and history are preserved. Choose **Keyword search**, **Semantic search**, **Hybrid search**, or **Hybrid + reranking** in the workspace.

**Scope:** one private owner on one machine. Keep the app on `127.0.0.1`; it is not configured for public hosting. Citation validation checks that source IDs exist and quotes occur in retrieved text. It does not establish semantic entailment or guarantee factual correctness.

## AI coding assistant and general chat (merged from the RAG project)

The workspace has three assistant modes, switched above the question box:

- **Knowledge Assistant** — the original grounded document workflow (citations, verification, overviews).
- **Coding Assistant** — code generation, explanation, debugging, improvement and review. Pick a task chip
  (Auto-detect/Generate/Explain/Debug/Improve/Review), optionally attach code and an error/traceback, and send.
  Answers render as language-labeled code blocks with syntax highlighting and a copy button. Generated and
  user-submitted code is **never executed** anywhere.
- **General AI** — plain conversation, independent of the documents.

**Knowledge-base-aware coding:** select a document scope in Coding Assistant mode and the request retrieves
passages from your uploaded documentation (API references, project notes, code listings) and grounds the
generated code in them, listing the passages used as sources. Leave the scope empty for general coding
questions — retrieval is not forced. Follow-up coding questions keep the last two exchanges in context.

**Model providers** (`OMNIRAG_CODE_PROVIDER`): `ollama` (default) uses the same local Ollama gateway as the
grounded path, with an optional separate `OMNIRAG_OLLAMA_CODE_MODEL`; `openrouter` enables the OpenRouter
OpenAI-compatible client ported from the RAG project (set `OMNIRAG_OPENROUTER_API_KEY`). Both stay disabled
until configured; nothing is hard-coded.

## Document and web ingestion

- **Formats:** PDF (async worker with OCR) plus **DOCX, TXT, Markdown, CSV and JSON**, which extract
  synchronously into the same chunk/FTS/vector pipeline with synthetic pages for page-linked citations.
- **Web pages:** paste an `http(s)` URL under *Add a web page*. The fetcher follows redirects manually,
  resolves DNS before connecting and rejects private, loopback, link-local and reserved addresses (SSRF guard),
  with size and time limits. Page text is indexed like a document; the source panel links back to the page.
- **Deletion:** every document row has a delete button that removes the document, its chunks, full-text
  index entries, page extractions, vector index points and stored file together.
- Empty or unparseable files fail with a visible error code instead of a silent bad index.

## Using the installed app

1. **Upload** — drag a PDF, DOCX, TXT, Markdown, CSV or JSON file onto *Add a document*,
   or paste a web page URL. The sidebar shows extraction, OCR and indexing states; a
   failed upload shows the reason with a retry button. Every source has a delete button.
2. **Overview** — click a document name to select it. The *About* card shows the
   document type, a short summary, topics and key points once generated
   (automatically for new uploads; press **Summarize document** once for older ones).
3. **Ask** — type a question, or click a suggested question. Answers cite pages;
   click a citation to open the original PDF page next to the quote.
4. **Follow up** — keep asking; the last exchange stays in context for “it”/“they”
   and stays inside the selected document.
5. **Backups** — `.venv/bin/python scripts/backup_workspace.py backup --label weekly`
   copies the workspace (database, original PDFs, indexes, auth) to
   `data/local/backups/`; `restore` puts one back. Nothing leaves this Mac.

## Limitations and roadmap

Known limits are listed in [Phase 12 validation](docs/PHASE12_VALIDATION.md): the 2B
model sometimes answers with true-but-non-responsive facts or withholds on loose
paraphrases (both fail visibly, never fabricate silently); tables and multi-column
layouts extract as plain text; charts and images are not interpreted; one model call
runs at a time. Sensible next steps, in rough value/effort order: table-aware
extraction with validated numeric answers; a mid-size local model (4–8B) if the Mac's
memory allows, measured against this fixture suite; searchable history with export;
incremental indexing and duplicate detection; multi-document comparison. None are
scheduled; each would start as its own evaluated phase.

## Development setup on Windows

Double-click [Open OmniRAG.bat](Open%20OmniRAG.bat) to start Ollama, the API, the
ingestion worker and the Vite dev server in separate windows. First-time setup:

```bat
winget install Ollama.Ollama
ollama pull qwen3-vl:2b-instruct
ollama pull qwen2.5-coder:3b
python -m venv .venv && .venv\Scripts\pip install -e .
npm --prefix frontend install
```

The UI is at http://127.0.0.1:5173 and the API at http://127.0.0.1:8000.
Close the service windows to stop the app.

## Development setup on the M1 Mac

For the installed private app, use the launcher above. The separate API/Vite commands below describe the development stack and do not enable authentication by default. Do not start them alongside the installed app. See [Phase 10](docs/PHASE10.md) for private startup, stop and recovery commands.

Prerequisites: supported macOS for Ollama (currently 14+), Python 3.11–3.13 through uv, and Node 22.12+ (Node 22 LTS recommended). If using Homebrew and the tools are absent:

```bash
brew install uv node@22 ollama tesseract
export PATH="/opt/homebrew/opt/node@22/bin:$PATH"
```

In the project root:

```bash
uv sync --frozen --python 3.11
npm --prefix frontend ci
cp -n .env.example .env
uv run python scripts/doctor.py
HF_HUB_DISABLE_TELEMETRY=1 uv run --frozen python scripts/prepare_retrieval.py
```

These are startup commands for stopped services. If the workspace is already running, open the application instead of starting another copy. An `address already in use` error means something already owns that port; inspect it before restarting. Keep only one ingestion worker. Copy the commands inside the code blocks; terminal labels are explanatory text, not shell commands.

Start native Ollama in a separate terminal with one loaded model and one concurrent request:

```bash
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NO_CLOUD=1 \
OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 \
OLLAMA_CONTEXT_LENGTH=4096 ollama serve
```

Then pull the configured model:

```bash
ollama pull qwen3-vl:2b-instruct
uv run python scripts/record_model.py
```

Keep these three processes running, all launched from the project root:

API terminal:

```bash
uv run --frozen uvicorn omnirag.main:app --host 127.0.0.1 --port 8000
```

Ingestion worker terminal:

```bash
uv run --frozen python -m omnirag.worker
```

Frontend terminal:

```bash
npm --prefix frontend run dev
```

Open **http://127.0.0.1:5173**. Upload `examples/mixed_policy.pdf`, wait for **Semantic index ready**, select that document, and ask: **How long must signed records be retained under Cedar Archives retention policy?** The scanned page 2 states **7 years**. The citation should show **OCR** and open the original page. OCR and small-model answers can contain errors; compare the quote against the image.

API documentation: **http://127.0.0.1:8000/docs**.

## Verification

```bash
uv run ruff check backend scripts
uv run pytest -q
npm --prefix frontend run build
uv run python scripts/check_http.py
```

Verify the AI coding assistant end to end (start the API first; scenarios 1-7 cover generation,
explanation, debugging, improvement, review, knowledge-base-grounded coding and follow-up context,
plus the history/readiness/deletion contracts the UI uses):

```bash
uv run python scripts/verify_coding.py --base-url http://127.0.0.1:8000
```

Run the Phase 4 evaluation from the project root, after local retrieval models are prepared:

```bash
# OCR + retrieval; answer scores explicitly remain unmeasured.
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --frozen python scripts/evaluate_quality.py

# Full evaluation using the existing local Ollama service.
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --frozen python scripts/evaluate_quality.py --with-model
```

Each run writes a new JSON report under `data/local/`. Add `--strict` to fail on core answer/citation misses or core CER > 5% / WER > 10%; current quality limitations can legitimately produce exit code 2. Execution/model failures return 1. Without `--strict`, a completed experiment may contain poor answers and still exit 0. See the guide for denominators, scoring limits and individual failure review. No app server, frontend or persistent worker is needed for this evaluation; keep Ollama running for `--with-model`.

Additional Phase 5 answer cases cover different entities, values, negation, return conditions and two-part questions:

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --frozen python scripts/evaluate_quality.py \
  --fixture evaluation/answer_regression_fixture.json --with-model --strict
```

Additional Phase 6 OCR cases use two native-header/scanned-body policies and a native control page:

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --frozen python scripts/evaluate_quality.py \
  --fixture evaluation/mixed_ocr_fixture.json --with-model --strict
```

For a manual check, upload `examples/mixed_content_policy.pdf` and ask **For how many years does Willow Records retain signed agreements?** The expected source says **13 years** on page 1, with an OCR citation.

Phase 7 compares general-policy paraphrases, unsupported personal status, and explicitly supplied individual records:

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --frozen python scripts/evaluate_quality.py \
  --fixture evaluation/paraphrase_fixture.json --with-model --strict
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --frozen python scripts/evaluate_quality.py \
  --fixture evaluation/record_support_fixture.json --with-model --strict
```

Select the procurement sample and **Hybrid + reranking**, then ask **How long until my purchased goods arrive?** The source gives the standard 14-calendar-day policy. Asking whether a particular order has arrived requires an individual record, which the sample does not contain.

With your API, worker, and Ollama running:

```bash
uv run python scripts/smoke.py --with-model
```

For browser smoke tests, also start the frontend:

```bash
cd frontend
npx playwright install chromium
npm run test:e2e
```

Include the real-model answer/citation/abstention browser gate (from `frontend`):

```bash
OMNIRAG_E2E_MODEL=1 npm run test:e2e
```

See [Phase 1 guide](docs/PHASE1.md) and [Phase 1 validation](docs/VALIDATION.md) for the historical baseline. See [architecture decisions](docs/DECISIONS.md) for the phase boundaries. Existing `.env` files are preserved: enable Phase 2 explicitly using the settings in the [Phase 2 guide](docs/PHASE2.md).

All dependencies are locked in `uv.lock` and `frontend/package-lock.json`. Model tags are mutable; `scripts/record_model.py` records the installed model digest and runtime version under the ignored local data directory.

The Phase 1 default is explicitly `qwen3-vl:2b-instruct`. The original `qwen3-vl:2b` tag selects a thinking variant that exhausted both 768- and 1,536-token budgets during Mac acceptance. Use the instruct variant with 4K context and the original 768-token output allowance. For an older `.env`, change only `OMNIRAG_OLLAMA_MODEL=qwen3-vl:2b-instruct`, pull it, record its digest, and restart the API. Keep disk space available for macOS swap and close unused heavy apps on the 8 GB Mac. `ollama stop qwen3-vl:2b-instruct` unloads idle weights without deleting them.

The included PDF is authored fictional test data. No paid model API, cloud subscription, or external database is required to run this phase. The repository's licensing policy should follow the owner's chosen GitHub repository license.
