# Financial Research & Risk Intelligence Workbench

[Explore the static workbench](https://nandevers.github.io/financial-research-workbench/) · [Open the reproducible notebook in Colab](https://colab.research.google.com/github/nandevers/financial-research-workbench/blob/main/notebooks/benchmark.ipynb)

A portfolio project by Fernando Barbosa: inspect authentic bank and insurer disclosures, follow a reproducible research workflow, and compare five downloaded open models. A static React application replays **recorded real executions**. Research and inference happen beforehand in Python or Google Colab. The interface follows the [portfolio design system](docs/DESIGN_SYSTEM.md): white surfaces, navy and blue accents, Georgia headings, and Segoe UI text.

**Current status:** 30 issuer reports, 26,058 source blocks, 12 extractive investigations, and a completed five-model smoke comparison over two banking/insurance questions. The corrected profiles produced 14 schema-valid memos and 6 schema failures across 20 generation attempts. All published memo references resolved to source IDs. Claim accuracy remains unmeasured; valid citations do not establish entailment. Earlier compatibility failures and the original smoke remain preserved separately.

## Run locally

Node 22+ and Python 3.11+ are required. Windows commands:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e '.[dev]'
npm ci
.venv/Scripts/workbench.exe corpus
.venv/Scripts/workbench.exe reference
.venv/Scripts/workbench.exe export
npm run dev
```

On Linux/macOS use `.venv/bin/python` and `.venv/bin/workbench`. Ingestion caches originals under ignored `artifacts/local/raw`. Issuer endpoints can change; failed downloads are recorded in the corpus manifest. Full reports and credentials are excluded from git.

The prepared public snapshot in `web/public/data/workbench.json` can run immediately after `npm ci`, without rebuilding the corpus. Open the URL printed by Vite, including `/financial-research-workbench/`.

## The five-model comparison

Model cards and licenses checked **26 October 2026**:

- [Qwen 3.5 9B](https://huggingface.co/Qwen/Qwen3.5-9B): Apache 2.0; broad current baseline.
- [MiMo V2.6 Distill Qwen 9B](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B): MIT; fresh agent-focused checkpoint sharing Qwen's base.
- [Spark X2.6 4B](https://huggingface.co/XHToken/Spark-X2.6-4B): Apache 2.0; compact recent challenger, isolated Transformers 4.57.1 environment.
- [Ministral 3 3B Instruct](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512): Apache 2.0; compact structured-output baseline.
- [Granite 4.1 8B](https://huggingface.co/ibm-granite/granite-4.1-8b): Apache 2.0; enterprise-oriented extraction and RAG baseline.

This is a practical current checkpoint selection, not a claim that these are the five highest-ranked models worldwide. Qwen and MiMo are related: do not present them as independent architecture families. The initial larger-GPU shortlist is preserved in `docs/models-larger-gpu-20261026.json`; all five default checkpoints completed generation on a free Colab T4 in the recorded smoke test. Qwen, MiMo and Granite use NF4. Spark uses native FP16 after its custom attention failed on packed 4-bit Byte weights. Ministral dequantizes FP8 to FP16 for older GPU compatibility. All model weights are downloaded from official Hugging Face repositories, with exact commit revision and actual runtime versions recorded at execution. Spark's custom modeling code is pinned to that revision.

```powershell
.venv/Scripts/python.exe scripts/prepare_colab.py
```

The generated notebook defaults to cloning the source repository and downloading the original reports directly from issuers, avoiding browser uploads. The local zip remains an optional offline transfer package; it contains full parsed text and must not be published. Use a free T4 GPU. The smoke set covers two banking/insurance comparisons: 2 questions × 2 approaches × 5 models = **20 generation attempts**. For the full showcase, set `CASE_SET='showcase'` and `LIMIT=12` for 120 attempts. Each model is a separate GPU process; failures remain failures. Use `MODEL` for independent jobs. The runner releases only its dedicated downloaded-weight cache after each model to bound disk use. Download the results zip and import locally:

```powershell
.venv/Scripts/python.exe scripts/import_results.py workbench-results.zip
.venv/Scripts/workbench.exe export
npm run build
```

No private inference API or model-provider key is used. Optional **LangSmith** tracing/evaluation uses its own observability credential. Store `LANGSMITH_API_KEY` in Colab Secrets, grant notebook access, and set `USE_LANGSMITH=True`. It is never embedded in exported files. The free Developer plan has a trace allowance; keep the account without a payment method and avoid enabling paid features. LangSmith is a hosted product, not part of the open model license.

## Architecture

- **LangChain:** typed filing-search and Decimal calculation tools, plus a native chat-model adapter for downloaded weights.
- **LangGraph:** declared task decomposition → retrieval → institution/period coverage check → one bounded additional search when needed → structured memo → reference verification. SQLite checkpoints persist local state. Planning is deterministic to keep model comparisons controlled; this is not an autonomous model-planned tool-use claim.
- **LangSmith:** optional datasets, model experiments, workflow traces and a deterministic citation-ID evaluator. No private LLM judge is configured. Invalid source references remain measured failures; their memos are withheld from publication pending review.
- **Retrieval:** transparent TF-IDF sparse baseline with issuer, fiscal-year and strict publication-date filtering. It is not a neural embedding or semantic-reranking implementation.
- **Corpus:** original HTML/PDF reports from Bank of America, JPMorgan Chase, Chubb and Travelers. Narrative, tables, period metadata, page provenance and SHA-256 fingerprints are retained. PDFs are extracted text, not layout-perfect structured tables.
- **Presentation:** React, TypeScript, Vite, Lucide, responsive CSS. Public assets contain bounded passages and measured events; no Python server or inference endpoint is required.

## Evaluation boundaries

`evals/cases.json` contains 60 generated research task specifications; `showcase.json` selects 12. The 40/20 development/test labels are organizational only: questions overlap by theme, so they are **not an independent gold holdout benchmark**. No human gold entailment labels exist yet.

Citation-ID membership establishes that a source reference exists. It does **not** establish claim support, numerical accuracy, completeness, fair institutional comparability, or investment suitability. Those metrics remain unset. The deterministic Decimal tool supports one curated, source-checked BAC liquidity calculation after inference. This calculation is independent of model output; it does not demonstrate general model-driven table extraction. Report dates that cannot be verified are excluded from strict as-of queries. Source whitespace is normalized; PDF reading order can differ from the page.

## Verification and publication

```powershell
.venv/Scripts/python.exe -m pytest -q
npm run test:web
npm run build
```

The GitHub Pages workflow is **manual** (`workflow_dispatch`). Only project sources and the bounded public snapshot are published. GitHub Actions builds and deploys the reviewed static snapshot; subsequent releases require manually running the workflow. Default path: `https://nandevers.github.io/financial-research-workbench/`. Set `PAGES_BASE=/` for a root user-site deployment. Deployment does not launch inference or consume LangSmith traces.

See `docs/DATA_AND_LICENSES.md` for source and redistribution boundaries.

## Initial smoke findings

The original 13,965-block snapshot is preserved locally for audit. Initial execution found missing Torchvision support for multimodal processors, unsupported packed Byte weights in Spark custom attention, a Mistral tokenizer padding mismatch, and one output-token-limit JSON failure. The runtime profiles and prompt were corrected before the broader rerun. HTML parsing now captures inline-XBRL div paragraphs, increasing the corpus to 26,058 blocks. These are engineering observations, not model quality rankings.

The comparison view plots measured tokens per second against peak allocated GPU memory and opens the selected model's recorded memo. See [the documented initial failure](docs/FAILURE_ANALYSIS.md) for why valid source references alone cannot establish factual accuracy. Older source snapshots and workflow versions are excluded from the current comparison summary.

The native LangChain chat-model adapter attaches measured input/output token usage to new LLM traces. The initial smoke used a generic runnable and custom runtime feedback; generic LangSmith token columns for those older runs can remain zero. The actual token measurements are retained in each run's `generation` record.

## Recorded smoke outcomes

Qwen produced 3 schema-valid records and 1 failure; MiMo 0 and 4; Spark 4 and 0; corrected Ministral 3 and 1; Granite 4 and 0. Each checkpoint attempted the same two questions with baseline and workflow variants. These are small-sample output-contract outcomes, not general model accuracy rankings. The efficiency chart includes measured generation calls that produced invalid JSON, marked with a hollow point when no workflow memo was usable.
