# Financial Research & Risk Intelligence Workbench

[Explore the static workbench](https://nandolab.github.io/financial-research-workbench/) · [Open the reproducible notebook in Colab](https://colab.research.google.com/github/nandolab/financial-research-workbench/blob/main/notebooks/benchmark.ipynb)

A portfolio project by Fernando Barbosa. **Can openly available AI make financial reports easier to understand while showing where its answers come from?** This project brings together real bank and insurer reports, retrieves relevant passages and asks models to draft short, source-linked explanations that a reader can inspect. A static React application replays **recorded real executions**. Research and inference happen beforehand in Python or Google Colab. The interface follows the [portfolio design system](docs/DESIGN_SYSTEM.md): white surfaces, navy and blue accents, Georgia headings, and Segoe UI text.

**Current result:** 30 original reports, 26,058 searchable passages and 12 source-only investigations. Five answer-writing models each tried two banking/insurance questions through two research methods, producing **20 attempts: 14 answers in the required format and 6 format failures**. This is a small pilot, not a factual-accuracy ranking. All published memo references resolved to supplied source IDs, which does not establish that those sources support every claim. Earlier compatibility failures and historical runs remain separate.

## Explore the research

- [Start here](https://nandolab.github.io/financial-research-workbench/#/overview): the business goal, actual pilot results and a clickable question-to-source-to-answer journey.
- [Research cases](https://nandolab.github.io/financial-research-workbench/#/research/liquidity-2024-2025-pair): paired answers, source maps, recorded execution steps and failures.
- [Model comparison](https://nandolab.github.io/financial-research-workbench/#/models): understand simple search versus guided research before exploring the results.
- [Evaluation lab](https://nandolab.github.io/financial-research-workbench/#/evaluation?case=liquidity-2024-2025-pair): five-way A–E comparison, session-only blind review, recorded generation metrics, source prompt and evaluator inspection. Native LangSmith records open separately and require account access; no credential is shipped to the site.
- [Named entities](https://nandolab.github.io/financial-research-workbench/#/entities): specialist extraction and entity-assisted retrieval, with measured views activated only by a verified complete five-model result export; source illustrations are explicitly editorial when no measured bundle is included.
- [How it works](https://nandolab.github.io/financial-research-workbench/#/method): plain-language methodology, a glossary, study limitations and the next improvements.

The guided route performs fixed related searches and checks company/year evidence coverage before a bounded round of targeted gap searches. Both routes use the same writing prompt. Candidate prompt tests remain unexecuted drafts.

## Named-entity research track

Named-entity recognition finds the exact names of people, organizations and places inside a passage. It is separate from generating a research answer. The five models used are [DistilBERT NER](https://huggingface.co/dslim/distilbert-NER), [BERT NER](https://huggingface.co/dslim/bert-base-NER), [GLiNER Medium v2.1](https://huggingface.co/urchade/gliner_medium-v2.1), [NuNER Zero](https://huggingface.co/numind/NuNER_Zero) and [GLiNER2 Base](https://huggingface.co/fastino/gliner2-base-v1). Model cards and MIT/Apache-2.0 licenses were checked on 27 October 2026. These are extraction specialists, not finance-trained models.

**Offline inference completed on 29 October 2026.** All five specialists processed every one of the 30 reports and 26,058 parsed source passages. The first four retained their completed free Colab GPU outputs; the final GLiNER2 artifacts were produced on a free Kaggle T4 after the earlier Colab filesystem expired. The saved runs contain actual entity predictions, separate model indexes and word-matching versus entity-assisted retrieval results. Their final processing records report zero failed corpus passages, reference snippets or queries. Complete processing establishes coverage; it does not establish correct recognition of every name or reliable financial conclusions.

GLiNER2's first pass completed 26,038 passages and failed on 20. Its text-preparation library adds a final period to some inputs; an entity could extend into that added punctuation beyond the original passage. A targeted rerun of those 20 passages used an audited mapping back to the source text. The original prediction history and base metadata were preserved, with a separate recovery identity and native-output audit for the appended replacements. The first four models' outputs and GLiNER2's reference/query predictions were unchanged.

**Validated result release:** strict local checks recomputed all 130,290 corpus records, five indexes, reference scores, retrieval rankings and recovery provenance. The separate CPU saved-output replay completed 260 runs across 15 experiments; one hosted run from every experiment was read back and matched to its saved inputs and outputs. The measured interface includes all five searchable indexes, the 32-snippet held-out tagging scores and actual lexical versus entity-assisted comparisons. Desktop and mobile checks cover model selection, filters, source highlights and empty results. Raw and recovered archives remain private, with independently verified local backups.

**What we learned:** GLiNER Medium v2.1 led exact-span tagging on the 32 held-out snippets: 84.7% F1, 78.3% precision and 92.3% recall. GLiNER2 reached 78.3% F1. Better tagging did not guarantee better retrieval. Over nine test queries, BERT gave the largest mean known-positive recall@5 increase, from 16.7% to 22.2% (+5.6 percentage points); only one query gained recall. DistilBERT reached 20.4%, while GLiNER Medium, NuNER and GLiNER2 retained the 16.7% baseline recall. NuNER improved first-hit rank in one query. These are descriptive results on a small agent-reviewed reference, not statistical significance or general financial accuracy.

The findings use the same held-out-fold policy as the completed LangSmith replay: development queries and development positive passages are excluded. Original all-judgment per-query metrics remain separately labelled. `scripts/summarize_ner_results.py` rebuilds the explanation data from saved outputs and checks it against the completed replay summaries; it performs no inference or hosted writes.

**NER results are gated by the actual exported artifact.** The site displays measured extraction and retrieval only when `web/public/data/ner-results.json` identifies all five unique expected models, each complete over all 30 documents and 26,058 source blocks, with the matching corpus and reviewed reference evaluation. While that file is loading the site shows a loading state; network, malformed or incomplete exports show an explicit error. A missing file means measured results are not included in that build, rather than proving that no inference has occurred elsewhere.

The frozen reference contains 40 bounded snippets authored by one agent and reviewed by a second before predictions. Its primary exact-span-and-type entity scores pool counts over the 32 held-out test snippets; eight development snippets are excluded. This is **agent-reviewed reference annotation, not human gold, independent professional validation or full-corpus accuracy**. The original Colab LangSmith upload includes all 40 examples with per-example feedback, so native averages across development and test can differ from the site's held-out micro-average. Native dataset/experiment links and upload status are shown only when their identities are actually exported; absent links are not invented. If separately recorded replay artifacts are attached, their held-out NER experiments contain 32 test examples, and retrieval experiments contain all 10 queries with summaries restricted to the nine test queries. The site keeps those records distinct from the original 40-example upload.

The offline runner saves resumable predictions and records the pinned model revisions, runtime and processing status. The private Kaggle continuation reused the frozen source archive and first four checkpoints, ran only GLiNER2, and saved checksum-verified raw and recovered backups before replay. The included results passed strict local import and provenance checks before publication. A site build without a verified complete result artifact presents explicitly editorial illustrations and the comparison setup with no fabricated scores. No model weights are downloaded and no inference runs in the website.

## Run locally

For the static explorer, Node 22+ is sufficient. The committed public snapshot runs without Python, model weights or API keys:

```powershell
npm ci
npm run dev
```

Open the URL printed by Vite, including `/financial-research-workbench/`. To rebuild the research corpus and offline engine, use Python 3.11+ as well. Windows commands:

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
- **LangGraph:** declared task decomposition → retrieval → institution/period coverage check → one bounded round of additional searches, one per missing institution/year pair → structured memo → reference verification. SQLite checkpoints persist local state. Planning is deterministic to keep model comparisons controlled; this is not an autonomous model-planned tool-use claim.
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

The GitHub Pages workflow is **manual** (`workflow_dispatch`). Only project sources and the bounded public snapshot are published. GitHub Actions builds and deploys the reviewed static snapshot; subsequent releases require manually running the workflow. Default path: `https://nandolab.github.io/financial-research-workbench/`. Set `PAGES_BASE=/` for a root user-site deployment. Deployment does not launch inference or consume LangSmith traces.

See `docs/DATA_AND_LICENSES.md` for source and redistribution boundaries.

## Initial smoke findings

The original 13,965-block snapshot is preserved locally for audit. Initial execution found missing Torchvision support for multimodal processors, unsupported packed Byte weights in Spark custom attention, a Mistral tokenizer padding mismatch, and one output-token-limit JSON failure. The runtime profiles and prompt were corrected before the broader rerun. HTML parsing now captures inline-XBRL div paragraphs, increasing the corpus to 26,058 blocks. These are engineering observations, not model quality rankings.

The comparison view plots measured tokens per second against peak allocated GPU memory and opens the selected model's recorded memo. See [the documented initial failure](docs/FAILURE_ANALYSIS.md) for why valid source references alone cannot establish factual accuracy. Older source snapshots and workflow versions are excluded from the current comparison summary.

The native LangChain chat-model adapter attaches measured input/output token usage to new LLM traces. The initial smoke used a generic runnable and custom runtime feedback; generic LangSmith token columns for those older runs can remain zero. The actual token measurements are retained in each run's `generation` record.

## Recorded smoke outcomes

Qwen produced 3 schema-valid records and 1 failure; MiMo 0 and 4; Spark 4 and 0; corrected Ministral 3 and 1; Granite 4 and 0. Each checkpoint attempted the same two questions with baseline and workflow variants. These are small-sample output-contract outcomes, not general model accuracy rankings. The efficiency chart includes measured generation calls that produced invalid JSON, marked with a hollow point when no workflow memo was usable.
