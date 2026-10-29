# Full-corpus NER and entity-assisted retrieval

This pipeline runs five downloadable checkpoints locally or on a free notebook GPU. It does not use a private model API. The existing corpus contains 30 source documents and 26,058 parsed blocks. Those counts are checked against the actual input; failed or incomplete coverage is never called complete.

```bash
python scripts/ner_colab_run.py \
  --corpus artifacts/local/corpus.json \
  --reference data/ner-reference-sample.json \
  --output artifacts/local/ner \
  --models all --device cuda --batch-size 16 \
  --resume --langsmith \
  --export web/public/data/ner-results.json
```

Omit `--langsmith` for local artifacts only. If enabled, `LANGSMITH_API_KEY` is read from the environment for observability; it is never part of an artifact or used for inference. The caller should obtain it from Colab Secrets, never put it in a notebook cell or repository. Corpus processing remains local JSONL rather than producing 130,290 individually hosted traces. Only the bounded reference sample is replayed to LangSmith as one experiment per checkpoint, with an evaluator returning exact-span precision, recall and F1. The fixed 40-passage agent-authored, separately agent-reviewed sample yields approximately 400 target/evaluator traces across five checkpoints, rather than a trace for every corpus block. It contains 8 development passages and 32 held-out test passages; those legacy experiments include both splits. They are not independent human gold.

## Reproducibility and isolation

`manifest.json` freezes each model's exact Hugging Face commit before inference. The two GLiNER-family checkpoints omit tokenizer files, so their DeBERTa backbone tokenizer/config commit is frozen separately. Small local GLiNER configuration copies point to that immutable backbone snapshot; original model weights remain pinned.

Profiles use separate virtual environments with Colab's installed torch runtime available through `--system-site-packages`. They are created without ensurepip; the base interpreter's `pip --python <target-interpreter>` installs the profile, matching the already verified Colab strategy. The Transformers adapter is pinned to 4.57.1. GLiNER Medium and NuNER use `gliner==0.2.29`; NuNER's old model-card pin requires obsolete scipy/flair dependencies and is not assumed compatible with modern Colab. An actual local load/predict smoke check must pass before corpus work. GLiNER2 uses `gliner2[local]==2.1.0`; the local extra and explicit `GLiNER2.from_pretrained` prevent accidental use of its API client. No fallback model is substituted if any loader fails.

Each model's metadata pins corpus ID/date and text-sensitive fingerprints, reference content, model/backbone revisions, main package versions, code fingerprint, labels, threshold, window overlap, batch setting and device. Resume rejects mismatches rather than mixing incompatible runs. Already complete identical runs skip repeated inference and LangSmith upload. Use a new output directory after changing the source, reference or inference configuration. `--skip-install` reuses prepared environments; `--no-isolation` is an explicit alternative for a caller-managed environment.

## Complete text coverage and failures

Tokenizer-aware overlapping windows cover every character of each source block, including long table blocks. Spans use Python Unicode character offsets, with exclusive ends, relative to the original full block. Exact duplicate predictions are removed; same-type overlapping window predictions keep the longer span, then confidence. NuNER's adjacent same-type word spans are merged with original offsets preserved. Only PERSON, ORGANIZATION and LOCATION enter the shared comparison. MISC is intentionally outside that taxonomy.

Batch OOMs or inference errors split adaptively down to one input. A failing input becomes an explicit failed block; the block receives no partial success label. All windows must succeed before the block counts as processed. Incremental JSONL includes block ID, text fingerprint, timing allocation, all spans and error records. A truncated final JSONL line from interruption can be recovered; an invalid interior record raises an error. Status is written after every block batch. Per-model completion requires all corpus blocks, all source documents, no failed blocks, reference predictions and query predictions.

Timings distinguish checkpoint download, model loading, corpus generation and allocated per-block inference. Per-block times are batch time allocated by window count, not separately measured requests. Actual exposed parameter dtypes, torch/device metadata and peak allocated GPU memory are recorded. Confidence is model-specific and is not treated as calibrated probability across checkpoints.

## Retrieval

Each model produces an inverted index over every successfully processed block: normalized entity strings, types, mention counts, document/block IDs, exact offsets and confidence. Query entities are inferred by that same pinned model. No regex-generated query entities or inferred aliases are substituted. With an empty query-entity set, the entity-assisted ranking equals the lexical ranking and explicitly reports no boost.

The lexical baseline uses TF-IDF unigram/bigram cosine similarity. Entity-assisted score is `0.75 × lexical_score + 0.25 × exact_entity_overlap`, where overlap is the fraction of distinct detected query entities present in that block. Entity strings use casefolding and whitespace normalization only. This is a declared heuristic, not a learned relevance model or proof that entities improve retrieval.

Reference retrieval judgments cover only the inspected sample. Unjudged full-corpus hits are not treated as negative. Results report recall of known positive blocks and the reciprocal rank of the first known positive, with this incomplete-judgment boundary. They do not claim full-corpus precision@k.

## Accuracy reference

The input reference file must match exact slices of the original corpus. It is agent-authored, with its review status carried into results; it is not independent human gold. Development and test passages are separated by document, including near duplicates. Primary exact-span-and-type metrics use held-out test passages only, with per-type counts. Uncertain reference entities and overlapping predictions are excluded explicitly from these scores and counted separately. Missing predictions remain visible rather than being converted into successful empty output. Scores apply to the checked sample, not all 26,058 blocks or financial research accuracy.

## Artifacts

Each `artifacts/local/ner/<model-slug>/` contains:

- `metadata.json` and `status.json`.
- `predictions.jsonl`, incremental full-corpus spans without full report text.
- `entity-index.json`, full processed-block entity postings.
- `reference-predictions.jsonl`, `evaluation.json`.
- `query-predictions.jsonl`, `retrieval.json`.
- Optional `langsmith.json` with dataset/experiment identity and no credentials.

`web/public/data/ner-results.json` is a bounded static export: all entity postings, shared document metadata and recorded retrieval results, with at most 350 representative block excerpts, each at most 900 characters. Blocks without a published snippet still have exact source/report provenance. It does not publish the full corpus or expose a browser model/API client. An interrupted job exports partial coverage after each checkpoint; missing models stay explicit.

Regenerate an existing export without loading models:

```bash
python scripts/ner_colab_run.py --export-only --resume --output artifacts/local/ner
```

Deterministic verification, without model downloads:

```bash
python -m pytest engine/tests/test_ner.py -q
```

## Separate held-out and retrieval LangSmith replay

`scripts/ner_langsmith_replay.py` is a separate post-processing helper. It reads existing, completed inference artifacts and does not load models, rerun predictions, perform retrieval, modify the frozen runner or change the reference file. Its default is local planning only:

```bash
python scripts/ner_langsmith_replay.py \
  --results artifacts/local/ner \
  --reference data/ner-reference-sample.json
```

After the five full-corpus runs finish, enable the already-authorized observability upload explicitly:

```bash
python scripts/ner_langsmith_replay.py \
  --results artifacts/local/ner \
  --reference data/ner-reference-sample.json \
  --upload
```

The helper requires only `LANGSMITH_API_KEY` from the environment for LangSmith. It has no credential command-line argument and never prints or stores the key or raw network exception text. It does not use any model inference API. In Colab, provide the same existing secret through the environment, then run this helper after inference without restarting the runtime. `--tasks ner` or `--tasks retrieval` can replay one series separately. Default `--tasks ner,retrieval` replays both.

It creates two clearly distinct datasets:

- **NER held-out:** only the 32 test passages, with five saved-output experiments. The eight development passages are excluded. Per-passage exact precision/recall/F1 feedback averages are **macro** scores. Separate experiment-level `heldout_entity_micro_precision`, `heldout_entity_micro_recall` and `heldout_entity_micro_f1` pool entity TP/FP/FN across the held-out passages. These quantities must not be described as the same aggregation.
- **Retrieval with labelled splits:** all ten saved queries, with five checkpoints × two recorded approaches, for ten experiments. Nine query inputs are test, one is development. Test feedback uses `test_*` keys and development feedback uses `development_*` keys. Experiment summaries average the nine test queries only. A test query's known positives are restricted to reference blocks in its test fold; development positives are likewise restricted to their fold. Full-corpus rankings are preserved, and unjudged hits never become negatives. The scores describe recall of the inspected known positives and first-known-positive reciprocal rank, rather than exhaustive retrieval quality or precision@k.

The default two series produce 160 held-out NER replay runs plus 100 retrieval replay runs: **260 hosted target traces and zero additional evaluator traces**. Feedback is computed deterministically from saved outputs. Native replay timestamps describe upload bookkeeping, not model latency. Existing local timing fields remain separately identified in outputs. No new GPU work occurs.

Datasets, example IDs, experiment names, replay run IDs and feedback IDs derive from stable fingerprints. A local checkpoint is saved after each feedback/run. After interruption, the helper checks existing deterministic server IDs for unfinished work and resumes without duplicating records. It refuses mismatched corpus, reference, checkpoint revision, text hashes or incomplete full-corpus coverage. Saved JSONL is read-only; a truncated tail is rejected rather than repaired by the replay helper.

### Rate-limited replay resume

LangSmith Cloud documents a limit of **30 `GET /runs/:id` lookups per minute per service key or PAT**. This is separate from run creation and feedback limits. The helper's `Client.read_run` checks can reach this limit even though replay creates only 260 target traces. See the official [Cloud rate limits](https://docs.langchain.com/langsmith/cloud#rate-limits) and [administration rate limits](https://docs.langchain.com/langsmith/administration-overview#rate-limits), checked on 29 October 2026.

The current helper does **not** pace these lookups. For an interrupted upload, keep the same helper file, inputs and `langsmith-replay/checkpoint.json`; a runtime-only wrapper can space lookup starts by **2.25 seconds**, approximately 27 explicit lookups per minute. This leaves the saved-output fingerprints and deterministic dataset, experiment, run and feedback IDs unchanged. It performs no model inference. SDK-internal retries and other clients sharing the key can still consume endpoint capacity.

First verify that the previous replay process is idle. Respect any `Retry-After` hint; otherwise allow a full minute after a lookup-throughput 429 before resuming. Then run this launcher in a **fresh process**, from the repository root, using the same Python environment and existing environment secret as the previous replay. In a Colab shell cell, put `%%bash` before the example; replace `python` with the previously used interpreter if necessary. Do not print the key, request headers or raw exception text.

```bash
python - <<'PY'
import runpy
import sys
import threading
import time
from functools import wraps
from pathlib import Path
from langsmith import Client

original_read_run = Client.read_run
lookup_lock = threading.Lock()
next_lookup_start = 0.0

@wraps(original_read_run)
def paced_read_run(self, *args, **kwargs):
    global next_lookup_start
    with lookup_lock:
        delay = max(0.0, next_lookup_start - time.monotonic())
        if delay:
            time.sleep(delay)
        next_lookup_start = time.monotonic() + 2.25
    return original_read_run(self, *args, **kwargs)

Client.read_run = paced_read_run
helper = Path('scripts/ner_langsmith_replay.py')
sys.argv = [
    str(helper),
    '--results', 'artifacts/local/ner',
    '--reference', 'data/ner-reference-sample.json',
    '--state-directory', 'artifacts/local/ner/langsmith-replay',
    '--upload',
]
runpy.run_path(str(helper), run_name='__main__')
PY
```

Do not run concurrent replay launchers. A continuing 429 may instead reflect an hourly, monthly or configured usage limit; pacing does not remove those limits. Preserve the checkpoint and wait for the applicable reset while keeping the free account. The helper records a sanitized error type rather than raw request details.

Resume observability with this separate replay helper, **never by restarting the five-model inference orchestrator against restored successful outputs**. Source/package identity checks happen before the worker's completed-run skip, and a failed orchestration attempt can replace a successful status with an error. Preserve completed model directories and their checksums; use a targeted worker or the audited recovery helper only for genuinely incomplete inference. `--export-only` regenerates the static bundle without invoking workers.

A launcher start, partial progress log, unexecuted packaging cell or browser download wait is not completion evidence. Verify the actual replay status and native records, then obtain and checksum the complete private archive and pass strict local import before publication. This procedure does not establish that the final replay or archive has been verified.

New files are written only under `artifacts/local/ner/langsmith-replay/` by default:

- `plan.json`: eligible models, held-out inputs, saved outputs, metrics and target-run counts; this is a local artifact, not a public corpus export.
- `checkpoint.json`: resumable dataset/experiment/run/feedback identities.
- `status.json`: actual native dataset IDs, experiment session IDs, authenticated comparison links, review provenance, summaries and completion/error type. Links remain private unless the user separately authorizes public sharing.

The replay does not overwrite the worker's older per-model `langsmith.json` files. When importing the new status into the static showcase, merge each matching model entry into an optional `model.langsmith.replay` field and preserve the legacy experiment provenance. Do not claim replay links exist until an actual successful upload has produced their IDs.

Pure replay tests need no key, network or models:

```bash
python -m pytest engine/tests/test_ner_langsmith_replay.py -q
```

## Recorded Kaggle continuation and backup gates

The final GLiNER2 corpus artifacts were produced on 29 October 2026 using a free Kaggle Tesla T4. The first four completed Colab checkpoints were restored from the verified fourth checkpoint archive; all 36 files remained byte-for-byte unchanged. Kaggle used the original model revision, worker, engine, corpus, manifest, threshold and window settings. Its recorded Torch build is 2.11.0+cu128; this runtime difference is retained in the fifth model metadata rather than presented as identical to Colab.

`scripts/prepare_ner_kaggle.py` prepares a private transfer bundle and a notebook with separate inspection, restore, GLiNER2-only setup and worker cells. Do not use Run All or restart the five-model orchestrator against restored successes. No LangSmith upload occurs during this fifth-only inference. The Kaggle account remained card-free and used only free quota.

Immediately after the fifth pass, the raw archive was downloaded and checked with `scripts/verify_ner_backup.py` before recovery. The audited helper then retried only the 20 failed blocks. The recovered archive was downloaded and checked separately, including all original predictions, frozen sources, recovery sources, native spans and recovery identity. The immediate verifier checks SHA-256, ZIP CRC, complete block identities and preservation of the first four models; it does not replace the final strict importer.

Kaggle's persistent saved-version Output file download worked. Saving a notebook defaults to Save & Run All: select **Quick Save** and explicitly save its outputs instead. A saved version is a backup operation, not evidence of a new inference run. Files in a live notebook alone are not a verified local backup.

Once the recovered backup is verified, export, deterministic LangSmith replay, strict import and static-site publication run on CPU. No GPU or model reload is needed for those steps.
