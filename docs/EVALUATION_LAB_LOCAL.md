# LangSmith evaluation lab — local review

These notes document the local review before publication. The user approved this reviewed release for GitHub Pages on 27 October 2026; see the README for the current feature status.
The evaluation lab is embedded in the existing static React application, using the established navy/blue/white palette, Georgia headings and Segoe UI body type. It is available from the sidebar and the Case File's LangSmith tab. This was the local review state before the publication approval recorded above.

## Review

Open `http://127.0.0.1:5175/financial-research-workbench/#/evaluation?case=liquidity-2024-2025-pair` while the existing Vite preview is running.

- Dashboard: filter question, approach, model and outcome; inspect saved generation metrics and the full 5 × 4 experiment matrix. Missing failed-attempt telemetry stays missing.
- Five-way comparison: A = Qwen, B = MiMo, C = Spark, D = Ministral, E = Granite. Baseline/LangGraph is a separate experimental axis. Cards scroll horizontally on narrow screens. Each source button opens the owning model/approach's Case File citation.
- Blind review: randomize model-to-letter assignments, inspect anonymized outputs and reveal identities. A preference is session-only and is never submitted as LangSmith feedback. Failed candidates cannot receive a preference vote.
- Prompts: inspect the single source template, output contract and a reconstructed input. The original tokenizer chat input is not claimed. A candidate instruction editor downloads an explicitly unexecuted prompt experiment protocol; it cannot run inference.
- Evaluation method: inspect the four example inputs, actual evaluator code, experiment identities, source fingerprints and checkpoint lineage.

## Provenance

`web/src/data/evaluation-provenance.json` contains five previously verified experiment session IDs, the existing dataset identity, source-derived prompt template/schema and evaluator definitions. Recreate it with:

```powershell
.\.venv\Scripts\python.exe scripts/export_evaluation_showcase.py
```

The exporter makes no API/model calls. It checks that LF-normalized research/retrieval/contracts sources match the recorded Colab workflow fingerprint before allowing prompt reconstruction. Source CRLF on Windows changes byte hashes; normalizing to the original Colab LF reproduces the recorded protocol hash.

The original public research bundle, model outputs and corpus are unchanged. Runtime charts read the saved local generation fields, not a newly downloaded native feedback export. Citation checks inspect the saved verification fields corresponding to the existing evaluator. They do not measure factual accuracy.

## Native LangSmith boundary

The UI links to the verified original five-experiment comparison. Authentication stays in LangSmith; no API credential is shipped to this website. There is no iframe or claim of a supported iframe integration. Official docs describe sharing links and native comparison views; private dashboard embedding has not been verified.

Native dashboard captures, parent/child trace exports and historical Prompt Hub commits are not currently attached. The application says so explicitly. It does not synthesize native spans from our recorded stage events, fabricate prompt history or enable public sharing. These native artifact additions remain a follow-up dependent on verified exports/access.

Actual prompt A/B generation is unexecuted and remains outside the approved no-new-runs scope. Current baseline/workflow results compare retrieval/control flow using the same synthesis prompt. With two questions and one recorded observation per condition, no statistical model winner or general accuracy ranking is claimed.

## Validation

`npm run test:web` covers 13 meaningful data, provenance, route, cohort and measurement checks. `npm run build` performs TypeScript checking and the production build. Browser review checks blind/reveal behavior, failure filters, prompt reconstruction, protocol download, citation ownership, Case File embedding and mobile overflow.
