# Research Case File — local review

These notes document the local review before publication. The user approved this reviewed release for GitHub Pages on 27 October 2026; see the README for the current feature status.
The implementation was prepared locally for user review before the publication approval recorded above.

The design follows the portfolio: ink #111318, navy #081b41, blue #0c2d69, gray #586374, borders #d9dfe8, white and #f3f5f8 surfaces. Georgia headings and Segoe UI interface text remain unchanged.

The main composition is a left-aligned case dossier: question and recorded scope above four inspectable visuals. Comparison is the focal view; evidence and execution open progressively. The outcome river remains a real accounting of attempts. Graph positions express relationships rather than relevance or accuracy scores. All motion is user-triggered recorded playback with reduced-motion support.

```
Case / checkpoint / recorded scope
Brief | Compare | Evidence | Execution | Outcomes
Baseline memo       Workflow memo
       cited source → original report drawer
```

The two current-cohort questions have 20 attempts: 14 schema-valid outputs and six schema failures. Historical records remain outside this cohort. Case selection distinguishes recorded, failed and not-recorded states. A failed language-model attempt never silently becomes the extractive reference.

No source corpus, model output, model weights, experiment result or private credential is added or changed.

## Review locally

The current review server runs at http://127.0.0.1:5175/financial-research-workbench/ . Open the banking case with Spark and the Compare tab to start. To restart it, run `npm run dev -- --port 5175 --strictPort` from the repository.

- Compare: the actual baseline and LangGraph outputs, with each side's own cited evidence, shared passage badges and citation previews.
- Evidence: selectable/draggable finding nodes, passage-to-document relationships, accessible equivalent controls and the source drawer.
- Execution: recorded stage replay, manual steps and scrubber, measured event timings, an explicitly untimed coverage node and an optional compatible recorded-baseline overlay.
- Outcomes: each stream is a recorded attempt. Global scope accounts for20attempts:14schema-valid outputs and6failures. Per-case scope accounts for10attempts.

`npm run test:web` passes seven checks, including cohort isolation, pair compatibility, failure-vs-reference handling and URL restoration. The production build passes. Browser QA covers both featured cases, failed and evidence-only cases, source reload/Escape/focus behavior, keyboard tabs, replay, draggable findings, outcome selection and mobile containment. Narrow-screen diagrams scroll inside their own containers, with equivalent text controls.

The public artifact JSON matches the existing committed snapshot. Source changes are local and uncommitted; no push, Pages deployment, new inference or corpus ingestion has occurred. The existing published workbench remains at commit80aba8d.
