# Reader guide and named entities — local review, 2026-10-27

These notes document the local review before publication. The user approved this reviewed release for GitHub Pages on 27 October 2026; see the README for the current feature status.
This update explains the existing research for visitors without an AI background. It keeps the established portfolio colors, typography and source artifacts. This describes the pre-publication review state.

## Review paths

- `#/overview`: new default entry. Business purpose, real banking/insurance questions, interactive source-to-answer journey, all twenty attempts and what remains unproven.
- `#/models`: purpose and two research methods before the original comparison visuals. Technical model specifications are expandable. Speed/memory charts explain their axes and limits.
- `#/method`: user-oriented methodology, interactive journey, two evidence-gathering routes, current checks versus independent review, four study improvements and a glossary. Tool names come after the method.
- `#/entities`: a separate named-entity research track, with real-source editorial illustrations and five specialist candidates.
- Existing Case File and evaluation URLs continue to work. Case File labels and introductory copy explain how to interpret answers and source references.

## Important distinctions

The initial memo-writing pilot remains two questions, five models and two approaches: twenty attempts, fourteen answers in the required format and six format failures. A format pass is not an accuracy pass. Citation-ID membership is not proof that the source supports the claim. No independent quality labels were added.

The new entity track is separate from the five memo-writing models. It considers:

- `dslim/distilbert-NER` — Apache-2.0
- `dslim/bert-base-NER` — MIT
- `urchade/gliner_medium-v2.1` — Apache-2.0
- `numind/NuNER_Zero` — MIT
- `fastino/gliner2-base-v1` — Apache-2.0

Model-card links, checked on 2026-10-27, accompany the candidate cards. These models specialize in entity extraction, not in the banking/insurance domain. There are zero prediction runs and zero independently reviewed NER reference passages. Every proposed performance value stays null / Not measured.

The real BAC regulator and Travelers location passages have explicit hand-authored tags to explain the task. They are labeled editorial illustrations, not predictions or gold benchmark examples. Selecting an entity opens an explanation; selecting its source opens the corresponding Case File and original report reference.

All five candidates share a proposed PERSON/ORGANIZATION/LOCATION task. Dates, amounts and percentages form a separate extended-label track for the three flexible-label models. Unsupported categories are not applicable rather than zero scores. The downloadable protocol calls for annotation rules, independent review, document-separated holdout data, threshold tuning on development passages and exact span/type precision, recall and F1. Timing and memory must be recorded on common hardware. This protocol has not been executed.

No report ingestion, model inference, native LangSmith changes, paid services, push or publication are part of this update.
