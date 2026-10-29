# NER and retrieval reference sample

This file describes `data/ner-reference-sample.json`. It is a small reference sample authored from the existing financial-report corpus before any of the new specialist-model predictions were available. It is not a synthetic dataset, a human gold standard, an independent financial-accuracy benchmark, or a substitute for running the five models over all 30 documents.

## Status and completed agent review

The frozen status is `agent_reviewed_not_human_gold`. The first pass was prepared by the `plain_language_review` agent. The root agent subsequently inspected all 40 original slices and entity lists, accepted the clear labels under the written policy and adjudicated the one uncertain mention before inference. That review is recorded in the JSON's `review` object. Agent agreement must not be described as human annotation or independent professional validation.

The first pass contains:

- 40 bounded snippets, including at least one from every one of the 30 existing reports.
- 13,738 characters of exact source text; individual snippets are 98–645 characters long.
- 151 agent-reviewed clear entity mentions: 67 organizations, 62 locations and 22 people.
- One context-specific adjudication of “European Union” as LOCATION, zero unresolved mentions, and one intentionally empty-entity example.
- 10 source-grounded retrieval queries with limited relevance sets, including 9 test queries and 1 development-only query.

These are counts of annotation instances, not unique real-world entities, model predictions or performance results. The counts must be refreshed if review changes the annotations.

## Source selection and scope

The sample uses an explicit, fixed manifest of source-block IDs. The first 30 records provide document coverage; the last 10 supplement person-name examples. The passages were chosen by an agent inspecting the source, independently of model predictions. This is purposeful selection, not random sampling. Officer descriptions, certifications, regulator disclosures, geography and some table fragments are represented; many other financial-report structures are not.

Each snippet is one contiguous, unchanged slice of its source block. No sentence is rewritten or assembled from multiple locations. Some snippets exclude trailing PDF printer metadata by stopping before that metadata. Others retain extraction artifacts that occur inside the selected source slice. Slicing is not evidence that the original document itself omitted material.

The corpus fingerprint is calculated with the existing engine convention:

```python
sha256(json.dumps([(block['id'], block.get('filing_date'))
                   for block in corpus['blocks']], sort_keys=True).encode())
```

Its expected value is recorded in the JSON. Validate source-block IDs, exact slice text and the fingerprint before using the sample. The complete local corpus remains outside version control; the public reference file contains only bounded excerpts and provenance.

## Span coordinates

`source_text_start` and `source_text_end` are half-open Python Unicode code-point offsets into the original corpus block. For every record:

```python
snippet['text'] == block['text'][snippet['source_text_start']:
                                 snippet['source_text_end']]
```

Entity `start` and `end` are half-open Unicode code-point offsets relative to the snippet. `entity['text']` must equal `snippet['text'][start:end]`. They are not UTF-8 byte offsets or JavaScript UTF-16 indices. A browser display must convert offsets if an excerpt contains supplementary Unicode characters.

Do not normalize spaces, punctuation, accents or letter case after annotating. For example, a PDF spelling such as `JPMorganChase` is retained rather than silently changed to a preferred company name.

## Shared entity policy

Only PERSON, ORGANIZATION and LOCATION are in the scored task. Dates, amounts and percentages are outside this sample's shared taxonomy even when they appear in the source.

### Person names

Label explicit names, including middle initials. Exclude external titles, ages, job descriptions and possessive suffixes. Label each repeated occurrence separately. A shortened surname can be a person when the source clearly uses it as a name, but generic pronouns are not entities. Do not infer a person from a role such as “the Chairman.”

### Organizations

Include named companies, regulators, courts and identifiable named corporate units or brands. This policy includes units such as `North America Insurance` and `Global Commercial Banking`; it is a documented adaptation of the generic label, not a claim that every pretrained model uses the same policy.

Keep a full legal suffix and internal punctuation in one span, such as `JPMorgan Chase Bank, N.A.`. A following possessive suffix is outside the name. Parenthesized acronyms and aliases are separate occurrences: a regulator's full name and `FDIC` are both annotated when both appear.

Do not label generic terms such as “the Corporation,” “the Company,” “the Firm,” “Board of Directors” or an unspecified “central bank.” Laws, report forms, capital frameworks and security tickers are excluded. For example, the explicit exchange acronym `NYSE` can be an organization; `TRV` in `NYSE: TRV` is a security ticker under this policy.

### Locations

Include explicit countries, named cities, states, geopolitical abbreviations and geographic regions. A named `Middle East` region is included; adjectival `European` or `Russian` is excluded. Street addresses and postal codes are excluded. `New York` and a separately written state abbreviation `NY` are distinct mentions in an address excerpt.

Explicit `U.S.` in a geopolitical context is a location. The complement `non-U.S.` is excluded because it does not name one specific place. `U.S.` inside a regulatory metric or accounting-framework expression, such as `U.S. LCR`, is excluded from this task. These choices are potentially important sources of disagreement with a model's original training labels and should remain visible.

### Organization versus location boundaries

Use the longest, non-overlapping name. A location inside a complete organization name is not a second nested label. Thus `Central Bank of Ireland` is one organization, while a separate `Ireland` elsewhere in the sentence can be a location. The same rule applies to the complete court and securities-regulator names in the sample.

The root reviewer adjudicated `European Union` as LOCATION in the reinsurer-jurisdiction excerpt because its explicit contextual use is geographic/jurisdictional. The span remains `certainty: clear` for the agreed scoring schema and carries separate root-adjudication metadata. A generic NER model may use a different political-institution taxonomy. This local policy decision must not be presented as a universal claim. Any uncertainty introduced in a later reference version must be resolved or reported separately rather than silently becoming false positives.

## Review checklist

For each snippet, the root reviewer should:

1. Confirm the source block and exact contiguous slice.
2. Check whether every eligible name in the snippet was annotated, including repeated occurrences and acronyms.
3. Check exact boundaries, full legal suffixes and excluded titles or possessives.
4. Check organization/location ambiguity, especially nested geography and business-unit names.
5. Keep uncertainty explicit; do not resolve it by consulting model predictions.
6. Update counts and review status after any changes, then freeze the reference version before scoring.

The deliberate zero-entity paragraph contains a nationality adjective and a disease name, both outside the three-type policy. Keep it as a small false-positive check. One negative example is far too little to characterize corpus-wide false positives.

## Development and test separation

Split at the source-document level. A document's snippets must never occur in both folds. Before assignment, selected snippets are case-folded and numeric sequences are replaced with `#`; documents are joined into one cluster if any selected pair has a `difflib.SequenceMatcher` similarity of at least 0.82. This keeps recurring signature/certification excerpts together rather than treating a changed date as independent test evidence.

Initial cluster ordering was deterministic: sort clusters by the SHA-256 of `ner-reference-v1:<lexically first document ID>`, assign whole clusters to development until at least 10 documents are present, then assign the rest to test. Before inference, the root reviewer moved the complete Travelers near-duplicate document cluster to test so that every issuer appears in test. No source document or duplicate cluster was split. The frozen reference has 8 development snippets from 7 documents and 32 test snippets from 23 documents.

Development contains Bank of America and JPMorgan Chase references; test includes those issuers plus all Travelers and Chubb references. This remains a small, uneven sample rather than a per-issuer balanced benchmark. The selected-reference grouping also does not prove that every near-duplicate in the full corpus has been detected. Entity names may recur across folds in different contexts, which is expected and distinct from copying the same excerpt.

Declare model thresholds and settings before test inference. This small development sample must not be used to optimize against test results. Always report reference counts by fold and label. Primary NER scores use only the 32 test snippets; display predictions and annotation coverage over all 40 snippets separately. Do not call aggregate scores across both folds a held-out result.

## Retrieval judgments

The queries were written from the explicit source annotations, without using any model's entities or search results. Each query includes groups of literal entity mentions. A snippet is provisionally relevant when at least one clear annotation from every group occurs inside it. Some groups list aliases, for example `James Dimon` or `Jamie Dimon`; this is a declared query criterion, not an inferred entity link.

`relevant_passage_ids` identify the judged snippets. `relevant_block_ids` map those snippets to the source blocks the full-corpus search returns. `relevant_passage_ids_by_split` lets the scorer keep source-document folds separate. A query is assigned to test if a known test positive exists; otherwise it is development-only. Only the Japan/Australia query is development-only in the frozen version.

Relevance is not exhaustively labeled for all 26,058 blocks. Even a source block containing one inspected snippet may contain additional uninspected material. The judgments therefore describe the bounded inspected material, not a complete block-level or corpus-level relevance survey.

For full-corpus search, report known-positive recall at k and the reciprocal rank of the first known positive, explicitly as incomplete-judgment measurements. An unjudged result is not automatically irrelevant. Do not report ordinary precision at k by counting every unjudged hit as a false positive. A completely judged candidate-pool experiment can use a separate scoped metric, with its exact criterion and candidate count disclosed.

## What model results may establish

Actual specialist-model predictions can be compared with the frozen clear spans on these snippets using exact-boundary-and-type entity precision, recall and F1. Retain false positives, false negatives, uncertain overlaps and per-label counts. Report source-sample provenance and second-review status next to the scores.

These scores describe a small, intentionally selected reference sample. They do not prove financial understanding, factual correctness of generated memos, full-corpus extraction accuracy, or a general best model. Running inference on every block establishes processing coverage; it does not create exhaustive truth labels for every block.
