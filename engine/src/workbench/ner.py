"""Offline named-entity corpus processing and transparent entity-assisted retrieval.

Model libraries are deliberately absent from module imports: deterministic contracts,
offset handling, retrieval and scoring can be checked without downloading a checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
LABELS = ('PERSON', 'ORGANIZATION', 'LOCATION')
MODEL_SPECS = [
    {'id': 'dslim/distilbert-NER', 'name': 'DistilBERT NER', 'license': 'Apache-2.0', 'profile': 'transformers', 'kind': 'token'},
    {'id': 'dslim/bert-base-NER', 'name': 'BERT NER', 'license': 'MIT', 'profile': 'transformers', 'kind': 'token'},
    {'id': 'urchade/gliner_medium-v2.1', 'name': 'GLiNER Medium v2.1', 'license': 'Apache-2.0', 'profile': 'gliner', 'kind': 'gliner', 'backbone': 'microsoft/deberta-v3-base'},
    {'id': 'numind/NuNER_Zero', 'name': 'NuNER Zero', 'license': 'MIT', 'profile': 'gliner', 'kind': 'nuner', 'backbone': 'microsoft/deberta-v3-large'},
    {'id': 'fastino/gliner2-base-v1', 'name': 'GLiNER2 Base', 'license': 'Apache-2.0', 'profile': 'gliner2', 'kind': 'gliner2'},
]
PROFILE_REQUIREMENTS = {
    'transformers': ['transformers==4.57.1', 'huggingface-hub>=0.34,<1', 'sentencepiece>=0.2,<0.3', 'scikit-learn>=1.5,<2', 'langsmith>=0.4,<1'],
    'gliner': ['gliner==0.2.29', 'transformers==4.57.1', 'huggingface-hub>=0.34,<1', 'sentencepiece>=0.2,<0.3', 'scikit-learn>=1.5,<2', 'langsmith>=0.4,<1'],
    'gliner2': ['gliner2[local]==2.1.0', 'transformers==4.57.1', 'huggingface-hub>=0.34,<1', 'sentencepiece>=0.2,<0.3', 'scikit-learn>=1.5,<2', 'langsmith>=0.4,<1'],
}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def slug(model_id):
    return model_id.replace('/', '--')


def corpus_fingerprints(corpus):
    # Preserve the research workbench's original ID/date fingerprint as well as a
    # stronger text-sensitive fingerprint for exact resumption of this new task.
    snapshot = hashlib.sha256(json.dumps([(b['id'], b.get('filing_date')) for b in corpus['blocks']], sort_keys=True).encode()).hexdigest()
    content = digest({'documents': [(d['id'], d.get('sha256')) for d in corpus['documents']], 'blocks': [(b['id'], b['text'], b.get('filing_date')) for b in corpus['blocks']]})
    return {'corpus_snapshot_sha256': snapshot, 'corpus_content_sha256': content}


def token_windows(text, tokenizer, max_tokens=320, overlap_tokens=48):
    """Cover the complete Unicode string with overlapping tokenizer-aware windows.

    Offsets are character/codepoint offsets in the original block. Whitespace-only
    regions are retained by extending window boundaries, never silently discarded.
    """
    if max_tokens < 8 or not 0 <= overlap_tokens < max_tokens:
        raise ValueError('Require max_tokens>=8 and 0<=overlap_tokens<max_tokens')
    encoded = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    offsets = [(int(a), int(b)) for a, b in encoded['offset_mapping'] if b > a]
    if not offsets:
        return [{'start': 0, 'end': len(text), 'text': text, 'tokens': 0}]
    windows = []
    step = max_tokens - overlap_tokens
    for i in range(0, len(offsets), step):
        last = min(i + max_tokens, len(offsets))
        start = 0 if i == 0 else offsets[i][0]
        end = len(text) if last == len(offsets) else offsets[last][0]
        windows.append({'start': start, 'end': end, 'text': text[start:end], 'tokens': last-i})
        if last == len(offsets):
            break
    if windows[0]['start'] != 0 or windows[-1]['end'] != len(text):
        raise AssertionError('Window coverage is incomplete')
    if any(a['end'] < b['start'] for a, b in zip(windows, windows[1:])):
        raise AssertionError('Gap between text windows')
    return windows


def normalize_type(value):
    value = str(value).upper().removeprefix('B-').removeprefix('I-')
    return {'PER': 'PERSON', 'PERSON': 'PERSON', 'ORG': 'ORGANIZATION', 'ORGANISATION': 'ORGANIZATION', 'ORGANIZATION': 'ORGANIZATION', 'LOC': 'LOCATION', 'GPE': 'LOCATION', 'LOCATION': 'LOCATION'}.get(value)


def normalize_entities(text, entities, offset=0):
    results = []
    for entity in entities:
        label = normalize_type(entity.get('type') or entity.get('label') or entity.get('entity_group'))
        if label is None:
            continue  # MISC is intentionally outside the common task.
        start, end = int(entity['start']), int(entity['end'])
        if not 0 <= start < end <= len(text):
            raise ValueError(f'Entity offset outside its input: {start}:{end}/{len(text)}')
        score = float(entity.get('score', entity.get('confidence', 0.0)))
        if not math.isfinite(score):
            raise ValueError('Non-finite entity confidence')
        results.append({'start': start+offset, 'end': end+offset, 'text': text[start:end], 'type': label, 'score': round(score, 6)})
    return results


def merge_entities(text, entities, merge_adjacent=False):
    unique = {}
    for entity in entities:
        key = (entity['start'], entity['end'], entity['type'])
        if key not in unique or entity['score'] > unique[key]['score']:
            unique[key] = entity
    # Overlapping windows can emit a clipped and a complete version of the same
    # entity. Keep the longer same-type interval (confidence breaks exact ties).
    retained = []
    for entity in sorted(unique.values(), key=lambda e: (-(e['end']-e['start']), -e['score'], e['start'])):
        if not any(entity['type']==old['type'] and entity['start']<old['end'] and entity['end']>old['start'] for old in retained):
            retained.append(entity)
    ordered = sorted(retained, key=lambda e: (e['start'], e['end'], e['type']))
    if not merge_adjacent:
        return ordered
    merged = []
    for entity in ordered:
        if merged and merged[-1]['type'] == entity['type'] and 0 <= entity['start']-merged[-1]['end'] <= 1 and not text[merged[-1]['end']:entity['start']].strip():
            old = merged[-1]
            old.update(end=entity['end'], text=text[old['start']:entity['end']], score=min(old['score'], entity['score']))
        else:
            merged.append(dict(entity))
    return merged


def append_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        handle.flush()


def read_jsonl(path, recover_tail=False):
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    records, valid_bytes = [], 0
    for i, line in enumerate(lines):
        try:
            records.append(json.loads(line))
            valid_bytes += len(line)
        except (ValueError, UnicodeDecodeError):
            if recover_tail and i == len(lines)-1 and not line.endswith(b'\n'):
                with path.open('r+b') as handle:
                    handle.truncate(valid_bytes)
                break
            raise ValueError(f'Invalid JSONL record at {path}:{i+1}')
    return records


def latest_records(path):
    return {r['id']: r for r in read_jsonl(path, recover_tail=True)}


def enforce_resume(output, metadata):
    path = Path(output) / 'metadata.json'
    if path.exists():
        old = load_json(path)
        if old['configuration_sha256'] != metadata['configuration_sha256']:
            raise ValueError('Refusing resume: corpus, model revision, code or inference configuration changed. Choose a new output directory.')
    else:
        save_json(path, metadata)


def normalized_name(text):
    return re.sub(r'\s+', ' ', text).strip().casefold()


def entity_index(corpus, records):
    blocks = {b['id']: b for b in corpus['blocks']}
    entries = {}
    for record in records.values():
        if record.get('status') != 'complete':
            continue
        block = blocks[record['id']]
        for e in record['entities']:
            key = normalized_name(e['text'])
            if not key:
                continue
            entry = entries.setdefault(key, {'text': e['text'], 'normalized': key, 'types': set(), 'mentions': [], 'document_ids': set(), 'block_ids': set()})
            entry['types'].add(e['type'])
            entry['document_ids'].add(block['document_id'])
            entry['block_ids'].add(block['id'])
            entry['mentions'].append({'block_id': block['id'], 'document_id': block['document_id'], 'start': e['start'], 'end': e['end'], 'type': e['type'], 'score': e['score']})
    for entry in entries.values():
        entry['types'] = sorted(entry['types'])
        entry['document_ids'] = sorted(entry['document_ids'])
        entry['block_ids'] = sorted(entry['block_ids'])
        entry['mention_count'] = len(entry['mentions'])
    return sorted(entries.values(), key=lambda e: (-e['mention_count'], e['normalized']))


class EntityRetrieval:
    """Measured lexical baseline plus a declared exact-entity overlap boost."""
    def __init__(self, corpus, entries):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.blocks = corpus['blocks']
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=45000, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(b['text'] for b in self.blocks)
        self.positions = {b['id']: i for i, b in enumerate(self.blocks)}
        self.postings = {e['normalized']: set(e['block_ids']) for e in entries}

    def rank(self, query, query_entities, k=5, allowed=None, entity_weight=.25):
        lexical = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        keys = sorted({normalized_name(e['text']) for e in query_entities})
        counts = Counter(block_id for key in keys for block_id in self.postings.get(key, set()))
        eligible = [(float(lexical[i]), i) for i, b in enumerate(self.blocks) if (allowed is None or b['id'] in allowed) and (lexical[i] > 0 or counts[b['id']])]
        baseline = sorted((x for x in eligible if x[0] > 0), key=lambda x: (-x[0], self.blocks[x[1]]['id']))
        hybrid = sorted(eligible, key=lambda x: (-((1-entity_weight)*x[0]+entity_weight*counts[self.blocks[x[1]]['id']]/len(keys)) if keys else -x[0], self.blocks[x[1]]['id']))
        def output(ranking, boosted):
            rows = []
            for lexical_score, i in ranking[:k]:
                block = self.blocks[i]
                entity_score = counts[block['id']]/len(keys) if keys else 0.0
                score = ((1-entity_weight)*lexical_score+entity_weight*entity_score) if boosted and keys else lexical_score
                rows.append({'block_id': block['id'], 'document_id': block['document_id'], 'score': round(score, 6), 'lexical_score': round(lexical_score, 6), 'entity_overlap': round(entity_score, 6)})
            return rows
        return {'lexical': output(baseline, False), 'entity_assisted': output(hybrid, True), 'query_entity_keys': keys, 'entity_boost_applied': bool(keys)}


def validate_reference(corpus, reference):
    fingerprints = corpus_fingerprints(corpus)
    if reference.get('corpus_snapshot_sha256') != fingerprints['corpus_snapshot_sha256']:
        raise ValueError('Reference sample has a different corpus snapshot')
    blocks = {b['id']: b for b in corpus['blocks']}
    ids = set()
    for passage in reference['passages']:
        if passage['id'] in ids:
            raise ValueError('Duplicate reference passage ID')
        ids.add(passage['id'])
        block = blocks[passage['source_block_id']]
        start, end = passage['source_text_start'], passage['source_text_end']
        if block['text'][start:end] != passage['text'] or block['document_id'] != passage['document_id']:
            raise ValueError(f'Reference snippet is not its original source slice: {passage["id"]}')
        for e in passage['entities']:
            if e['type'] not in LABELS or passage['text'][e['start']:e['end']] != e['text']:
                raise ValueError(f'Invalid reference entity offset/type: {passage["id"]}')


def span_counts(expected, predicted):
    gold = {(e['start'], e['end'], normalize_type(e.get('type', e.get('label')))) for e in expected if e.get('certainty', 'clear') == 'clear'}
    uncertain = [(e['start'], e['end']) for e in expected if e.get('certainty', 'clear') != 'clear']
    guesses = {(e['start'], e['end'], normalize_type(e.get('type', e.get('label')))) for e in predicted if not any(e['start'] < end and e['end'] > start for start, end in uncertain)}
    return {'tp': len(gold & guesses), 'fp': len(guesses-gold), 'fn': len(gold-guesses), 'uncertain_reference_entities': len(uncertain)}


def scores(counts):
    tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
    precision = tp/(tp+fp) if tp+fp else (1.0 if not fn else 0.0)
    recall = tp/(tp+fn) if tp+fn else (1.0 if not fp else 0.0)
    return {**counts, 'precision': precision, 'recall': recall, 'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.0}


def evaluate_reference(reference, predictions):
    totals = Counter(tp=0, fp=0, fn=0, uncertain_reference_entities=0)
    per_type = {label: Counter(tp=0, fp=0, fn=0) for label in LABELS}
    rows = []
    for passage in reference['passages']:
        prediction = predictions.get(passage['id'])
        if not prediction or prediction.get('status') != 'complete':
            rows.append({'passage_id': passage['id'], 'status': 'missing_prediction'})
            continue
        counts = span_counts(passage['entities'], prediction['entities'])
        if passage.get('split') == 'test':
            totals.update(counts)
            for label in LABELS:
                expected=[e for e in passage['entities'] if e['type'] == label or e.get('certainty','clear') != 'clear']
                per_type[label].update(span_counts(expected, [e for e in prediction['entities'] if e['type'] == label]))
        rows.append({'passage_id': passage['id'], 'split': passage.get('split'), 'status': 'scored', **scores(counts)})
    test_rows = [r for r in rows if r.get('split') == 'test' and r['status'] == 'scored']
    reviewed = reference.get('review',{}).get('second_review_status') == 'completed'
    boundary = ('Agent-authored and separately agent-reviewed reference sample; not human gold. ' if reviewed else 'Agent-authored reference sample; second-agent review not completed. Not human gold. ') + 'Scores apply only to this sample, not all corpus blocks.'
    return {'schema_version': SCHEMA_VERSION, 'reference_status': reference.get('status'), 'reference_review': reference.get('review'), 'boundary': boundary, 'test_passages': len(test_rows), 'missing_predictions': sum(r['status'] != 'scored' for r in rows), 'metrics': scores(totals) if test_rows else None, 'per_type': {k: scores(v) for k, v in per_type.items()} if test_rows else {}, 'passages': rows}


def known_positive_retrieval(query, ranking, k=5):
    relevant = set(query.get('relevant_block_ids', []))
    hits = [row['block_id'] for row in ranking[:k]]
    found = relevant.intersection(hits)
    return {'known_positive_recall_at_k': len(found)/len(relevant) if relevant else None, 'first_known_positive_reciprocal_rank': next((1/(i+1) for i, b in enumerate(hits) if b in relevant), 0.0) if relevant else None, 'known_relevant_blocks': len(relevant), 'known_hits': len(found), 'k': k, 'boundary': 'Incomplete relevance judgments. Unjudged full-corpus hits are not treated as negative; this is not precision@k.'}


def export_bundle(corpus, manifest, output, destination, excerpt_limit=350, excerpt_chars=900):
    """Public entity postings cover every processed block; only bounded text is exported."""
    output = Path(output)
    blocks = {b['id']: b for b in corpus['blocks']}
    bundle = {'schema_version': SCHEMA_VERSION, 'prepared_at': now(), **corpus_fingerprints(corpus), 'source_documents': len(corpus['documents']), 'source_blocks': len(corpus['blocks']), 'documents': [{k: d.get(k) for k in ('id', 'issuer', 'ticker', 'fiscal_period', 'filing_date', 'url', 'sha256')} for d in corpus['documents']], 'models': [], 'excerpts': {}, 'method': {'entity_weight': .25, 'lexical_weight': .75, 'entity_matching': 'Exact casefolded whitespace-normalized mention string; no inferred aliases or relationship graph.', 'reference_boundary': 'Agent-authored sample; review status is recorded with each evaluation. Not human gold.', 'excerpt_policy': f'At most {excerpt_limit} representative block excerpts, each <= {excerpt_chars} characters; all processed entity postings retain block/document offsets.'}}
    wanted = []
    for spec in manifest['models']:
        directory = output / slug(spec['id'])
        status = load_json(directory/'status.json') if (directory/'status.json').exists() else {'status': 'not_run', 'processed_blocks': 0}
        model = {**{k: spec[k] for k in ('id', 'name', 'license', 'revision', 'profile')}, 'status': status, 'metadata': load_json(directory/'metadata.json') if (directory/'metadata.json').exists() else None, 'entities': load_json(directory/'entity-index.json') if (directory/'entity-index.json').exists() else [], 'retrieval': load_json(directory/'retrieval.json') if (directory/'retrieval.json').exists() else [], 'evaluation': load_json(directory/'evaluation.json') if (directory/'evaluation.json').exists() else None, 'langsmith':load_json(directory/'langsmith.json') if (directory/'langsmith.json').exists() else None}
        for query in model['retrieval']:
            for approach in ('lexical', 'entity_assisted'):
                wanted.extend(row['block_id'] for row in query.get(approach, []))
        for entry in model['entities'][:100]:
            wanted.extend(entry['block_ids'][:2])
        bundle['models'].append(model)
    for block_id in dict.fromkeys(wanted):
        if len(bundle['excerpts']) >= excerpt_limit:
            break
        if block_id in blocks:
            block = blocks[block_id]
            bundle['excerpts'][block_id] = {k: block.get(k) for k in ('id', 'document_id', 'ticker', 'fiscal_period', 'source_url', 'section', 'kind', 'page')}
            bundle['excerpts'][block_id].update(text=block['text'][:excerpt_chars], full_text_length=len(block['text']), truncated=len(block['text']) > excerpt_chars)
    bundle['complete_models'] = sum(m['status'].get('status') == 'complete' for m in bundle['models'])
    bundle['completion_status'] = 'complete' if bundle['complete_models'] == len(manifest['models']) else 'partial_or_failed'
    save_json(destination, bundle)
    return bundle
