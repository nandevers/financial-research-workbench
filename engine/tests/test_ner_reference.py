import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / 'data/ner-reference-sample.json'


def load_reference():
    return json.loads(REFERENCE.read_text(encoding='utf-8'))


def test_reference_is_bounded_source_annotation_with_explicit_provenance():
    ref = load_reference()
    passages = ref['passages']
    assert 30 <= len(passages) <= 45
    assert len({p['id'] for p in passages}) == len(passages)
    assert len({p['document_id'] for p in passages}) == ref['corpus_documents'] == 30
    assert ref['status'] in {'agent_authored_pending_second_review', 'agent_reviewed_not_human_gold'}
    assert ref['review']['model_predictions_seen'] is False
    assert 'Not human gold' in ref['annotation_origin']
    for passage in passages:
        assert 0 < len(passage['text']) <= 700
        assert passage['source_text_end'] - passage['source_text_start'] == len(passage['text'])
        previous_end = 0
        for entity in passage['entities']:
            assert entity['type'] in {'PERSON', 'ORGANIZATION', 'LOCATION'}
            assert entity['certainty'] in {'clear', 'uncertain'}
            assert 0 <= entity['start'] < entity['end'] <= len(passage['text'])
            assert passage['text'][entity['start']:entity['end']] == entity['text']
            assert previous_end <= entity['start']
            previous_end = entity['end']
    counts = Counter(e['type'] for p in passages for e in p['entities'] if e['certainty'] == 'clear')
    assert dict(counts) == ref['selection']['clear_entity_counts']
    assert sum(e['certainty'] == 'uncertain' for p in passages for e in p['entities']) == ref['selection']['uncertain_entity_count']
    if ref['status'] == 'agent_reviewed_not_human_gold':
        assert ref['review']['second_review_status'] == 'completed'
        assert ref['review']['reviewed_passages'] == len(passages)
        assert 'test snippets only' in ref['reporting_policy']['primary_ner_metrics']


def test_document_folds_and_declared_duplicate_groups_do_not_overlap():
    ref = load_reference()
    by_id = {p['id']: p for p in ref['passages']}
    dev = {p['document_id'] for p in ref['passages'] if p['split'] == 'development'}
    test = {p['document_id'] for p in ref['passages'] if p['split'] == 'test'}
    assert dev and test and dev.isdisjoint(test)
    assert dev == set(ref['split_policy']['development_document_ids'])
    assert test == set(ref['split_policy']['test_document_ids'])
    for pair in ref['split_policy']['near_duplicate_pairs']:
        a, b = [by_id[pid] for pid in pair['passage_ids']]
        assert a['split'] == b['split']


def test_retrieval_relevance_comes_from_clear_annotations_not_predictions():
    ref = load_reference()
    assert 8 <= len(ref['retrieval_queries']) <= 12
    assert ref['retrieval_judgments']['unjudged_policy'] == 'unjudged_not_negative'
    for query in ref['retrieval_queries']:
        relevant = []
        for passage in ref['passages']:
            mentions = {e['text'] for e in passage['entities'] if e['certainty'] == 'clear'}
            if all(any(alias in mentions for alias in group) for group in query['entity_mention_groups']):
                relevant.append(passage)
        assert relevant
        assert [p['id'] for p in relevant] == query['relevant_passage_ids']
        assert list(dict.fromkeys(p['source_block_id'] for p in relevant)) == query['relevant_block_ids']
        assert query['unjudged_policy'] == 'unjudged_not_negative'
        assert query['judgment_scope'] == 'reference_sample_blocks_only'
        assert query['relevant_passage_ids_by_split'][query['split']]
        for split in ('development', 'test'):
            assert [p['id'] for p in relevant if p['split'] == split] == query['relevant_passage_ids_by_split'][split]


def test_exact_slices_and_fingerprint_against_local_corpus_when_available():
    path = ROOT / 'artifacts/local/corpus.json'
    if not path.exists():
        pytest.skip('Full local corpus is intentionally not committed; fixture-only integrity checks still run.')
    ref = load_reference()
    corpus = json.loads(path.read_text(encoding='utf-8'))
    fingerprint = hashlib.sha256(json.dumps([(b['id'], b.get('filing_date')) for b in corpus['blocks']], sort_keys=True).encode()).hexdigest()
    assert fingerprint == ref['corpus_snapshot_sha256']
    assert len(corpus['blocks']) == ref['corpus_blocks']
    blocks = {b['id']: b for b in corpus['blocks']}
    for passage in ref['passages']:
        block = blocks[passage['source_block_id']]
        assert block['document_id'] == passage['document_id']
        assert block['text'][passage['source_text_start']:passage['source_text_end']] == passage['text']
