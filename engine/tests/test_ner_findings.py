"""Reconcile published conclusions with independently recomputed fold metrics."""
import hashlib
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_findings_use_the_actual_results_and_exclude_development_evidence():
    source = ROOT/'web/public/data/ner-results.json'
    results = json.loads(source.read_text(encoding='utf-8'))
    findings = json.loads((ROOT/'web/src/data/ner-findings.json').read_text(encoding='utf-8'))
    reference = json.loads((ROOT/'data/ner-reference-sample.json').read_text(encoding='utf-8'))
    assert findings['source_result_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    test_blocks = {p['source_block_id'] for p in reference['passages'] if p['split'] == 'test'}
    queries = {q['id']: q for q in reference['retrieval_queries']}
    assert findings['heldout_retrieval_queries'] == 9 and findings['heldout_ner_passages'] == 32
    for model in findings['models']:
        original = next(m for m in results['models'] if m['id'] == model['id'])
        assert model['tagging'] == original['evaluation']['metrics']
        assert model['configuration_sha256'] == original['metadata']['configuration_sha256']
        assert len(model['queries']) == 10
        for variant in ('lexical', 'entity_assisted'):
            recalls, rrs = [], []
            for q in model['queries']:
                ref = queries[q['id']]
                relevant = set(ref['relevant_block_ids'])
                relevant = relevant & test_blocks if ref['split'] == 'test' else relevant-test_blocks
                saved = next(row for row in original['retrieval'] if row['id'] == q['id'])
                top = [h['block_id'] for h in saved[variant]][:5]
                recall = len(set(top) & relevant)/len(relevant)
                rr = next((1/(i+1) for i, block in enumerate(top) if block in relevant), 0)
                assert q[variant]['known_positive_recall_at_5'] == pytest.approx(recall)
                assert q[variant]['first_known_positive_reciprocal_rank_at_5'] == pytest.approx(rr)
                if q['split'] == 'test': recalls.append(recall); rrs.append(rr)
            assert len(recalls) == 9
            assert model[variant]['mean_test_known_positive_recall_at_5'] == pytest.approx(sum(recalls)/9)
            assert model[variant]['mean_test_first_known_positive_reciprocal_rank_at_5'] == pytest.approx(sum(rrs)/9)
    medium = next(m for m in findings['models'] if m['id'] == 'urchade/gliner_medium-v2.1')
    first_republic = next(q for q in medium['queries'] if q['id'] == 'ner-query-04-first-republic')
    # The original detail has a development-fold hit; it is not a test gain.
    assert first_republic['entity_assisted']['known_positive_recall_at_5'] == 0
