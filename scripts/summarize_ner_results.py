"""Explain the validated saved results; no model inference or hosted writes."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from ner_langsmith_replay import build_plan

ROOT = Path(__file__).resolve().parents[1]


def summarize(root=ROOT):
    root = Path(root)
    source = root/'web/public/data/ner-results.json'
    results = json.loads(source.read_text(encoding='utf-8'))
    reference = json.loads((root/'data/ner-reference-sample.json').read_text(encoding='utf-8'))
    plan = build_plan(root/'artifacts/local/ner', reference)
    assert results['complete_models'] == len(plan['models']) == 5
    assert not plan['pending_models'] and plan['heldout_retrieval_queries'] == 9
    models = []
    for saved in results['models']:
        model = next(m for m in plan['models'] if m['model_id'] == saved['id'])
        assert model['configuration_sha256'] == saved['metadata']['configuration_sha256']
        experiments = {e['variant']: e for e in model['experiments'] if e['task'] == 'retrieval'}
        native = {e['variant']: e for e in saved['langsmith']['replay']['experiments'] if e['task'] == 'retrieval'}
        for variant, experiment in experiments.items():
            assert experiment['summary'] == native[variant]['summary']
        ner = next(e['summary'] for e in model['experiments'] if e['task'] == 'ner')
        for metric in ('precision', 'recall', 'f1', 'tp', 'fp', 'fn'):
            assert ner[metric] == saved['evaluation']['metrics'][metric]
        queries = []
        for lexical, assisted in zip(experiments['lexical']['rows'], experiments['entity_assisted']['rows']):
            assert lexical['id'] == assisted['id']
            queries.append({'id': lexical['id'], 'query': lexical['inputs']['query'], 'split': lexical['split'],
                            'lexical': lexical['metric'], 'entity_assisted': assisted['metric']})
        test = [q for q in queries if q['split'] == 'test']
        models.append({'id': saved['id'], 'name': saved['name'],
                       'configuration_sha256': model['configuration_sha256'],
                       'tagging': saved['evaluation']['metrics'],
                       'lexical': experiments['lexical']['summary'],
                       'entity_assisted': experiments['entity_assisted']['summary'],
                       'recall_improved_test_queries': sum(q['entity_assisted']['known_positive_recall_at_5'] > q['lexical']['known_positive_recall_at_5'] for q in test),
                       'recall_worse_test_queries': sum(q['entity_assisted']['known_positive_recall_at_5'] < q['lexical']['known_positive_recall_at_5'] for q in test),
                       'queries': queries})
    return {'schema_version': 1, 'source_result_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'corpus_content_sha256': results['corpus_content_sha256'],
            'reference_sha256': plan['reference_sha256'], 'heldout_ner_passages': plan['heldout_ner_passages'],
            'heldout_retrieval_queries': plan['heldout_retrieval_queries'],
            'development_retrieval_queries': plan['retrieval_queries']-plan['heldout_retrieval_queries'],
            'scope': 'Nine test queries; known-positive blocks in the held-out reference fold only. Development evidence excluded. Unjudged blocks are not negatives.',
            'models': models}


if __name__ == '__main__':
    data = summarize()
    target = ROOT/'web/src/data/ner-findings.json'
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': 'saved_results_summarized', 'models': len(data['models']),
                      'test_queries': data['heldout_retrieval_queries'], 'no_inference_performed': True}))
