"""Synthetic exporter-shaped fixtures test publication; no model inference runs."""
import copy
import hashlib
import importlib.util
import json
import shutil
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from workbench.ner import (EntityRetrieval, MODEL_SPECS, corpus_fingerprints,
                           digest, entity_index, evaluate_reference, export_bundle,
                           known_positive_retrieval, merge_entities, normalize_entities,
                           save_json, slug)


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/import_ner_results.py'
SPEC = importlib.util.spec_from_file_location('ner_import_for_test', SCRIPT)
importer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(importer)


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def prediction(item, entities):
    return {'schema_version': 1, 'id': item['id'], 'document_id': item.get('document_id'),
            'text_sha256': digest(item['text']), 'text_length': len(item['text']),
            'status': 'complete', 'window_count': 1, 'covered_characters': len(item['text']),
            'entities': entities, 'errors': [], 'inference_seconds_allocated': .01}


@pytest.fixture
def job(tmp_path):
    root, incoming = tmp_path / 'repo', tmp_path / 'incoming'
    blocks = [{'id': 'b1', 'document_id': 'd1', 'text': 'Bank One holds capital.', 'ticker': 'ONE'},
              {'id': 'b2', 'document_id': 'd2', 'text': 'Bank Two', 'ticker': 'TWO'}]
    corpus = {'documents': [{'id': 'd1', 'ticker': 'ONE', 'sha256': 'a'},
                            {'id': 'd2', 'ticker': 'TWO', 'sha256': 'b'}], 'blocks': blocks}
    fingerprints = corpus_fingerprints(corpus)
    entities = [{'start': 0, 'end': 8, 'text': b['text'][:8], 'type': 'ORGANIZATION', 'score': .9} for b in blocks]
    passages = [{'id': f'p{i}', 'document_id': b['document_id'], 'source_block_id': b['id'],
                 'source_text_start': 0, 'source_text_end': len(b['text']), 'text': b['text'],
                 'split': split, 'entities': [{**entity, 'certainty': 'clear'}]}
                for i, (b, entity, split) in enumerate(zip(blocks, entities, ['development', 'test']))]
    reference = {'status': 'agent_reviewed_not_human_gold',
                 'review': {'second_review_status': 'completed'},
                 'corpus_snapshot_sha256': fingerprints['corpus_snapshot_sha256'],
                 'passages': passages,
                 'retrieval_queries': [
                     {'id': 'q1', 'split': 'development', 'query': 'Bank One liquidity', 'relevant_block_ids': ['b1']},
                     {'id': 'q2', 'split': 'test', 'query': 'Bank Two capital', 'relevant_block_ids': ['b2']},
                 ]}
    save_json(root / 'artifacts/local/corpus.json', corpus)
    save_json(root / 'data/ner-reference-sample.json', reference)
    save_json(root / importer.PUBLIC_PATH, {'existing_public': True})
    save_json(root / 'web/public/data/ner-index/existing.json', {'existing_index': True})
    save_json(root / 'artifacts/local/ner/existing-private.json', {'existing_private': True})
    raw = incoming / 'artifacts/local/ner'
    manifest = {'schema_version': 1, **fingerprints, 'total_documents': 2, 'total_blocks': 2,
                'models': [{**spec, 'revision': 'a' * 40} for spec in MODEL_SPECS]}
    save_json(raw / 'manifest.json', manifest)
    for spec in manifest['models']:
        directory = raw / slug(spec['id'])
        records = {b['id']: prediction(b, [entity]) for b, entity in zip(blocks, entities)}
        entries = entity_index(corpus, records)
        reference_predictions = {p['id']: prediction(p, [entity]) for p, entity in zip(passages, entities)}
        query_records, searches = [], []
        retriever = EntityRetrieval(corpus, entries)
        for query, entity in zip(reference['retrieval_queries'], entities):
            query_record = prediction({'id': query['id'], 'text': query['query']}, [entity])
            query_records.append(query_record)
            retrieval = retriever.rank(query['query'], query_record['entities'])
            searches.append({'id': query['id'], 'query': query['query'], 'status': 'complete',
                             'query_entities': query_record['entities'], **retrieval,
                             'known_positive_metrics': {name: known_positive_retrieval(query, retrieval[name])
                                                        for name in ('lexical', 'entity_assisted')}})
        identity = {'model_id': spec['id'], 'revision': spec['revision'],
                    'backbone_revision': None, **fingerprints, 'reference_sha256': digest(reference),
                    'config': {'entity_weight': .25}, 'code_sha256': 'b' * 64, 'packages': {}}
        metadata = {'schema_version': 1, **identity, 'configuration_sha256': digest(identity)}
        status = {'model_id': spec['id'], 'status': 'complete', 'loader_smoke_passed': True,
                  'total_documents': 2, 'processed_documents': 2, 'total_blocks': 2,
                  'processed_blocks': 2, 'failed_blocks': 0, 'reference_failures': 0,
                  'query_failures': 0, 'entity_mentions': 2, 'unique_entities': 2}
        for name, value in [('metadata.json', metadata), ('status.json', status),
                            ('entity-index.json', entries), ('retrieval.json', searches),
                            ('evaluation.json', evaluate_reference(reference, reference_predictions))]:
            save_json(directory / name, value)
        jsonl(directory / 'predictions.jsonl', records.values())
        jsonl(directory / 'reference-predictions.jsonl', reference_predictions.values())
        jsonl(directory / 'query-predictions.jsonl', query_records)
    export_bundle(corpus, manifest, raw, incoming / importer.PUBLIC_PATH)

    def archive():
        destination = tmp_path / 'results.zip'
        with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as zipped:
            for path in incoming.rglob('*'):
                if path.is_file():
                    zipped.write(path, path.relative_to(incoming).as_posix())
        return destination

    return root, incoming, archive


def test_complete_bundle_promotes_content_addressed_indexes_and_preserves_raw_input(job):
    root, incoming, archive = job
    original = (incoming / importer.PUBLIC_PATH).read_bytes()
    first = importer.import_results(archive(), root)
    public = importer.read_json(root / importer.PUBLIC_PATH)
    assert len(first['models']) == len(public['models']) == 5
    assert (incoming / importer.PUBLIC_PATH).read_bytes() == original
    assert (Path(first['private_import_directory']) / importer.PUBLIC_PATH).read_bytes() == original
    assert importer.read_json(Path(first['private_import_directory']) / 'previous-current-ner/existing-private.json') == {'existing_private': True}
    old_index = root / 'web/public/data/ner-index/existing.json'
    assert old_index.exists()
    for model in public['models']:
        encoded = (root / 'web/public' / model['entity_index_url']).read_bytes()
        assert hashlib.sha256(encoded).hexdigest() == model['entity_index_sha256']
        assert model['entity_index_sha256'][:16] in model['entity_index_url']
        assert len(json.loads(encoded)['entities']) == model['entity_count'] == 2
        assert model['entities'] == []
    second = importer.import_results(archive(), root)
    assert second['models'] == first['models']
    assert old_index.exists()


@pytest.mark.parametrize('problem', [
    'same_count_different_corpus', 'duplicate_model', 'changed_reference',
    'missing_corpus_prediction', 'wrong_source_hash', 'missing_reference_prediction',
    'invented_reference_score', 'missing_query_prediction', 'missing_retrieval_query',
    'stale_entity_index', 'altered_retrieval_score', 'credential_in_summary', 'nonfinite_score',
])
def test_invalid_bundle_preserves_current_public_and_private_files_and_incoming_artifacts(job, problem):
    root, incoming, archive = job
    path = incoming / importer.PUBLIC_PATH
    bundle = importer.read_json(path)
    model = bundle['models'][0]
    raw = incoming / 'artifacts/local/ner' / slug(model['id'])
    if problem == 'same_count_different_corpus':
        bundle['corpus_content_sha256'] = 'f' * 64
    elif problem == 'duplicate_model':
        bundle['models'].append(copy.deepcopy(model))
    elif problem == 'changed_reference':
        model['metadata']['reference_sha256'] = 'f' * 64
        save_json(raw / 'metadata.json', model['metadata'])
    elif problem in {'missing_corpus_prediction', 'wrong_source_hash', 'missing_reference_prediction', 'missing_query_prediction'}:
        name = {'missing_corpus_prediction': 'predictions.jsonl', 'wrong_source_hash': 'predictions.jsonl',
                'missing_reference_prediction': 'reference-predictions.jsonl',
                'missing_query_prediction': 'query-predictions.jsonl'}[problem]
        records = [json.loads(line) for line in (raw / name).read_text().splitlines()]
        if problem == 'wrong_source_hash':
            records[0]['text_sha256'] = 'f' * 64
        else:
            records.pop()
        jsonl(raw / name, records)
    elif problem in {'invented_reference_score', 'nonfinite_score'}:
        model['evaluation']['metrics']['f1'] = float('nan') if problem == 'nonfinite_score' else .123
        save_json(raw / 'evaluation.json', model['evaluation'])
    elif problem == 'missing_retrieval_query':
        model['retrieval'] = []
        save_json(raw / 'retrieval.json', [])
    elif problem == 'stale_entity_index':
        model['entities'][0]['mentions'] = []
        save_json(raw / 'entity-index.json', model['entities'])
    elif problem == 'altered_retrieval_score':
        model['retrieval'][0]['entity_assisted'][0]['score'] = .123
        save_json(raw / 'retrieval.json', model['retrieval'])
    elif problem == 'credential_in_summary':
        bundle['method']['unexpected_note'] = 'lsv2_pt_' + 'synthetic' * 5
    save_json(path, bundle)
    source_archive = archive()
    archive_bytes = source_archive.read_bytes()
    current = {p: p.read_bytes() for p in [root / importer.PUBLIC_PATH,
               root / 'web/public/data/ner-index/existing.json',
               root / 'artifacts/local/ner/existing-private.json']}
    with pytest.raises(ValueError, match='retained privately'):
        importer.import_results(source_archive, root)
    assert all(path.read_bytes() == value for path, value in current.items())
    assert source_archive.read_bytes() == archive_bytes
    staged = list((root / 'artifacts/local/ner-imports').iterdir())
    assert len(staged) == 1
    assert (staged[0] / importer.PUBLIC_PATH).read_bytes() == path.read_bytes()
    assert (staged[0] / 'artifacts/local/ner/manifest.json').exists()
    assert sorted(p.name for p in (root / 'web/public/data/ner-index').iterdir()) == ['existing.json']
    assert sorted(p.name for p in (root / 'artifacts/local/ner').iterdir()) == ['existing-private.json']


@pytest.mark.parametrize('name', ['artifacts/local/ner/../../outside.json',
                                 'artifacts\\local\\ner\\..\\..\\outside.json',
                                 'artifacts/local/ner/./status.json',
                                 'artifacts/local/ner//status.json',
                                 '/artifacts/local/ner/outside.json',
                                 'web/public/data/unexpected.json'])
def test_archive_paths_cannot_write_outside_private_staging(tmp_path, name):
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as zipped:
        zipped.writestr(name, '{}')
    with pytest.raises(ValueError, match='Unexpected result archive path'):
        importer.import_results(archive, tmp_path / 'repo')
    assert not (tmp_path / 'repo/web').exists()


def test_duplicate_and_symlink_zip_members_are_rejected(tmp_path):
    for kind in ('duplicate', 'symlink'):
        archive = tmp_path / (kind + '.zip')
        with zipfile.ZipFile(archive, 'w') as zipped:
            if kind == 'duplicate':
                zipped.writestr('artifacts/local/ner/status.json', '{}')
                with pytest.warns(UserWarning, match='Duplicate name'):
                    zipped.writestr('artifacts/local/ner/status.json', '{}')
            else:
                item = zipfile.ZipInfo('artifacts/local/ner/link')
                item.create_system = 3
                item.external_attr = (stat.S_IFLNK | 0o777) << 16
                zipped.writestr(item, '../../outside')
        with pytest.raises(ValueError, match='Duplicate result|symlinks'):
            importer.import_results(archive, tmp_path / kind)


def test_summary_replace_failure_keeps_previous_summary_and_indexes_usable(job, monkeypatch):
    root, _, archive = job
    previous = (root / importer.PUBLIC_PATH).read_bytes()
    real_write = importer.atomic_write

    def fail_summary(path, encoded):
        if path == root / importer.PUBLIC_PATH:
            raise OSError('Simulated publication interruption')
        real_write(path, encoded)

    monkeypatch.setattr(importer, 'atomic_write', fail_summary)
    with pytest.raises(OSError, match='Simulated publication interruption'):
        importer.import_results(archive(), root)
    assert (root / importer.PUBLIC_PATH).read_bytes() == previous
    assert importer.read_json(root / 'web/public/data/ner-index/existing.json') == {'existing_index': True}
    assert importer.read_json(root / 'artifacts/local/ner/existing-private.json') == {'existing_private': True}
    assert sorted(p.name for p in (root / 'artifacts/local/ner').iterdir()) == ['existing-private.json']
    assert len(list((root / 'artifacts/local/ner-imports').glob('*/artifacts/local/ner/manifest.json'))) == 1


def add_replay_status(job, source='incoming'):
    """Create synthetic native identities around a real saved-output replay plan."""
    root, incoming, _ = job
    replay_spec = importlib.util.spec_from_file_location('replay_for_import_fixture', SCRIPT.parent / 'ner_langsmith_replay.py')
    replay = importlib.util.module_from_spec(replay_spec)
    replay_spec.loader.exec_module(replay)
    reference = importer.read_json(root / 'data/ner-reference-sample.json')
    plan = replay.build_plan(incoming / 'artifacts/local/ner', reference)
    status = {key: plan[key] for key in ('schema_version', 'protocol', 'fingerprint',
                                        'corpus_snapshot_sha256', 'corpus_content_sha256', 'reference_sha256',
                                        'reference_status', 'reference_review', 'heldout_ner_passages',
                                        'retrieval_queries', 'heldout_retrieval_queries', 'target_run_count')}
    status.update(status='complete', no_inference_performed=True, pending_models=[],
                  additional_evaluator_traces=0, completed_replay_runs=plan['target_run_count'],
                  finished_at='2026-10-28T00:00:00+00:00', models=[])
    experiments = []
    for model in plan['models']:
        result = {key: model[key] for key in ('model_id', 'revision', 'configuration_sha256')}
        if model.get('recovery_sha256'):
            result['recovery_sha256'] = model['recovery_sha256']
        result['experiments'] = []
        for experiment in model['experiments']:
            fingerprint = replay.experiment_fingerprint(plan['fingerprint'], model, experiment)
            dataset_id = replay.identity('synthetic-dataset', experiment['dataset_key'])
            session_id = replay.identity('synthetic-session', fingerprint)
            tenant_id = replay.identity('synthetic-tenant')
            record = {'id': session_id, 'name': slug(model['model_id']) + '-' + experiment['task'] + '-' + experiment['variant'] + '-' + fingerprint[:10],
                      'dataset_id': dataset_id, 'dataset_name': 'workbench-' + experiment['dataset_key'] + '-' + plan['fingerprint'][:12],
                      'fingerprint': fingerprint, 'tenant_id': tenant_id,
                      'url': f'https://smith.langchain.com/o/{tenant_id}/datasets/{dataset_id}/compare?selectedSessions={session_id}',
                      'task': experiment['task'], 'variant': experiment['variant'],
                      'rows': len(experiment['rows']), 'summary': experiment['summary']}
            if model.get('recovery_sha256'):
                record['recovery_sha256'] = model['recovery_sha256']
            result['experiments'].append(record)
            experiments.append(record)
        status['models'].append(result)
    status['comparison_links'] = replay.comparison_links(experiments)
    path = (incoming if source == 'incoming' else root) / 'artifacts/local/ner/langsmith-replay/status.json'
    save_json(path, status)
    return status, path


@pytest.mark.parametrize('source', ['incoming', 'local'])
def test_optional_completed_replay_merges_by_model_and_preserves_legacy_experiment(job, source):
    root, incoming, archive = job
    bundle = importer.read_json(incoming / importer.PUBLIC_PATH)
    legacy = {'dataset_id': 'legacy-dataset', 'dataset_name': 'legacy-all-40',
              'experiment_name': 'legacy-40-reference-examples', 'source': 'Original recorded inference replay'}
    for model in bundle['models']:
        model['langsmith'] = copy.deepcopy(legacy)
        save_json(incoming / 'artifacts/local/ner' / slug(model['id']) / 'langsmith.json', legacy)
    save_json(incoming / importer.PUBLIC_PATH, bundle)
    status, _ = add_replay_status(job, source)
    importer.import_results(archive(), root)
    public = importer.read_json(root / importer.PUBLIC_PATH)
    recorded = {model['model_id']: model for model in status['models']}
    for model in public['models']:
        assert {key: model['langsmith'][key] for key in legacy} == legacy
        replay = model['langsmith']['replay']
        assert replay['status'] == 'complete' and replay['no_inference_performed'] is True
        assert replay['model_id'] == model['id']
        assert replay['experiments'] == recorded[model['id']]['experiments']
        assert replay['configuration_sha256'] == model['metadata']['configuration_sha256']
        assert replay['completed_replay_runs'] == 5
        assert replay['total_completed_replay_runs'] == 25
        assert replay['heldout_ner_passages'] == replay['heldout_retrieval_queries'] == 1
        assert replay['retrieval_queries'] == 2
        assert replay['comparison_links'] == status['comparison_links']
    assert importer.read_json(root / 'artifacts/local/ner/langsmith-replay/status.json') == status


@pytest.mark.parametrize('problem', [
    'wrong_corpus', 'wrong_reference', 'duplicate_model', 'wrong_revision', 'wrong_configuration',
    'partial', 'pending_model', 'incomplete_runs', 'missing_task', 'invented_summary',
    'wrong_experiment_fingerprint', 'http_url', 'wrong_host', 'url_userinfo', 'wrong_session',
    'external_comparison_link',
])
def test_invalid_replay_never_attaches_observability_claims_or_changes_current_files(job, problem):
    root, _, archive = job
    status, path = add_replay_status(job)
    model = status['models'][0]
    experiment = model['experiments'][0]
    if problem == 'wrong_corpus':
        status['corpus_content_sha256'] = 'f' * 64
    elif problem == 'wrong_reference':
        status['reference_sha256'] = 'f' * 64
    elif problem == 'duplicate_model':
        status['models'][-1] = copy.deepcopy(model)
    elif problem == 'wrong_revision':
        model['revision'] = 'f' * 40
    elif problem == 'wrong_configuration':
        model['configuration_sha256'] = 'f' * 64
    elif problem == 'partial':
        status['status'] = 'partial_pending_models'
    elif problem == 'pending_model':
        status['pending_models'] = [{'model_id': 'some/model', 'reason': 'Not complete'}]
    elif problem == 'incomplete_runs':
        status['completed_replay_runs'] -= 1
    elif problem == 'missing_task':
        model['experiments'].pop()
    elif problem == 'invented_summary':
        experiment['summary']['f1'] = .123
    elif problem == 'wrong_experiment_fingerprint':
        experiment['fingerprint'] = 'f' * 64
    elif problem == 'http_url':
        experiment['url'] = experiment['url'].replace('https:', 'http:')
    elif problem == 'wrong_host':
        experiment['url'] = experiment['url'].replace('smith.langchain.com', 'smith.langchain.com.invalid')
    elif problem == 'url_userinfo':
        experiment['url'] = experiment['url'].replace('smith.langchain.com', 'example.invalid@smith.langchain.com')
    elif problem == 'wrong_session':
        experiment['url'] = experiment['url'].replace(experiment['id'], '00000000-0000-4000-8000-000000000000')
    elif problem == 'external_comparison_link':
        status['comparison_links']['ner'] = 'https://example.invalid/'
    save_json(path, status)
    current = (root / importer.PUBLIC_PATH).read_bytes()
    with pytest.raises(ValueError, match='retained privately'):
        importer.import_results(archive(), root)
    assert (root / importer.PUBLIC_PATH).read_bytes() == current
    assert sorted(p.name for p in (root / 'artifacts/local/ner').iterdir()) == ['existing-private.json']
    assert sorted(p.name for p in (root / 'web/public/data/ner-index').iterdir()) == ['existing.json']


def test_stale_local_replay_is_not_reused_for_new_checkpoint_configuration(job):
    root, _, archive = job
    status, path = add_replay_status(job, 'local')
    status['models'][0]['configuration_sha256'] = 'f' * 64
    save_json(path, status)
    original = path.read_bytes()
    current = (root / importer.PUBLIC_PATH).read_bytes()
    with pytest.raises(ValueError, match='configuration mismatch'):
        importer.import_results(archive(), root)
    assert path.read_bytes() == original
    assert (root / importer.PUBLIC_PATH).read_bytes() == current
    assert importer.read_json(root / 'artifacts/local/ner/existing-private.json') == {'existing_private': True}


def add_targeted_recovery(job, monkeypatch):
    """A synthetic two-block base run plus one audited native model recovery."""
    root, incoming, _ = job
    corpus = importer.read_json(root / 'artifacts/local/corpus.json')
    directory = incoming / 'artifacts/local/ner/fastino--gliner2-base-v1'
    original_dir = directory / 'recovery-originals'
    original_dir.mkdir()
    sources = directory / 'recovery-sources'
    sources.mkdir()
    helper = importer.recovery_helper()
    snapshots = {'ner_recover_gliner2.py': (SCRIPT.parent / 'ner_recover_gliner2.py').read_bytes(),
                 'gliner2-processor.py': b'# Synthetic processor snapshot for this fixture\n',
                 'gliner2-runtime.py': b'# Synthetic runtime snapshot for this fixture\n',
                 'gliner2-normalize-text.py': b'# Synthetic append-period normalizer snapshot\n'}
    for name, encoded in snapshots.items():
        (sources / name).write_bytes(encoded)
    # The real pure projection is exercised; only fixture library source hashes
    # replace installed-package guards. No package/model is installed or loaded.
    monkeypatch.setattr(importer, 'recovery_helper', lambda: SimpleNamespace(
        project_native_entities=helper.project_native_entities,
        EXPECTED_PROCESSOR_SHA256=hashlib.sha256(snapshots['gliner2-processor.py']).hexdigest(),
        EXPECTED_RUNTIME_SHA256=hashlib.sha256(snapshots['gliner2-runtime.py']).hexdigest()))
    frozen_hashes = {}
    base_source_bytes = []
    for relative in ('engine/src/workbench/ner.py', 'scripts/ner_worker.py'):
        encoded = (SCRIPT.parents[1] / relative).read_bytes()
        source_path = original_dir / 'source' / Path(relative).name
        source_path.parent.mkdir(exist_ok=True)
        source_path.write_bytes(encoded)
        frozen_hashes[relative] = hashlib.sha256(encoded).hexdigest()
        base_source_bytes.append(encoded.replace(b'\r\n', b'\n'))
    metadata = importer.read_json(directory / 'metadata.json')
    metadata['code_sha256'] = hashlib.sha256(b''.join(base_source_bytes)).hexdigest()
    metadata['packages']['gliner2'] = '2.1.0'
    identity_keys = ('model_id', 'revision', 'backbone_revision', 'corpus_snapshot_sha256',
                     'corpus_content_sha256', 'reference_sha256', 'config', 'code_sha256', 'packages')
    metadata['configuration_sha256'] = digest({key: metadata.get(key) for key in identity_keys})
    save_json(directory / 'metadata.json', metadata)
    original_rows = importer.jsonl_records(directory / 'predictions.jsonl')
    original_rows[-1].update(status='failed', covered_characters=None, entities=[],
                             errors=['ValueError: Entity offset outside its input: 0:9/8'])
    jsonl(directory / 'predictions.jsonl', original_rows)
    original_records = {row['id']: row for row in original_rows}
    original_index = entity_index(corpus, original_records)
    save_json(directory / 'entity-index.json', original_index)
    status = importer.read_json(directory / 'status.json')
    status.update(status='partial', processed_blocks=1, processed_documents=1, failed_blocks=1,
                  entity_mentions=1, unique_entities=1, corpus_inference_seconds=.02)
    save_json(directory / 'status.json', status)
    for name in helper.ARTIFACTS:
        if (directory / name).exists():
            shutil.copyfile(directory / name, original_dir / name)
    original_hashes = {path.name: importer.file_sha256(path) for path in original_dir.iterdir() if path.is_file()}
    identity = {'model_id': metadata['model_id'], 'revision': metadata['revision'],
                'base_configuration_sha256': metadata['configuration_sha256'], 'base_code_sha256': metadata['code_sha256'],
                'corpus_snapshot_sha256': metadata['corpus_snapshot_sha256'],
                'corpus_content_sha256': metadata['corpus_content_sha256'], 'reference_sha256': metadata['reference_sha256'],
                'helper_source_sha256': importer.file_sha256(sources / 'ner_recover_gliner2.py'),
                'library': {'version': '2.1.0', 'processor_sha256': importer.file_sha256(sources / 'gliner2-processor.py'),
                            'runtime_sha256': importer.file_sha256(sources / 'gliner2-runtime.py'),
                            'normalizer_source_sha256': importer.file_sha256(sources / 'gliner2-normalize-text.py')},
                'original_artifact_sha256': original_hashes, 'frozen_source_sha256': frozen_hashes,
                'original_predictions_bytes': (directory / 'predictions.jsonl').stat().st_size,
                'failed_block_ids': ['b2'], 'inference_config': metadata['config'],
                'window_batch_policy': 'One window per local model call; does not exceed the immutable base maximum batch size.'}
    recovery_sha = digest(identity)
    block = corpus['blocks'][-1]
    native = {'organization': [{'start': 0, 'end': 9, 'text': 'Bank Two.', 'confidence': .9}]}
    projected, decisions = helper.project_native_entities(block['text'], native)
    audit = {'schema_version': 1, 'block_id': 'b2', 'producer_recovery_sha256': recovery_sha,
             'parent_prediction_sha256': digest(original_records['b2']), 'text_sha256': digest(block['text']),
             'windows': [{'start': 0, 'end': 8, 'input_sha256': digest(block['text']), 'input_length': 8,
                          'normalized_sha256': digest('Bank Two.'), 'native_entities': native,
                          'projection_decisions': decisions}], 'inference_seconds': .01,
             'recorded_at': '2026-10-28T00:00:00+00:00'}
    jsonl(directory / 'recovery-native.jsonl', [audit])
    record = prediction(block, merge_entities(block['text'], normalize_entities(block['text'], projected)))
    record.update(producer_recovery_sha256=recovery_sha, base_configuration_sha256=metadata['configuration_sha256'],
                  parent_prediction_sha256=digest(original_records['b2']), native_audit_record_sha256=digest(audit))
    with (directory / 'predictions.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record) + '\n')
    records = {**original_records, 'b2': record}
    entries = entity_index(corpus, records)
    save_json(directory / 'entity-index.json', entries)
    status.update(status='complete', processed_blocks=2, processed_documents=2, failed_blocks=0,
                  entity_mentions=2, unique_entities=2, corpus_inference_seconds=.03,
                  recovery_inference_seconds=.01,
                  recovery={'policy': helper.POLICY, 'recovery_sha256': recovery_sha, 'recovered_blocks': 1,
                            'original_success_count': 1, 'base_configuration_sha256': metadata['configuration_sha256']})
    save_json(directory / 'status.json', status)
    recovery = {'schema_version': 1, 'policy': helper.POLICY, 'status': 'complete', 'identity': identity,
                'recovery_sha256': recovery_sha, 'started_at': '2026-10-28T00:00:00+00:00',
                'finished_at': '2026-10-28T00:00:01+00:00', 'original_success_count': 1, 'remaining_failed_count': 0,
                'append_records': [{'block_id': 'b2', 'parent_prediction_sha256': digest(original_records['b2']),
                                    'appended_prediction_sha256': digest(record), 'native_audit_record_sha256': digest(audit)}],
                'final_predictions_sha256': importer.file_sha256(directory / 'predictions.jsonl'),
                'post_artifact_sha256': {name: importer.file_sha256(directory / name) for name in helper.ARTIFACTS
                                         if name != 'predictions.jsonl' and (directory / name).exists()}}
    save_json(directory / 'recovery.json', recovery)
    refresh_public_bundle(job)
    return directory, recovery


def refresh_public_bundle(job):
    root, incoming, _ = job
    corpus = importer.read_json(root / 'artifacts/local/corpus.json')
    raw = incoming / 'artifacts/local/ner'
    export_bundle(corpus, importer.read_json(raw / 'manifest.json'), raw, incoming / importer.PUBLIC_PATH)


def test_audited_targeted_recovery_preserves_base_identity_successful_prefix_and_replay_lineage(job, monkeypatch):
    root, _, archive = job
    directory, recovery = add_targeted_recovery(job, monkeypatch)
    original_metadata = (directory / 'metadata.json').read_bytes()
    original_prefix = (directory / 'recovery-originals/predictions.jsonl').read_bytes()
    add_replay_status(job)
    importer.import_results(archive(), root)
    public = importer.read_json(root / importer.PUBLIC_PATH)
    model = next(model for model in public['models'] if model['id'] == 'fastino/gliner2-base-v1')
    assert model['recovery'] == recovery
    assert model['metadata'] == json.loads(original_metadata)
    assert (root / 'artifacts/local/ner/fastino--gliner2-base-v1/metadata.json').read_bytes() == original_metadata
    assert (directory / 'predictions.jsonl').read_bytes().startswith(original_prefix)
    assert model['langsmith']['replay']['recovery_sha256'] == recovery['recovery_sha256']
    assert all('recovery' not in other and 'recovery_sha256' not in other['langsmith']['replay']
               for other in public['models'] if other['id'] != model['id'])


@pytest.mark.parametrize('problem', ['missing_manifest', 'changed_base_metadata', 'changed_success_prefix',
                                     'wrong_targets', 'wrong_library', 'wrong_parent_hash',
                                     'wrong_native_decision', 'missing_source_snapshot', 'wrong_post_hash'])
def test_recovery_does_not_relax_original_import_validation_or_native_proof(job, monkeypatch, problem):
    root, _, archive = job
    directory, recovery = add_targeted_recovery(job, monkeypatch)
    if problem == 'missing_manifest':
        (directory / 'recovery.json').unlink()
    elif problem == 'changed_base_metadata':
        metadata = importer.read_json(directory / 'metadata.json')
        metadata['unexpected_change'] = True
        save_json(directory / 'metadata.json', metadata)
    elif problem == 'changed_success_prefix':
        path = directory / 'predictions.jsonl'
        path.write_bytes(path.read_bytes().replace(b'"inference_seconds_allocated": 0.01', b'"inference_seconds_allocated": 0.02', 1))
    elif problem == 'wrong_targets':
        recovery['identity']['failed_block_ids'] = ['b1']
        recovery['recovery_sha256'] = digest(recovery['identity'])
        save_json(directory / 'recovery.json', recovery)
    elif problem == 'wrong_library':
        recovery['identity']['library']['version'] = '2.1.1'
        recovery['recovery_sha256'] = digest(recovery['identity'])
        save_json(directory / 'recovery.json', recovery)
    elif problem == 'wrong_parent_hash':
        recovery['append_records'][0]['parent_prediction_sha256'] = 'f' * 64
        save_json(directory / 'recovery.json', recovery)
    elif problem == 'wrong_native_decision':
        audit = importer.jsonl_records(directory / 'recovery-native.jsonl')[0]
        audit['windows'][0]['projection_decisions'][0]['projected_text'] = 'Invented name'
        jsonl(directory / 'recovery-native.jsonl', [audit])
        rows = importer.jsonl_records(directory / 'predictions.jsonl')
        rows[-1]['native_audit_record_sha256'] = digest(audit)
        jsonl(directory / 'predictions.jsonl', rows)
        recovery['append_records'][0].update(native_audit_record_sha256=digest(audit), appended_prediction_sha256=digest(rows[-1]))
        recovery['final_predictions_sha256'] = importer.file_sha256(directory / 'predictions.jsonl')
        save_json(directory / 'recovery.json', recovery)
    elif problem == 'missing_source_snapshot':
        (directory / 'recovery-sources/gliner2-processor.py').unlink()
    elif problem == 'wrong_post_hash':
        recovery['post_artifact_sha256']['entity-index.json'] = 'f' * 64
        save_json(directory / 'recovery.json', recovery)
    refresh_public_bundle(job)
    current = (root / importer.PUBLIC_PATH).read_bytes()
    with pytest.raises(ValueError, match='retained privately'):
        importer.import_results(archive(), root)
    assert (root / importer.PUBLIC_PATH).read_bytes() == current
    assert sorted(path.name for path in (root / 'artifacts/local/ner').iterdir()) == ['existing-private.json']
