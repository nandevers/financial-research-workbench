"""Validate completed saved model artifacts before publishing their lazy entity indexes."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'engine/src'))
from workbench.ner import (LABELS, corpus_fingerprints, digest, entity_index,
                           evaluate_reference, known_positive_retrieval,
                           merge_entities, normalize_entities, normalized_name,
                           slug, validate_reference)

MODEL_IDS = {'dslim/distilbert-NER', 'dslim/bert-base-NER',
             'urchade/gliner_medium-v2.1', 'numind/NuNER_Zero',
             'fastino/gliner2-base-v1'}
PUBLIC_PATH = Path('web/public/data/ner-results.json')
CREDENTIAL_PATTERN = re.compile(rb'(?:lsv2_pt_|lsv2_sk_|sk-proj-|ghp_)[A-Za-z0-9_-]{12,}')


def read_json(path):
    def reject_constant(value):
        raise ValueError('Non-finite JSON number: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject_constant)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_by_id(rows, expected, description, key='id'):
    require(isinstance(rows, list), description + ' must be a list.')
    ids = [row.get(key) for row in rows]
    require(len(ids) == len(expected) and len(set(ids)) == len(ids)
            and set(ids) == set(expected), 'Unexpected, duplicate or missing ' + description + '.')
    return {row[key]: row for row in rows}


def stage_archive(archive, root, archive_hash):
    """Keep incoming raw artifacts private, including when validation fails."""
    parent = root / 'artifacts/local/ner-imports'
    parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=archive_hash[:12] + '-', dir=parent))
    try:
        with zipfile.ZipFile(archive) as zipped:
            names = set()
            for item in zipped.infolist():
                name = item.filename.replace('\\', '/')
                parts = PurePosixPath(name).parts
                require(not name.startswith('/') and ':' not in name
                        and '..' not in parts and '.' not in name.split('/')
                        and name.rstrip('/') == PurePosixPath(name).as_posix(),
                        'Unexpected result archive path: ' + name)
                require(name not in names, 'Duplicate result archive path: ' + name)
                names.add(name)
                allowed = name.startswith('artifacts/local/ner/') or name == PUBLIC_PATH.as_posix()
                require(allowed, 'Unexpected result archive path: ' + name)
                require(not stat.S_ISLNK(item.external_attr >> 16), 'Archive symlinks are not permitted.')
                target = staged.joinpath(*parts)
                require(staged in target.resolve().parents, 'Archive path escapes staging directory.')
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zipped.open(item) as source, target.open('wb') as destination:
                        shutil.copyfileobj(source, destination)
    except Exception as error:
        raise ValueError(f'{error} Incoming artifacts retained privately at {staged}') from error
    return staged


def validate_entities(entities, text, description):
    require(isinstance(entities, list), 'Missing entity predictions: ' + description)
    for entity in entities:
        start, end, score = entity.get('start'), entity.get('end'), entity.get('score')
        require(type(start) is int and type(end) is int and 0 <= start < end <= len(text),
                'Invalid entity offsets: ' + description)
        require(entity.get('text') == text[start:end] and entity.get('type') in LABELS,
                'Entity text/type does not match its source: ' + description)
        require(isinstance(score, (int, float)) and not isinstance(score, bool)
                and math.isfinite(score) and 0 <= score <= 1,
                'Invalid entity confidence: ' + description)


def validate_predictions(path, items, description):
    """Allow checkpoint retries, but require a complete final record for every input."""
    expected = {item['id']: item for item in items}
    latest = {}
    with path.open(encoding='utf-8-sig') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError(f'Invalid {description} JSONL at line {line_number}.') from error
            require(isinstance(row, dict) and row.get('id') in expected,
                    'Unexpected input in ' + description + '.')
            latest[row['id']] = row
    require(set(latest) == set(expected), 'Missing input predictions in ' + description + '.')
    for item_id, item in expected.items():
        row, text = latest[item_id], item['text']
        require(row.get('schema_version') == 1 and row.get('status') == 'complete',
                'Incomplete final prediction in ' + description + ': ' + item_id)
        require(row.get('text_sha256') == digest(text) and row.get('text_length') == len(text)
                and row.get('document_id') == item.get('document_id'),
                'Prediction input provenance mismatch in ' + description + ': ' + item_id)
        require(row.get('covered_characters') == len(text) and not row.get('errors')
                and type(row.get('window_count')) is int and row['window_count'] > 0,
                'Prediction does not cover the complete input in ' + description + ': ' + item_id)
        validate_entities(row.get('entities'), text, description + ': ' + item_id)
    return latest


def file_sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def jsonl_records(path):
    """Read audit records without repairing or truncating any input artifact."""
    with path.open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def recovery_helper():
    spec = importlib.util.spec_from_file_location('ner_recovery_import_validation', ROOT / 'scripts/ner_recover_gliner2.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def validate_recovery(directory, model, blocks, predictions):
    """Verify a separately identified repair while preserving base-run identity."""
    path = directory / 'recovery.json'
    if not path.exists():
        require(not any(row.get('producer_recovery_sha256') for row in predictions.values()),
                'Recovered predictions have no recovery provenance manifest.')
        return
    manifest = read_json(path)
    identity = manifest.get('identity', {})
    metadata = model['metadata']
    require(model['id'] == 'fastino/gliner2-base-v1' and manifest.get('schema_version') == 1
            and manifest.get('policy') == 'gliner2-2.1.0-synthetic-terminal-period-v1'
            and manifest.get('status') == 'complete'
            and manifest.get('recovery_sha256') == digest(identity), 'Invalid GLiNER2 recovery manifest.')
    for key, expected in {'model_id': model['id'], 'revision': model['revision'],
                          'base_configuration_sha256': metadata['configuration_sha256'],
                          'base_code_sha256': metadata['code_sha256'],
                          'corpus_snapshot_sha256': metadata['corpus_snapshot_sha256'],
                          'corpus_content_sha256': metadata['corpus_content_sha256'],
                          'reference_sha256': metadata['reference_sha256'],
                          'inference_config': metadata['config']}.items():
        require(identity.get(key) == expected, 'Recovery/base execution identity mismatch: ' + key)
    require(identity.get('window_batch_policy') == 'One window per local model call; does not exceed the immutable base maximum batch size.',
            'Recovery window execution policy is not recorded explicitly.')
    originals = directory / 'recovery-originals'
    original_hashes = identity.get('original_artifact_sha256', {})
    required = {'predictions.jsonl', 'status.json', 'metadata.json', 'entity-index.json',
                'reference-predictions.jsonl', 'query-predictions.jsonl', 'evaluation.json', 'retrieval.json'}
    if (directory / 'langsmith.json').exists():
        required.add('langsmith.json')
    require(required.issubset(original_hashes), 'Recovery did not preserve all original inference artifacts.')
    for name, expected_hash in original_hashes.items():
        require(name == Path(name).name and '/' not in name and '\\' not in name
                and re.fullmatch('[a-f0-9]{64}', expected_hash) is not None
                and file_sha256(originals / name) == expected_hash,
                'Original recovery artifact checksum mismatch: ' + name)
    for name in ('metadata.json', 'reference-predictions.jsonl', 'query-predictions.jsonl'):
        require(file_sha256(directory / name) == original_hashes[name],
                'Recovery altered immutable base metadata/reference/query predictions: ' + name)
    base_sources = [originals / 'source' / name for name in ('ner.py', 'ner_worker.py')]
    frozen_hashes = identity.get('frozen_source_sha256', {})
    require(frozen_hashes == {relative: file_sha256(originals / 'source' / Path(relative).name)
                              for relative in ('engine/src/workbench/ner.py', 'scripts/ner_worker.py')},
            'Preserved base source snapshot checksums do not match.')
    require(hashlib.sha256(b''.join(source.read_bytes().replace(b'\r\n', b'\n') for source in base_sources)).hexdigest()
            == metadata['code_sha256'], 'Preserved base source does not match its original code identity.')
    sources = directory / 'recovery-sources'
    library = identity.get('library', {})
    helper = recovery_helper()
    require(library.get('version') == '2.1.0' and metadata.get('packages', {}).get('gliner2') == '2.1.0',
            'Recovery applies only to the verified GLiNER2 2.1.0 normalizer.')
    require(library.get('processor_sha256') == helper.EXPECTED_PROCESSOR_SHA256
            and library.get('runtime_sha256') == helper.EXPECTED_RUNTIME_SHA256,
            'Recovery library sources differ from the inspected normalization proof.')
    source_hashes = {'ner_recover_gliner2.py': identity.get('helper_source_sha256'),
                     'gliner2-processor.py': library.get('processor_sha256'),
                     'gliner2-runtime.py': library.get('runtime_sha256'),
                     'gliner2-normalize-text.py': library.get('normalizer_source_sha256')}
    require(all(isinstance(expected_hash, str) and re.fullmatch('[a-f0-9]{64}', expected_hash)
                and file_sha256(sources / name) == expected_hash for name, expected_hash in source_hashes.items()),
            'Recovery helper/library source snapshot mismatch.')
    original_path = originals / 'predictions.jsonl'
    prefix_bytes = identity.get('original_predictions_bytes')
    require(type(prefix_bytes) is int and prefix_bytes == original_path.stat().st_size and prefix_bytes > 0,
            'Invalid original prediction prefix size.')
    with (directory / 'predictions.jsonl').open('rb') as stream:
        remaining, prefix_hash = prefix_bytes, hashlib.sha256()
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            require(bool(chunk), 'Recovered prediction file lost the original prefix.')
            prefix_hash.update(chunk)
            remaining -= len(chunk)
        suffix = stream.read().decode('utf-8')
    require(prefix_hash.hexdigest() == original_hashes['predictions.jsonl'],
            'Recovery rewrote original successful/failed prediction history.')
    original_records = {row['id']: row for row in jsonl_records(original_path)}
    require(set(original_records) == set(blocks), 'Original recovery run did not process the entire corpus.')
    require(not any(row.get('producer_recovery_sha256') for row in original_records.values()),
            'The preserved base run already contains an unsupported earlier recovery producer.')
    failed = sorted(row['id'] for row in original_records.values() if row.get('status') != 'complete')
    targets = identity.get('failed_block_ids')
    require(failed and isinstance(targets, list) and len(targets) == len(failed) and set(targets) == set(failed)
            and manifest.get('original_success_count') == len(blocks) - len(failed)
            and manifest.get('remaining_failed_count') == 0,
            'Recovery scope must equal exactly the originally failed corpus blocks.')
    for block_id in failed:
        errors = original_records[block_id].get('errors')
        require(isinstance(errors, list) and errors, 'Originally failed block has no recorded offset failure.')
        for error in errors:
            match = re.fullmatch(r'ValueError: Entity offset outside its input: (\d+):(\d+)/(\d+)', error)
            require(match is not None and int(match[2]) == int(match[3]) + 1
                    and 0 <= int(match[1]) <= int(match[3]), 'Original failure is outside the synthetic-period recovery policy.')
    appended = [json.loads(line) for line in suffix.splitlines() if line.strip()]
    by_id = unique_by_id(appended, failed, 'appended recovery predictions')
    appended_manifest = unique_by_id(manifest.get('append_records'), failed, 'recovery append manifest', 'block_id')
    # An interrupted recovery can leave an audit before its prediction append.
    # Preserve those attempts; only each final prediction's hash selects evidence.
    audits = {}
    for audit in jsonl_records(directory / 'recovery-native.jsonl'):
        require(audit.get('block_id') in failed
                and audit.get('producer_recovery_sha256') == manifest['recovery_sha256']
                and audit.get('parent_prediction_sha256') == digest(original_records[audit['block_id']])
                and audit.get('text_sha256') == digest(blocks[audit['block_id']]['text']),
                'Unrelated or mismatched native recovery audit.')
        audits[digest(audit)] = audit
    for block_id in failed:
        row = by_id[block_id]
        audit = audits.get(row.get('native_audit_record_sha256'))
        require(audit is not None and audit.get('block_id') == block_id, 'Missing linked native recovery audit.')
        link = appended_manifest[block_id]
        parent_hash, audit_hash = digest(original_records[block_id]), digest(audit)
        require(row == predictions[block_id]
                and row.get('producer_recovery_sha256') == manifest['recovery_sha256']
                and row.get('base_configuration_sha256') == metadata['configuration_sha256']
                and row.get('parent_prediction_sha256') == parent_hash
                and row.get('native_audit_record_sha256') == audit_hash
                and link.get('parent_prediction_sha256') == parent_hash
                and link.get('appended_prediction_sha256') == digest(row)
                and link.get('native_audit_record_sha256') == audit_hash,
                'Recovered prediction/parent/native audit hash chain mismatch: ' + block_id)
        text = blocks[block_id]['text']
        require(audit.get('schema_version') == 1 and audit.get('producer_recovery_sha256') == manifest['recovery_sha256']
                and audit.get('parent_prediction_sha256') == parent_hash
                and audit.get('text_sha256') == digest(text), 'Native recovery audit input mismatch: ' + block_id)
        windows = audit.get('windows')
        require(isinstance(windows, list) and windows and windows[0].get('start') == 0
                and windows[-1].get('end') == len(text) and row['window_count'] == len(windows)
                and isinstance(audit.get('inference_seconds'), (int, float))
                and math.isfinite(audit['inference_seconds']) and audit['inference_seconds'] >= 0
                and row.get('inference_seconds_allocated') == audit['inference_seconds'],
                'Recovery windows/timing do not match the full source block.')
        recovered_entities, previous_end = [], 0
        for window in windows:
            start, end = window.get('start'), window.get('end')
            require(type(start) is int and type(end) is int and 0 <= start <= end <= len(text)
                    and start <= previous_end, 'Invalid or incomplete recovery window coverage.')
            window_text = text[start:end]
            normalized = window_text if window_text.endswith(('.', '!', '?')) else window_text + '.'
            require(window.get('input_length') == len(window_text) and window.get('input_sha256') == digest(window_text)
                    and window.get('normalized_sha256') == digest(normalized),
                    'Recovery normalization changed original source characters.')
            projected, decisions = helper.project_native_entities(window_text, window.get('native_entities'))
            require(window.get('projection_decisions') == decisions, 'Recovery projection audit mismatch.')
            recovered_entities.extend(normalize_entities(window_text, projected, offset=start))
            previous_end = max(previous_end, end)
        require(row['entities'] == merge_entities(text, recovered_entities),
                'Recovered entities do not equal the audited actual native predictions: ' + block_id)
    require(manifest.get('final_predictions_sha256') == file_sha256(directory / 'predictions.jsonl'),
            'Final recovered prediction checksum mismatch.')
    post_hashes = manifest.get('post_artifact_sha256', {})
    require((required - {'predictions.jsonl'}).issubset(post_hashes), 'Recovery is missing final derived artifact checksums.')
    for name, expected_hash in post_hashes.items():
        require(name == Path(name).name and '/' not in name and '\\' not in name
                and file_sha256(directory / name) == expected_hash,
                'Final recovery artifact checksum mismatch: ' + name)
    expected_status = {'policy': manifest['policy'], 'recovery_sha256': manifest['recovery_sha256'],
                       'recovered_blocks': len(failed), 'original_success_count': manifest['original_success_count'],
                       'base_configuration_sha256': metadata['configuration_sha256']}
    require(model['status'].get('recovery') == expected_status,
            'Final corpus status does not disclose its targeted recovery provenance.')
    # Base metadata remains byte-for-byte unchanged. The separately hashed sidecar
    # identifies precisely which real predictions used the correction policy.
    model['recovery'] = manifest


def validate_retrieval(rows, queries, predictions, blocks, entries):
    by_id = unique_by_id(rows, [query['id'] for query in queries], 'retrieval queries')
    postings = {entry['normalized']: set(entry['block_ids']) for entry in entries}
    for query in queries:
        row = by_id[query['id']]
        entities = predictions[query['id']]['entities']
        keys = sorted({normalized_name(entity['text']) for entity in entities})
        require(row.get('status') == 'complete' and row.get('query') == query['query']
                and row.get('query_entities') == entities and row.get('query_entity_keys') == keys
                and row.get('entity_boost_applied') is bool(keys),
                'Retrieval query inference mismatch: ' + query['id'])
        for approach in ('lexical', 'entity_assisted'):
            hits = row.get(approach)
            require(isinstance(hits, list) and len(hits) <= 5
                    and len({hit['block_id'] for hit in hits}) == len(hits),
                    'Invalid retrieval ranking: ' + query['id'])
            for hit in hits:
                block = blocks.get(hit.get('block_id'))
                require(block is not None and hit.get('document_id') == block['document_id'],
                        'Retrieval result is not an existing corpus block: ' + query['id'])
                for metric in ('score', 'lexical_score', 'entity_overlap'):
                    value = hit.get(metric)
                    require(isinstance(value, (int, float)) and not isinstance(value, bool)
                            and math.isfinite(value) and 0 <= value <= 1,
                            'Invalid retrieval score: ' + query['id'])
                overlap = sum(hit['block_id'] in postings.get(key, set()) for key in keys) / len(keys) if keys else 0.0
                score = .75 * hit['lexical_score'] + .25 * overlap if approach == 'entity_assisted' and keys else hit['lexical_score']
                require(abs(hit['entity_overlap'] - overlap) <= 1e-6
                        and abs(hit['score'] - score) <= 1e-6,
                        'Retrieval score does not match actual entity postings: ' + query['id'])
            require(row.get('known_positive_metrics', {}).get(approach) == known_positive_retrieval(query, hits),
                    'Retrieval evaluation does not match fixed relevance judgments: ' + query['id'])


def validate_bundle(staged, root):
    data = read_json(staged / PUBLIC_PATH)
    corpus = read_json(root / 'artifacts/local/corpus.json')
    reference = read_json(root / 'data/ner-reference-sample.json')
    validate_reference(corpus, reference)
    fingerprints = corpus_fingerprints(corpus)
    require(data.get('schema_version') == 1 and data.get('completion_status') == 'complete'
            and data.get('complete_models') == len(MODEL_IDS), 'All five models must finish before publication.')
    require(all(data.get(key) == value for key, value in fingerprints.items()),
            'Result corpus fingerprints do not match the existing local corpus.')
    require(data.get('source_documents') == len(corpus['documents'])
            and data.get('source_blocks') == len(corpus['blocks']), 'Result corpus coverage does not match.')
    expected_documents = [{key: doc.get(key) for key in
                           ('id', 'issuer', 'ticker', 'fiscal_period', 'filing_date', 'url', 'sha256')}
                          for doc in corpus['documents']]
    require(data.get('documents') == expected_documents, 'Result document provenance does not match.')
    models = unique_by_id(data.get('models'), MODEL_IDS, 'checkpoint models')
    raw = staged / 'artifacts/local/ner'
    manifest = read_json(raw / 'manifest.json')
    require(manifest.get('schema_version') == 1
            and all(manifest.get(key) == value for key, value in fingerprints.items())
            and manifest.get('total_documents') == len(corpus['documents'])
            and manifest.get('total_blocks') == len(corpus['blocks']), 'Raw manifest corpus does not match.')
    specs = unique_by_id(manifest.get('models'), MODEL_IDS, 'manifest checkpoint models')
    blocks = {block['id']: block for block in corpus['blocks']}
    queries = reference.get('retrieval_queries', [])
    require(queries and len({query['id'] for query in queries}) == len(queries),
            'The fixed reference must contain unique retrieval queries.')
    test_count = sum(passage.get('split') == 'test' for passage in reference['passages'])
    require(test_count > 0, 'The fixed reference has no held-out test passages.')
    for model_id, model in models.items():
        directory = raw / slug(model_id)
        require(all(model.get(key) == specs[model_id].get(key) for key in
                    ('id', 'name', 'license', 'revision', 'profile'))
                and re.fullmatch('[a-f0-9]{40}', model.get('revision', '')) is not None,
                'Unpinned or inconsistent checkpoint: ' + model_id)
        status = read_json(directory / 'status.json')
        require(model.get('status') == status and status.get('model_id') == model_id
                and status.get('status') == 'complete' and status.get('loader_smoke_passed') is True
                and status.get('processed_documents') == status.get('total_documents') == len(corpus['documents'])
                and status.get('processed_blocks') == status.get('total_blocks') == len(corpus['blocks'])
                and status.get('failed_blocks') == status.get('reference_failures') == status.get('query_failures') == 0,
                'Incomplete corpus/reference/query inference: ' + model_id)
        metadata = read_json(directory / 'metadata.json')
        identity_keys = ('model_id', 'revision', 'backbone_revision', 'corpus_snapshot_sha256',
                         'corpus_content_sha256', 'reference_sha256', 'config', 'code_sha256', 'packages')
        require(model.get('metadata') == metadata and metadata.get('schema_version') == 1
                and metadata.get('model_id') == model_id and metadata.get('revision') == model['revision']
                and all(metadata.get(key) == value for key, value in fingerprints.items())
                and metadata.get('reference_sha256') == digest(reference)
                and metadata.get('configuration_sha256') == digest({key: metadata.get(key) for key in identity_keys}),
                'Checkpoint corpus/reference/configuration provenance mismatch: ' + model_id)
        require(metadata.get('config', {}).get('entity_weight') == .25,
                'Unexpected entity-assisted retrieval configuration: ' + model_id)
        predictions = validate_predictions(directory / 'predictions.jsonl', corpus['blocks'], model_id + ' corpus')
        validate_recovery(directory, model, blocks, predictions)
        expected_entities = entity_index(corpus, predictions)
        require(model.get('entities') == expected_entities == read_json(directory / 'entity-index.json'),
                'Entity index does not match complete raw predictions: ' + model_id)
        require(status.get('entity_mentions') == sum(len(row['entities']) for row in predictions.values())
                and status.get('unique_entities') == len(expected_entities), 'Entity coverage counts do not match: ' + model_id)
        del predictions
        reference_predictions = validate_predictions(directory / 'reference-predictions.jsonl', reference['passages'], model_id + ' reference')
        evaluation = evaluate_reference(reference, reference_predictions)
        require(model.get('evaluation') == evaluation == read_json(directory / 'evaluation.json')
                and evaluation['missing_predictions'] == 0 and evaluation['test_passages'] == test_count,
                'Reference metrics do not match the fixed held-out predictions: ' + model_id)
        query_inputs = [{'id': query['id'], 'text': query['query']} for query in queries]
        query_predictions = validate_predictions(directory / 'query-predictions.jsonl', query_inputs, model_id + ' query')
        require(model.get('retrieval') == read_json(directory / 'retrieval.json'), 'Raw/public retrieval mismatch: ' + model_id)
        validate_retrieval(model['retrieval'], queries, query_predictions, blocks, expected_entities)
    excerpts = data.get('excerpts')
    require(isinstance(excerpts, dict) and len(excerpts) <= 350, 'Invalid bounded source excerpts.')
    for block_id, excerpt in excerpts.items():
        block = blocks.get(block_id)
        require(block is not None and excerpt.get('text') == block['text'][:900]
                and excerpt.get('full_text_length') == len(block['text'])
                and excerpt.get('truncated') is (len(block['text']) > 900)
                and all(excerpt.get(key) == block.get(key) for key in
                        ('id', 'document_id', 'ticker', 'fiscal_period', 'source_url', 'section', 'kind', 'page')),
                'Source excerpt does not match its original corpus block: ' + block_id)
    return data


def trusted_replay_url(url, tenant_id, dataset_id, session_ids):
    """Native comparison links must identify exactly the recorded sessions."""
    require(isinstance(url, str), 'Missing native LangSmith replay URL.')
    parsed = urlsplit(url)
    require(parsed.scheme == 'https' and parsed.netloc == 'smith.langchain.com'
            and not parsed.fragment
            and parsed.path == f'/o/{tenant_id}/datasets/{dataset_id}/compare'
            and parse_qs(parsed.query, strict_parsing=True) == {'selectedSessions': [','.join(session_ids)]},
            'Untrusted or mismatched native LangSmith replay URL.')


def merge_langsmith_replay(data, staged, root):
    """Attach optional completed observability replay, never infer its completion."""
    relative = Path('artifacts/local/ner/langsmith-replay/status.json')
    path = staged / relative if (staged / relative).exists() else root / relative
    if not path.exists():
        return None
    status = read_json(path)
    # The replay helper only reads previously saved predictions. Rebuilding its
    # plan lets us check every declared summary/fingerprint without model loading,
    # network access, new search, or treating SDK upload timestamps as inference.
    spec = importlib.util.spec_from_file_location('ner_replay_import_validation', ROOT / 'scripts/ner_langsmith_replay.py')
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    reference = read_json(root / 'data/ner-reference-sample.json')
    expected = replay.build_plan(staged / 'artifacts/local/ner', reference)
    require(status.get('schema_version') == 1 and status.get('protocol') == replay.PROTOCOL
            and status.get('status') == 'complete' and status.get('no_inference_performed') is True
            and status.get('pending_models') == [] and not expected['pending_models']
            and status.get('additional_evaluator_traces') == 0,
            'LangSmith replay is not a completed saved-output replay.')
    for key in ('fingerprint', 'corpus_snapshot_sha256', 'corpus_content_sha256',
                'reference_sha256', 'reference_status', 'reference_review',
                'heldout_ner_passages', 'retrieval_queries', 'heldout_retrieval_queries', 'target_run_count'):
        require(status.get(key) == expected[key], 'LangSmith replay provenance/count mismatch: ' + key)
    require(status.get('completed_replay_runs') == expected['target_run_count'],
            'LangSmith replay run count is incomplete.')
    actual_models = unique_by_id(status.get('models'), MODEL_IDS, 'LangSmith replay checkpoint models', 'model_id')
    expected_models = {model['model_id']: model for model in expected['models']}
    public_models = {model['id']: model for model in data['models']}
    completed, sessions, tenants, datasets = [], set(), set(), {}
    for model_id, model in actual_models.items():
        wanted = expected_models[model_id]
        require(model.get('revision') == wanted['revision']
                and model.get('configuration_sha256') == wanted['configuration_sha256']
                and model.get('recovery_sha256') == wanted.get('recovery_sha256')
                and model.get('recovery_sha256') == public_models[model_id].get('recovery', {}).get('recovery_sha256'),
                'LangSmith replay checkpoint configuration mismatch: ' + model_id)
        experiments = model.get('experiments')
        required_variants = {('ner', 'heldout'), ('retrieval', 'lexical'), ('retrieval', 'entity_assisted')}
        require(isinstance(experiments, list) and len(experiments) == 3
                and {(experiment.get('task'), experiment.get('variant')) for experiment in experiments} == required_variants,
                'LangSmith replay requires held-out NER and both retrieval experiments: ' + model_id)
        wanted_experiments = {(experiment['task'], experiment['variant']): experiment for experiment in wanted['experiments']}
        for experiment in experiments:
            wanted_experiment = wanted_experiments[(experiment['task'], experiment['variant'])]
            fingerprint = replay.experiment_fingerprint(expected['fingerprint'], wanted, wanted_experiment)
            require(experiment.get('fingerprint') == fingerprint
                    and experiment.get('recovery_sha256') == wanted.get('recovery_sha256')
                    and experiment.get('rows') == len(wanted_experiment['rows'])
                    and experiment.get('summary') == wanted_experiment['summary']
                    and experiment.get('dataset_name') == 'workbench-' + wanted_experiment['dataset_key'] + '-' + expected['fingerprint'][:12]
                    and experiment.get('name') == slug(model_id) + '-' + experiment['task'] + '-' + experiment['variant'] + '-' + fingerprint[:10],
                    'LangSmith replay experiment differs from saved outputs: ' + model_id)
            for key in ('id', 'dataset_id', 'tenant_id'):
                value = experiment.get(key)
                require(isinstance(value, str) and str(UUID(value)) == value,
                        'Invalid native LangSmith replay identity: ' + key)
            require(experiment['id'] not in sessions, 'Duplicate native LangSmith replay session.')
            sessions.add(experiment['id'])
            tenants.add(experiment['tenant_id'])
            dataset_key = wanted_experiment['dataset_key']
            require(datasets.setdefault(dataset_key, experiment['dataset_id']) == experiment['dataset_id'],
                    'LangSmith replay comparison datasets do not match.')
            trusted_replay_url(experiment.get('url'), experiment['tenant_id'], experiment['dataset_id'], [experiment['id']])
            completed.append(experiment)
    require(len(tenants) == 1 and len(set(datasets.values())) == 2,
            'LangSmith replay experiments must share one workspace and the two declared datasets.')
    require(status.get('comparison_links') == replay.comparison_links(completed),
            'LangSmith replay comparison links do not match recorded experiments.')
    for task, url in status['comparison_links'].items():
        chosen = [experiment for experiment in completed if experiment['task'] == task]
        trusted_replay_url(url, chosen[0]['tenant_id'], chosen[0]['dataset_id'], [experiment['id'] for experiment in chosen])
    for model_id, model in actual_models.items():
        public_model = public_models[model_id]
        legacy = public_model.get('langsmith') or {}
        require(isinstance(legacy, dict), 'Invalid original LangSmith experiment metadata.')
        # Preserve the original 40-example experiment and add a separately scoped
        # record for the completed held-out/retrieval replay, with its own links.
        public_model['langsmith'] = {**legacy, 'replay': {
            **{key: status[key] for key in ('schema_version', 'protocol', 'status', 'fingerprint',
                                          'no_inference_performed', 'corpus_snapshot_sha256',
                                          'corpus_content_sha256', 'reference_sha256',
                                          'reference_status', 'reference_review', 'heldout_ner_passages',
                                          'retrieval_queries', 'heldout_retrieval_queries', 'comparison_links')},
            **{key: model[key] for key in ('model_id', 'revision', 'configuration_sha256', 'experiments')},
            **({'recovery_sha256': model['recovery_sha256']} if model.get('recovery_sha256') else {}),
            'completed_replay_runs': sum(experiment['rows'] for experiment in model['experiments']),
            'total_completed_replay_runs': status['completed_replay_runs'],
            'finished_at': status.get('finished_at'),
        }}
    return path


def public_exports(data, archive_hash):
    indexes, summaries = {}, []
    for model in data['models']:
        entities = model.pop('entities')
        encoded = json.dumps({'schema_version': 1, 'model_id': model['id'],
                              'corpus_content_sha256': data['corpus_content_sha256'], 'entities': entities},
                             ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        index_hash = hashlib.sha256(encoded).hexdigest()
        name = slug(model['id']) + '.' + index_hash[:16] + '.json'
        indexes[name] = encoded
        model.update(entities=[], entity_index_url='data/ner-index/' + name,
                     entity_count=len(entities), entity_index_sha256=index_hash)
        summaries.append({'model_id': model['id'], 'reports': model['status']['processed_documents'],
                          'blocks': model['status']['processed_blocks'], 'entities': len(entities),
                          'index_bytes': len(encoded), 'metrics': model['evaluation']['metrics']})
    data['publication_export'] = {'indexes': 'Loaded one checkpoint at a time from actual complete corpus postings.',
                                  'source': 'Validated saved model outputs; no placeholder scores.',
                                  'original_archive_sha256': archive_hash}
    summary = (json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')
    require(not any(CREDENTIAL_PATTERN.search(encoded) for encoded in [summary, *indexes.values()]),
            'Credential-like text in public export.')
    return summary, indexes, summaries


def atomic_write(path, encoded):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def import_results(archive, root=ROOT):
    root, archive = Path(root).resolve(), Path(archive).resolve()
    with archive.open('rb') as stream:
        archive_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    staged = stage_archive(archive, root, archive_hash)
    try:
        data = validate_bundle(staged, root)
        replay_path = merge_langsmith_replay(data, staged, root)
        summary, indexes, summaries = public_exports(data, archive_hash)
    except Exception as error:
        raise ValueError(f'{error} Incoming artifacts retained privately at {staged}; current published and raw files were not changed.') from error
    public = root / PUBLIC_PATH
    # Immutable, content-addressed index names keep the previous summary usable
    # until the last atomic replace publishes the new summary and its references.
    for name, encoded in indexes.items():
        atomic_write(public.parent / 'ner-index' / name, encoded)
    # Preserve both the untouched incoming archive and the previous raw snapshot.
    # A failed final summary replace restores the previous current raw directory.
    current_raw = root / 'artifacts/local/ner'
    candidate = staged / 'validated-current-ner'
    previous = staged / 'previous-current-ner'
    shutil.copytree(staged / 'artifacts/local/ner', candidate)
    if replay_path is not None and replay_path != staged / 'artifacts/local/ner/langsmith-replay/status.json':
        # A separately downloaded replay can accompany an older original ZIP.
        # Keep its validated status with the promoted raw artifacts on re-import.
        shutil.copytree(replay_path.parent, candidate / 'langsmith-replay', dirs_exist_ok=True)
    had_previous = current_raw.exists()
    if had_previous:
        current_raw.replace(previous)
    try:
        candidate.replace(current_raw)
        atomic_write(public, summary)
    except Exception:
        if current_raw.exists():
            current_raw.replace(staged / 'unpublished-validated-ner')
        if had_previous:
            previous.replace(current_raw)
        raise
    return {'public_summary_bytes': len(summary), 'models': summaries, 'private_import_directory': str(staged)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    print(json.dumps(import_results(args.archive), indent=2))


if __name__ == '__main__':
    main()
