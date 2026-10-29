"""Append audited recoveries for GLiNER2's proven synthetic terminal period only.

Frozen inference sources and base metadata are never changed. Completed records
are retained byte-for-byte. Only the originally failed blocks are inferred again.
The new producer is identified on every appended record and in recovery.json.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'engine/src'))
from workbench.ner import (EntityRetrieval, corpus_fingerprints, digest, entity_index, evaluate_reference, known_positive_retrieval, load_json, merge_entities, normalize_entities, normalize_type, now, read_jsonl, save_json, slug, token_windows, validate_reference)

MODEL_ID='fastino/gliner2-base-v1'
POLICY='gliner2-2.1.0-synthetic-terminal-period-v1'
EXPECTED_PROCESSOR_SHA256='fdeb0814c29a0d1fec4e66358e936058af224d011a37b5c7f1e800cb2cd9c521'
EXPECTED_RUNTIME_SHA256='acc500efe68209df7620c5810b5ff38a4d6151e8799063e0139ec60220183a60'
ARTIFACTS=('predictions.jsonl','status.json','metadata.json','reference-predictions.jsonl','query-predictions.jsonl','evaluation.json','retrieval.json','entity-index.json','langsmith.json')


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stock_normalized_text(text):
    """The inspected 2.1.0 rule; its installed source hash is checked at runtime."""
    if not text:return '.'
    return text if text.endswith(('.', '!', '?')) else text+'.'


def project_native_entities(original_text,native_entities):
    """Prove a native span before mapping the one synthetic suffix to its source.

    No coordinate is shifted. In-bounds spans are returned unchanged. Only an
    exact, verified span intersecting the known appended final period is projected.
    Every other mismatch raises; no generic clipping or native-surface searching.
    """
    if not isinstance(native_entities,dict):raise ValueError('Native entities must be a dictionary')
    augmented=stock_normalized_text(original_text)
    has_suffix=augmented==original_text+'.'
    projected,decisions=[],[]
    for native_label,entities in native_entities.items():
        label=normalize_type(native_label)
        if label is None:raise ValueError('Unexpected native entity label')
        if not isinstance(entities,list):raise ValueError('Native entity collection must be a list')
        for entity in entities:
            start,end=entity.get('start'),entity.get('end')
            if isinstance(start,bool) or isinstance(end,bool) or not isinstance(start,int) or not isinstance(end,int):
                raise ValueError('Native offsets must be integer character boundaries')
            surface=entity.get('text')
            if not isinstance(surface,str) or not 0<=start<end<=len(augmented):
                raise ValueError('Native span outside proven normalized input')
            if surface!=augmented[start:end].strip():
                raise ValueError('Native surface does not match its proven normalized input slice')
            score=float(entity['confidence'])
            if not math.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid native confidence')
            decision={'type':label,'native_label':native_label,'native_text':surface,'native_start':start,'native_end':end}
            if end<=len(original_text):
                decision.update(action='unchanged_original_span',projected_start=start,projected_end=end,projected_text=original_text[start:end])
            elif has_suffix and end==len(original_text)+1 and augmented[-1]=='.':
                if start==len(original_text):
                    decision.update(action='omit_synthetic_only',projected_start=None,projected_end=None,projected_text='')
                elif start<len(original_text):
                    decision.update(action='project_synthetic_period',projected_start=start,projected_end=len(original_text),projected_text=original_text[start:])
                else:raise ValueError('Native entity starts outside original source')
            else:raise ValueError('Out-of-bounds span is not the proven synthetic terminal period')
            decisions.append(decision)
            if decision['action']!='omit_synthetic_only':
                projected.append({'start':decision['projected_start'],'end':decision['projected_end'],'text':decision['projected_text'],'label':label,'score':score})
    return projected,decisions


def latest(path):
    return {record['id']:record for record in read_jsonl(path,recover_tail=False)}


def append_record(path,record):
    with Path(path).open('a',encoding='utf-8') as handle:
        handle.write(json.dumps(record,ensure_ascii=False)+'\n');handle.flush();os.fsync(handle.fileno())


def worker_module():
    spec=importlib.util.spec_from_file_location('frozen_ner_worker_for_recovery',ROOT/'scripts/ner_worker.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def library_evidence():
    import gliner2.processor as processor
    import gliner2.inference.runtime as runtime
    version=importlib.metadata.version('gliner2')
    paths={'gliner2-processor.py':Path(processor.__file__),'gliner2-runtime.py':Path(runtime.__file__)}
    processor_hash,runtime_hash=file_sha(paths['gliner2-processor.py']),file_sha(paths['gliner2-runtime.py'])
    if version!='2.1.0' or processor_hash!=EXPECTED_PROCESSOR_SHA256 or runtime_hash!=EXPECTED_RUNTIME_SHA256:
        raise ValueError('Installed GLiNER2 sources differ from the inspected 2.1.0 normalization proof')
    normalizer=processor.SchemaTransformer._normalize_text
    for text in ('','Acme','Acme.','Acme!','Acme?','Acme ','United States','😀 Corp'):
        if normalizer(text)!=stock_normalized_text(text):raise ValueError('Installed normalization semantics differ')
    normalizer_source=inspect.getsource(normalizer).encode('utf-8')
    result={'version':version,'processor_sha256':processor_hash,'runtime_sha256':runtime_hash,'normalizer_source_sha256':hashlib.sha256(normalizer_source).hexdigest()}
    return result,paths,normalizer_source,normalizer


def validate_base(corpus,reference,manifest,metadata,directory):
    if metadata['model_id']!=MODEL_ID:raise ValueError('Recovery is restricted to the specified GLiNER2 checkpoint')
    spec=next(s for s in manifest['models'] if s['id']==MODEL_ID)
    if spec['revision']!=metadata['revision']:raise ValueError('Checkpoint revision changed')
    for key,value in corpus_fingerprints(corpus).items():
        if metadata.get(key)!=value or manifest.get(key)!=value:raise ValueError('Corpus changed')
    if metadata['reference_sha256']!=digest(reference):raise ValueError('Reference annotations changed')
    validate_reference(corpus,reference)
    keys=('model_id','revision','backbone_revision','corpus_snapshot_sha256','corpus_content_sha256','reference_sha256','config','code_sha256','packages')
    if metadata['configuration_sha256']!=digest({key:metadata.get(key) for key in keys}):raise ValueError('Base configuration identity is invalid')
    worker=worker_module()
    if worker.source_hash()!=metadata['code_sha256']:raise ValueError('Frozen source fingerprint changed')
    for package,version in metadata['packages'].items():
        if importlib.metadata.version(package)!=version:raise ValueError('Base dependency versions changed')
    return spec,worker


def originals(directory):
    backup=directory/'recovery-originals';backup.mkdir(exist_ok=True)
    hashes={}
    for name in ARTIFACTS:
        path=directory/name
        if not path.exists():continue
        destination=backup/name
        if destination.exists():
            if file_sha(path)!=file_sha(destination):raise ValueError('Existing recovery backup differs before recovery identity was committed')
        else:shutil.copy2(path,destination)
        hashes[name]=file_sha(destination)
    source_dir=backup/'source';source_dir.mkdir(exist_ok=True)
    frozen={}
    for relative in ('engine/src/workbench/ner.py','scripts/ner_worker.py'):
        path=ROOT/relative;destination=source_dir/path.name
        if destination.exists() and file_sha(path)!=file_sha(destination):raise ValueError('Frozen source backup differs')
        if not destination.exists():shutil.copy2(path,destination)
        frozen[relative]=file_sha(destination)
    return hashes,frozen


def verify_prefix(directory,manifest):
    identity=manifest['identity']
    raw=(directory/'predictions.jsonl').read_bytes()
    size=identity['original_predictions_bytes']
    if hashlib.sha256(raw[:size]).hexdigest()!=identity['original_artifact_sha256']['predictions.jsonl']:
        raise ValueError('Original prediction-file prefix was changed')
    if file_sha(directory/'metadata.json')!=identity['original_artifact_sha256']['metadata.json']:
        raise ValueError('Frozen base metadata was changed')
    if manifest['recovery_sha256']!=digest(identity):raise ValueError('Recovery identity hash is invalid')


def recover(corpus,reference,manifest,metadata,directory):
    spec,worker=validate_base(corpus,reference,manifest,metadata,directory)
    evidence,library_paths,normalizer_source,normalizer=library_evidence()
    sources=directory/'recovery-sources';sources.mkdir(exist_ok=True)
    source_bytes={'ner_recover_gliner2.py':Path(__file__).read_bytes(),'gliner2-normalize-text.py':normalizer_source,**{name:path.read_bytes() for name,path in library_paths.items()}}
    for name,value in source_bytes.items():
        destination=sources/name
        if destination.exists() and destination.read_bytes()!=value:raise ValueError('Recovery source snapshot changed')
        if not destination.exists():destination.write_bytes(value)
    path=directory/'recovery.json'
    if path.exists():
        recovery=load_json(path);verify_prefix(directory,recovery)
        if recovery['identity']['helper_source_sha256']!=file_sha(__file__) or recovery['identity']['library']!=evidence:
            raise ValueError('Recovery producer changed; refusing mixed recovery')
        if recovery.get('status')=='complete':
            if file_sha(directory/'predictions.jsonl')!=recovery['final_predictions_sha256']:raise ValueError('Completed recovery predictions changed')
            for name,expected in recovery['post_artifact_sha256'].items():
                if file_sha(directory/name)!=expected:raise ValueError('Completed recovery output changed: '+name)
            return recovery
    else:
        records=latest(directory/'predictions.jsonl')
        bad=[r['id'] for r in records.values() if r.get('status')!='complete']
        if not bad:raise ValueError('There are no failed blocks to recover')
        if any(any(not error.startswith('ValueError: Entity offset outside its input:') for error in record.get('errors',[])) for record in records.values() if record['id'] in bad):
            raise ValueError('An original failure is outside the inspected offset error class')
        hashes,frozen=originals(directory)
        identity={'model_id':MODEL_ID,'revision':spec['revision'],'base_configuration_sha256':metadata['configuration_sha256'],'base_code_sha256':metadata['code_sha256'],'corpus_snapshot_sha256':metadata['corpus_snapshot_sha256'],'corpus_content_sha256':metadata['corpus_content_sha256'],'reference_sha256':metadata['reference_sha256'],'helper_source_sha256':file_sha(__file__),'library':evidence,'original_artifact_sha256':hashes,'frozen_source_sha256':frozen,'original_predictions_bytes':(directory/'predictions.jsonl').stat().st_size,'failed_block_ids':bad,'inference_config':metadata['config'],'window_batch_policy':'One window per local model call; does not exceed the immutable base maximum batch size.'}
        recovery={'schema_version':1,'policy':POLICY,'status':'recovering','identity':identity,'recovery_sha256':digest(identity),'started_at':now(),'append_records':[],'original_success_count':sum(r.get('status')=='complete' for r in records.values()),'remaining_failed_count':len(bad)}
        save_json(path,recovery)
    verify_prefix(directory,recovery)
    original_records=latest(directory/'recovery-originals/predictions.jsonl')
    records=latest(directory/'predictions.jsonl')
    audits=read_jsonl(directory/'recovery-native.jsonl',recover_tail=False)
    audit_by_sha={digest(audit):audit for audit in audits}
    spec={**spec,'load_directory':str(directory/'pinned-load')}
    config=metadata['config'];load_started=time.perf_counter()
    tagger=worker.LocalTagger(spec,config['device'],config['threshold'])
    load_seconds=time.perf_counter()-load_started
    blocks={block['id']:block for block in corpus['blocks']}
    recovered=[]
    for block_id in recovery['identity']['failed_block_ids']:
        parent=original_records[block_id];current=records.get(block_id)
        if current and current.get('status')=='complete':
            audit_hash=current.get('native_audit_record_sha256')
            if current.get('producer_recovery_sha256')!=recovery['recovery_sha256'] or current.get('parent_prediction_sha256')!=digest(parent) or audit_hash not in audit_by_sha:
                raise ValueError('Unexpected producer in recovered prediction suffix')
            recovered.append({'block_id':block_id,'parent_prediction_sha256':digest(parent),'appended_prediction_sha256':digest(current),'native_audit_record_sha256':audit_hash});continue
        block=blocks[block_id]
        if parent['text_sha256']!=digest(block['text']):raise ValueError('Originally failed block text changed')
        windows=token_windows(block['text'],tagger.tokenizer,config['max_tokens'],config['overlap_tokens'])
        entities=[];window_audits=[]
        if tagger.torch.cuda.is_available():tagger.torch.cuda.synchronize()
        started=time.perf_counter()
        for window in windows:
            normalized=normalizer(window['text'])
            if normalized!=stock_normalized_text(window['text']):raise ValueError('Installed preprocessing changed this input beyond the proved suffix')
            with tagger.torch.inference_mode():
                payload=tagger.model.batch_extract_entities([window['text']],['person','organization','location'],batch_size=1,threshold=config['threshold'],include_confidence=True,include_spans=True)
            if len(payload)!=1 or not isinstance(payload[0].get('entities'),dict):raise ValueError('Invalid native result shape')
            projected,decisions=project_native_entities(window['text'],payload[0]['entities'])
            entities.extend(normalize_entities(window['text'],projected,window['start']))
            window_audits.append({'start':window['start'],'end':window['end'],'input_sha256':digest(window['text']),'input_length':len(window['text']),'normalized_sha256':digest(normalized),'native_entities':payload[0]['entities'],'projection_decisions':decisions})
        if tagger.torch.cuda.is_available():tagger.torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        entities=merge_entities(block['text'],entities,merge_adjacent=False)
        audit={'schema_version':1,'block_id':block_id,'producer_recovery_sha256':recovery['recovery_sha256'],'parent_prediction_sha256':digest(parent),'text_sha256':digest(block['text']),'windows':window_audits,'inference_seconds':elapsed,'recorded_at':now()}
        audit_hash=digest(audit);append_record(directory/'recovery-native.jsonl',audit);audit_by_sha[audit_hash]=audit
        record={'schema_version':1,'id':block_id,'document_id':block['document_id'],'text_sha256':digest(block['text']),'text_length':len(block['text']),'status':'complete','window_count':len(windows),'covered_characters':len(block['text']),'entities':entities,'errors':[],'inference_seconds_allocated':elapsed,'timing_note':'Actual targeted recovery block time; its windows were inferred sequentially at batch size one. Original failed attempt remains in the JSONL history.','recorded_at':now(),'producer_recovery_sha256':recovery['recovery_sha256'],'base_configuration_sha256':metadata['configuration_sha256'],'parent_prediction_sha256':digest(parent),'native_audit_record_sha256':audit_hash}
        append_record(directory/'predictions.jsonl',record);records[block_id]=record
        recovered.append({'block_id':block_id,'parent_prediction_sha256':digest(parent),'appended_prediction_sha256':digest(record),'native_audit_record_sha256':audit_hash})
        recovery.update(append_records=recovered,remaining_failed_count=len(recovery['identity']['failed_block_ids'])-len(recovered),updated_at=now());save_json(path,recovery)
        print(json.dumps({'model_id':MODEL_ID,'recovered_blocks':len(recovered),'target_blocks':len(recovery['identity']['failed_block_ids'])}),flush=True)
    if set(records)!={b['id'] for b in corpus['blocks']} or any(r['status']!='complete' for r in records.values()):raise ValueError('Whole-corpus coverage is not complete after targeted recovery')
    entries=entity_index(corpus,records);save_json(directory/'entity-index.json',entries)
    references=latest(directory/'reference-predictions.jsonl');evaluation=evaluate_reference(reference,references)
    if evaluation!=load_json(directory/'recovery-originals/evaluation.json'):raise ValueError('Unchanged reference predictions produced a different evaluation')
    # Query NER is reused exactly; only deterministic rankings use the recovered
    # full-corpus index. No query inference or LangSmith experiment is repeated.
    query_predictions=latest(directory/'query-predictions.jsonl');retriever=EntityRetrieval(corpus,entries);searches=[]
    for query in reference['retrieval_queries']:
        prediction=query_predictions[query['id']]
        if prediction['status']!='complete' or prediction['text_sha256']!=digest(query['query']):raise ValueError('Saved query prediction is invalid')
        measured=retriever.rank(query['query'],prediction['entities'],k=5)
        searches.append({'id':query['id'],'query':query['query'],'status':'complete','query_entities':prediction['entities'],**measured,'known_positive_metrics':{approach:known_positive_retrieval(query,measured[approach]) for approach in ('lexical','entity_assisted')}})
    save_json(directory/'retrieval.json',searches)
    old_status=load_json(directory/'recovery-originals/status.json')
    recovery_seconds=sum(audit_by_sha[row['native_audit_record_sha256']]['inference_seconds'] for row in recovered)
    status={**old_status,'status':'complete','processed_blocks':len(records),'processed_documents':len({r['document_id'] for r in records.values()}),'failed_blocks':0,'reference_failures':0,'query_failures':0,'entity_mentions':sum(len(r['entities']) for r in records.values()),'unique_entities':len(entries),'corpus_inference_seconds':old_status.get('corpus_inference_seconds',0)+recovery_seconds,'recovery_model_load_seconds':load_seconds,'recovery_inference_seconds':recovery_seconds,'recovery':{'policy':POLICY,'recovery_sha256':recovery['recovery_sha256'],'recovered_blocks':len(recovered),'original_success_count':recovery['original_success_count'],'base_configuration_sha256':metadata['configuration_sha256']},'updated_at':now()}
    save_json(directory/'status.json',status)
    verify_prefix(directory,recovery)
    recovery.update(status='complete',append_records=recovered,remaining_failed_count=0,final_predictions_sha256=file_sha(directory/'predictions.jsonl'),post_artifact_sha256={name:file_sha(directory/name) for name in ARTIFACTS if name!='predictions.jsonl' and (directory/name).exists()},finished_at=now())
    save_json(path,recovery);return recovery


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',default='artifacts/local/corpus.json');parser.add_argument('--reference',default='data/ner-reference-sample.json');parser.add_argument('--output',default='artifacts/local/ner');parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();output=Path(args.output).resolve();directory=output/slug(MODEL_ID)
    metadata=load_json(directory/'metadata.json');records=latest(directory/'predictions.jsonl')
    failed=[record for record in records.values() if record['status']!='complete']
    print(json.dumps({'model_id':MODEL_ID,'mode':'execute' if args.execute else 'read_only_plan','successful_existing_blocks':sum(r['status']=='complete' for r in records.values()),'failed_block_ids':[r['id'] for r in failed],'base_configuration_sha256':metadata['configuration_sha256'],'frozen_metadata_will_be_changed':False}),flush=True)
    if not args.execute:return 0
    result=recover(load_json(args.corpus),load_json(args.reference),load_json(output/'manifest.json'),metadata,directory)
    print(json.dumps({'status':result['status'],'policy':POLICY,'recovery_sha256':result['recovery_sha256'],'recovered_blocks':len(result['append_records']),'remaining_failed_count':result['remaining_failed_count'],'no_legacy_langsmith_upload':True}),flush=True)
    return 0


def safe_error_message(error):
    if type(error) not in (ValueError,AssertionError,KeyError,FileNotFoundError):
        return 'External runtime error; inspect the local model/runtime diagnostics.'
    message=str(error)[:1000]
    message=re.sub(r'(?i)(bearer\s+)[A-Za-z0-9._~+/-]+',r'\1[REDACTED]',message)
    message=re.sub(r'(?:lsv2_[A-Za-z]+_|hf_|sk-proj-|ghp_)[A-Za-z0-9_-]{8,}','[REDACTED]',message)
    return re.sub(r'(?i)((?:api[_-]?key|access_token|token)=)[^&\s]+',r'\1[REDACTED]',message)


if __name__=='__main__':
    try:sys.exit(main())
    except Exception as error:
        # A diagnostic failure must not silently become successful coverage.
        print(json.dumps({'status':'recovery_stopped','error_type':type(error).__name__,'reason':safe_error_message(error)}),flush=True)
        sys.exit(2)
