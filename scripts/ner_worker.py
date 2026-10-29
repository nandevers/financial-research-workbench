"""One pinned NER model in an isolated process. Never invokes a model API."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'engine/src'))
from workbench.ner import (LABELS, SCHEMA_VERSION, EntityRetrieval, append_jsonl, corpus_fingerprints, digest, enforce_resume, entity_index, evaluate_reference, known_positive_retrieval, latest_records, load_json, merge_entities, normalize_entities, now, save_json, slug, token_windows, validate_reference)

# Full-corpus spans stay on disk. LangSmith receives only the bounded evaluation
# sample when explicitly enabled, avoiding 130k hosted corpus traces.
os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def source_hash():
    return hashlib.sha256(b''.join((ROOT/p).read_bytes().replace(b'\r\n', b'\n') for p in ('engine/src/workbench/ner.py', 'scripts/ner_worker.py'))).hexdigest()


class LocalTagger:
    def __init__(self, spec, device, threshold):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        self.spec, self.device, self.threshold = spec, device, threshold
        self.torch = torch
        download_started=time.perf_counter()
        path = snapshot_download(spec['id'], revision=spec['revision'])
        self.download_seconds=time.perf_counter()-download_started
        self.snapshot = path
        kind = spec['kind']
        if kind == 'token':
            self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
            self.model = AutoModelForTokenClassification.from_pretrained(path, local_files_only=True).to(device).eval()
        elif kind in ('gliner', 'nuner'):
            from gliner import GLiNER
            # Pin the missing secondary tokenizer/config dependency too. A local
            # copy of the small GLiNER config points to that immutable snapshot;
            # checkpoint tensors remain the original pinned snapshot files.
            download_started=time.perf_counter()
            backbone = snapshot_download(spec['backbone'], revision=spec['backbone_revision'], allow_patterns=['config.json','tokenizer*','special_tokens_map.json','added_tokens.json','vocab*','merges.txt','spm.model','sentencepiece*'])
            self.download_seconds+=time.perf_counter()-download_started
            self.tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True, local_files_only=True)
            prepared = Path(spec['load_directory'])
            prepared.mkdir(parents=True, exist_ok=True)
            for file in Path(path).iterdir():
                if file.is_file() and file.name != 'gliner_config.json':
                    dest = prepared/file.name
                    if not dest.exists():
                        try:
                            dest.symlink_to(file.resolve())
                        except OSError:
                            shutil.copy2(file, dest)
            config = load_json(Path(path)/'gliner_config.json')
            config['model_name'] = str(Path(backbone).resolve())
            save_json(prepared/'gliner_config.json', config)
            self.model = GLiNER.from_pretrained(str(prepared), local_files_only=True).to(device).eval()
        elif kind == 'gliner2':
            from gliner2 import GLiNER2
            self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
            # The local runtime is explicit; never instantiate gliner2.API.
            self.model = GLiNER2.from_pretrained(path, map_location=device, quantize=False)
        else:
            raise ValueError('Unsupported local model kind: '+kind)
        if not getattr(self.tokenizer, 'is_fast', False):
            raise RuntimeError('A fast tokenizer with exact offsets is required')

    def predict(self, texts):
        torch = self.torch
        kind = self.spec['kind']
        with torch.inference_mode():
            if kind == 'token':
                encoded = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=False, return_offsets_mapping=True)
                offsets = encoded.pop('offset_mapping').tolist()
                if encoded['input_ids'].shape[1] > self.model.config.max_position_embeddings:
                    raise RuntimeError('Token window would exceed model positions; refusing silent truncation')
                logits = self.model(**{k:v.to(self.device) for k,v in encoded.items()}).logits
                probabilities = logits.softmax(-1)
                confidence, classes = probabilities.max(-1)
                confidence, classes = confidence.cpu().tolist(), classes.cpu().tolist()
                batches = []
                for text, positions, predictions, scores_ in zip(texts, offsets, classes, confidence):
                    spans, current = [], None
                    for (start,end), cls, score in zip(positions,predictions,scores_):
                        label = self.model.config.id2label[int(cls)]
                        if not end > start or label == 'O':
                            if current: spans.append(current); current=None
                            continue
                        prefix, _, tag = label.partition('-')
                        if not tag: tag=prefix; prefix='B'
                        if prefix=='I' and current and current['label']==tag:
                            current.update(end=end, score=min(current['score'],score))
                        else:
                            if current: spans.append(current)
                            current={'start':start,'end':end,'label':tag,'score':score}
                    if current: spans.append(current)
                    batches.append([s for s in spans if s['score']>=self.threshold])
                return batches
            if kind in ('gliner','nuner'):
                return self.model.batch_predict_entities(texts, ['person','organization','location'], flat_ner=True, threshold=self.threshold)
            rows = self.model.batch_extract_entities(texts, ['person','organization','location'], batch_size=len(texts), threshold=self.threshold, include_confidence=True, include_spans=True)
            return [[{'start':e['start'],'end':e['end'],'label':label,'score':e['confidence']} for label,entities in row['entities'].items() for e in entities] for row in rows]


def adaptive_predict(tagger, texts, batch_size):
    """Split failed/OOM batches down to single inputs; never drop an input."""
    results=[]
    def run(batch):
        try:
            predicted=tagger.predict(batch)
            if len(predicted)!=len(batch):
                raise ValueError('Model returned a different number of input rows')
            return [{'entities':r} for r in predicted]
        except Exception as error:
            if tagger.torch.cuda.is_available(): tagger.torch.cuda.empty_cache()
            if len(batch)>1:
                middle=len(batch)//2
                return run(batch[:middle])+run(batch[middle:])
            return [{'error':type(error).__name__+': '+str(error)[:700]}]
    for i in range(0,len(texts),batch_size):
        results.extend(run(texts[i:i+batch_size]))
    return results


def infer_items(tagger, items, path, config, progress=None):
    existing=latest_records(path)
    pending=[]
    for item in items:
        old=existing.get(item['id'])
        text_hash=digest(item['text'])
        if old and old.get('text_sha256')!=text_hash:
            raise ValueError('Text changed for saved prediction '+item['id'])
        if not old or old.get('status')!='complete': pending.append(item)
    batch_size=config['batch_size']
    for start in range(0,len(pending),batch_size):
        group=pending[start:start+batch_size]
        flattened, windows_by_item=[],[]
        for item in group:
            windows=token_windows(item['text'],tagger.tokenizer,config['max_tokens'],config['overlap_tokens'])
            windows_by_item.append(windows); flattened.extend(w['text'] for w in windows)
        if tagger.torch.cuda.is_available(): tagger.torch.cuda.synchronize()
        before=time.perf_counter()
        predictions=adaptive_predict(tagger,flattened,batch_size)
        if tagger.torch.cuda.is_available(): tagger.torch.cuda.synchronize()
        elapsed=time.perf_counter()-before
        cursor=0; records=[]
        for item,windows in zip(group,windows_by_item):
            outputs=predictions[cursor:cursor+len(windows)];cursor+=len(windows)
            errors=[r['error'] for r in outputs if 'error' in r]
            entities=[]
            if not errors:
                try:
                    for window,result in zip(windows,outputs):
                        entities.extend(normalize_entities(window['text'],result['entities'],window['start']))
                    entities=merge_entities(item['text'],entities,merge_adjacent=tagger.spec['kind']=='nuner')
                except Exception as error:
                    errors=[type(error).__name__+': '+str(error)[:700]]
            record={'schema_version':SCHEMA_VERSION,'id':item['id'],'document_id':item.get('document_id'),'text_sha256':digest(item['text']),'text_length':len(item['text']),'status':'failed' if errors else 'complete','window_count':len(windows),'covered_characters':len(item['text']) if not errors else None,'entities':entities if not errors else [],'errors':errors,'inference_seconds_allocated':elapsed*len(windows)/len(flattened),'timing_note':'Batch inference seconds allocated in proportion to windows, not separately timed per block.','recorded_at':now()}
            records.append(record);existing[item['id']]=record
        append_jsonl(path,records)
        if progress: progress(existing)
    return existing


def upload_langsmith(reference, predictions, spec, evaluation, output):
    if not os.environ.get('LANGSMITH_API_KEY'):
        raise RuntimeError('--langsmith requires LANGSMITH_API_KEY for observability only')
    from langsmith import Client
    from langsmith.evaluation import evaluate
    client=Client()
    fingerprint=digest(reference)[:12]
    dataset_name='workbench-ner-reference-'+fingerprint
    datasets=list(client.list_datasets(dataset_name=dataset_name))
    review_description=('Agent-authored and separately agent-reviewed' if reference.get('review',{}).get('second_review_status')=='completed' else 'Agent-authored; second-agent review not completed')
    dataset=datasets[0] if datasets else client.create_dataset(dataset_name=dataset_name,description=review_description+' financial report entity spans; not human gold. Existing corpus only. Reference status: '+str(reference.get('status')))
    if not datasets:
        client.create_examples(inputs=[{'passage_id':p['id'],'text':p['text'],'split':p.get('split')} for p in reference['passages']], outputs=[{'entities':p['entities']} for p in reference['passages']],dataset_id=dataset.id)
    def target(inputs):
        record=predictions.get(inputs['passage_id'])
        if not record or record['status']!='complete':
            raise RuntimeError('Recorded local inference failed or missing')
        return record
    def metrics(run,example):
        from workbench.ner import span_counts,scores
        measured=scores(span_counts(example.outputs['entities'],(run.outputs or {}).get('entities',[])))
        return {'results':[{'key':'entity_exact_'+key,'score':measured[key],'comment':'Agent-authored sample; not independent gold.'} for key in ('precision','recall','f1')]}
    experiment=evaluate(target,data=dataset_name,evaluators=[metrics],experiment_prefix=slug(spec['id'])+'-ner',metadata={'checkpoint_revision':spec['revision'],'task':'ner','local_predictions_replayed':True,'reference_status':reference.get('status')},max_concurrency=1)
    result={'dataset_id':str(dataset.id),'dataset_name':dataset_name,'experiment_name':experiment.experiment_name,'source':'Previously recorded local inference, uploaded without rerunning the model.','evaluation':evaluation.get('metrics')}
    save_json(output/'langsmith.json',result)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--corpus',required=True);parser.add_argument('--manifest',required=True);parser.add_argument('--model',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--reference');parser.add_argument('--device',default='cuda');parser.add_argument('--batch-size',type=int,default=16);parser.add_argument('--max-tokens',type=int,default=320);parser.add_argument('--overlap-tokens',type=int,default=48);parser.add_argument('--threshold',type=float,default=.5);parser.add_argument('--langsmith',action='store_true')
    args=parser.parse_args()
    corpus=load_json(args.corpus);manifest=load_json(args.manifest)
    spec=dict(next(s for s in manifest['models'] if s['id']==args.model))
    output=Path(args.output)/slug(spec['id']);output.mkdir(parents=True,exist_ok=True)
    spec['load_directory']=str(output/'pinned-load')
    config={'batch_size':args.batch_size,'max_tokens':args.max_tokens,'overlap_tokens':args.overlap_tokens,'threshold':args.threshold,'precision_policy':'native local loader; actual parameter dtypes recorded after load','device':args.device,'labels':LABELS,'nuner_merge_adjacent':True,'overlap_deduplication':'longest same-type overlapping span, then confidence','entity_weight':.25}
    packages={}
    for name in ('torch','transformers','gliner','gliner2','huggingface-hub','scikit-learn'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    reference=load_json(args.reference) if args.reference else None
    if reference:validate_reference(corpus,reference)
    identity={'model_id':spec['id'],'revision':spec['revision'],'backbone_revision':spec.get('backbone_revision'),**corpus_fingerprints(corpus),'reference_sha256':digest(reference) if reference else None,'config':config,'code_sha256':source_hash(),'packages':packages}
    metadata={'schema_version':SCHEMA_VERSION,**identity,'configuration_sha256':digest(identity),'created_at':now(),'python':platform.python_version(),'source':'Actual local checkpoint inference; no hosted inference API.'}
    enforce_resume(output,metadata)
    if (output/'status.json').exists():
        saved_status=load_json(output/'status.json')
        if saved_status.get('status')=='complete' and (not args.langsmith or (output/'langsmith.json').exists()):
            print(json.dumps({'model_id':spec['id'],'status':'complete','resume':'Already complete under the identical verified configuration; no repeated inference or LangSmith experiment.'}),flush=True)
            return 0
    status={'schema_version':SCHEMA_VERSION,'model_id':spec['id'],'status':'loading','total_documents':len(corpus['documents']),'total_blocks':len(corpus['blocks']),'processed_documents':0,'processed_blocks':0,'failed_blocks':0,'updated_at':now()}
    save_json(output/'status.json',status)
    try:
        started=time.perf_counter();tagger=LocalTagger(spec,args.device,args.threshold)
        status['model_download_seconds']=tagger.download_seconds
        status['model_load_seconds']=time.perf_counter()-started-tagger.download_seconds
        status['hardware']=tagger.torch.cuda.get_device_name(0) if args.device.startswith('cuda') else platform.processor() or 'CPU'
        parameters=getattr(tagger.model,'parameters',None)
        status['parameter_dtypes']=sorted({str(p.dtype) for p in parameters()}) if callable(parameters) else ['Not exposed by local loader']
        # One actual short warm-up must succeed before attempting the entire corpus.
        warmup=tagger.predict(['Bank of America is based in the United States.'])
        if len(warmup)!=1:raise RuntimeError('Local loader smoke check returned invalid shape')
        status['loader_smoke_passed']=True
        if tagger.torch.cuda.is_available():tagger.torch.cuda.reset_peak_memory_stats()
        previous=latest_records(output/'predictions.jsonl')
        prior_seconds=sum(r.get('inference_seconds_allocated',0) for r in previous.values())
        before=time.perf_counter()
        def progress(records):
            complete=[r for r in records.values() if r['status']=='complete']
            status.update(status='processing',processed_blocks=len(complete),processed_documents=len({r['document_id'] for r in complete}),failed_blocks=sum(r['status']=='failed' for r in records.values()),corpus_inference_seconds=prior_seconds+time.perf_counter()-before,updated_at=now())
            save_json(output/'status.json',status)
            print(json.dumps({k:status[k] for k in ('model_id','status','processed_blocks','total_blocks','failed_blocks')}),flush=True)
        records=infer_items(tagger,corpus['blocks'],output/'predictions.jsonl',config,progress)
        progress(records)
        status['corpus_inference_seconds']=sum(r.get('inference_seconds_allocated',0) for r in records.values())
        status['peak_gpu_memory_gb']=tagger.torch.cuda.max_memory_allocated()/1024**3 if args.device.startswith('cuda') else None
        entries=entity_index(corpus,records);save_json(output/'entity-index.json',entries)
        reference_predictions={}
        if reference:
            reference_predictions=infer_items(tagger,reference['passages'],output/'reference-predictions.jsonl',config)
            evaluation=evaluate_reference(reference,reference_predictions);save_json(output/'evaluation.json',evaluation)
        queries=reference.get('retrieval_queries',[]) if reference else []
        if not queries:
            queries=[{'id':f'issuer-{i}','query':f'What does {name} disclose about liquidity, capital, reinsurance or catastrophe risk?'} for i,name in enumerate(('Bank of America','JPMorgan Chase','Chubb Limited','The Travelers Companies'))]
        query_inputs=[{'id':q['id'],'text':q['query']} for q in queries]
        query_predictions=infer_items(tagger,query_inputs,output/'query-predictions.jsonl',config)
        retrieval=EntityRetrieval(corpus,entries);searches=[]
        for query in queries:
            prediction=query_predictions[query['id']]
            if prediction['status']!='complete':
                searches.append({'id':query['id'],'query':query['query'],'status':'query_ner_failed','error':prediction['errors'],'lexical':[],'entity_assisted':[]});continue
            measured=retrieval.rank(query['query'],prediction['entities'],k=5)
            searches.append({'id':query['id'],'query':query['query'],'status':'complete','query_entities':prediction['entities'],**measured,'known_positive_metrics':{name:known_positive_retrieval(query,measured[name]) for name in ('lexical','entity_assisted')}})
        save_json(output/'retrieval.json',searches)
        if args.langsmith and reference:
            try:upload_langsmith(reference,reference_predictions,spec,evaluation,output)
            except Exception as error:
                status['langsmith_error']=type(error).__name__+': '+str(error)[:500]
        failures=sum(r['status']!='complete' for r in records.values())
        reference_missing=sum(r.get('status')!='complete' for r in reference_predictions.values()) if reference else 0
        complete=len(records)==len(corpus['blocks']) and failures==0 and len({r['document_id'] for r in records.values()})==len(corpus['documents']) and all(s['status']=='complete' for s in searches) and reference_missing==0
        status.update(status='complete' if complete else 'partial',failed_blocks=failures,entity_mentions=sum(len(r['entities']) for r in records.values()),unique_entities=len(entries),query_failures=sum(s['status']!='complete' for s in searches),reference_failures=reference_missing,updated_at=now())
        save_json(output/'status.json',status)
        print(json.dumps(status),flush=True)
        return 0 if complete else 2
    except Exception as error:
        status.update(status='failed',error=type(error).__name__+': '+str(error)[:1200],updated_at=now());save_json(output/'status.json',status)
        raise


if __name__=='__main__':
    sys.exit(main())
