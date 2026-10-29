"""Publish saved NER/retrieval outputs to LangSmith, without loading any model.

Default mode only builds a local plan. Pass --upload to write datasets, experiment
sessions, saved-output replay runs and deterministic feedback. Credentials come only
from LANGSMITH_API_KEY in the environment and are never saved or printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid5

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'engine/src'))
from workbench.ner import digest, load_json, normalize_type, read_jsonl, save_json, scores, slug, span_counts

PROTOCOL='saved-ner-heldout-and-retrieval-replay-v1'
NAMESPACE=UUID('f22618e4-7d9b-5ebd-96dc-faa775c387ef')
RETRIEVAL_BOUNDARY='Known positives in an agent-authored, agent-reviewed reference sample only. Unjudged full-corpus hits are not negative. Not exhaustive recall, precision@k, or financial accuracy.'
NER_BOUNDARY='Exact spans/types on held-out agent-authored, separately agent-reviewed reference passages. Not human gold, full-corpus accuracy or financial claim correctness.'
RECOVERY_POLICY='gliner2-2.1.0-synthetic-terminal-period-v1'


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def identity(*parts):
    return str(uuid5(NAMESPACE,':'.join(str(part) for part in parts)))


def predictions_by_id(path):
    # Read-only: unlike the inference resumer, replay must not truncate a partial
    # JSONL tail or repair any frozen inference artifact.
    return {record['id']:record for record in read_jsonl(path,recover_tail=False)}


def validate_entities(text,entities):
    for entity in entities:
        if not 0<=entity['start']<entity['end']<=len(text) or text[entity['start']:entity['end']]!=entity['text']:
            raise ValueError('Saved entity is not an exact input-text span')
        if normalize_type(entity['type']) not in ('PERSON','ORGANIZATION','LOCATION'):
            raise ValueError('Saved entity type is outside the shared task')


def heldout_ner_rows(reference,predictions):
    rows=[]
    for passage in reference['passages']:
        if passage.get('split')!='test':continue
        record=predictions.get(passage['id'])
        if not record or record.get('status')!='complete':
            raise ValueError('Held-out prediction missing or failed: '+passage['id'])
        if record.get('text_sha256')!=digest(passage['text']):
            raise ValueError('Held-out input fingerprint differs: '+passage['id'])
        validate_entities(passage['text'],record['entities'])
        measured=scores(span_counts(passage['entities'],record['entities']))
        rows.append({
            'id':passage['id'],'split':'test',
            'inputs':{'passage_id':passage['id'],'text':passage['text'],'split':'test','source_block_id':passage['source_block_id'],'document_id':passage['document_id']},
            'reference_outputs':{'entities':passage['entities']},
            'outputs':{'entities':record['entities'],'local_recorded_at':record.get('recorded_at'),'local_inference_seconds_allocated':record.get('inference_seconds_allocated'),'status':'complete','recording_mode':'saved_prediction_replay_no_inference'},
            'counts':{key:measured[key] for key in ('tp','fp','fn','uncertain_reference_entities')},
            'feedback':{f'passage_exact_{key}':measured[key] for key in ('precision','recall','f1')},
            'feedback_comment':NER_BOUNDARY+' Native averages of passage scores are macro averages; experiment micro summaries pool TP/FP/FN.',
        })
    if not rows:raise ValueError('No held-out test reference passages')
    return rows


def heldout_micro_summary(rows):
    counts=Counter(tp=0,fp=0,fn=0,uncertain_reference_entities=0)
    for row in rows:counts.update(row['counts'])
    measured=scores(counts)
    return {'sample_passages':len(rows),'aggregation':'Pooled exact entity TP/FP/FN over held-out passages only; micro, not mean passage F1.','boundary':NER_BOUNDARY,**measured}


def retrieval_metric(query,ranking,test_block_ids,k=5):
    """Use held-out positives for test queries and development positives for dev.

    Full-corpus ranks are retained. A hit without an inspected judgment never
    becomes a negative, so this deliberately does not compute precision@k.
    """
    relevant=set(query.get('relevant_block_ids',[]))
    if query.get('split')=='test':relevant.intersection_update(test_block_ids)
    else:relevant.difference_update(test_block_ids)
    if not relevant:
        raise ValueError('Query has no known positives in its declared fold: '+query['id'])
    top=[row['block_id'] for row in ranking[:k]]
    hits=relevant.intersection(top)
    return {'known_positive_recall_at_5':len(hits)/len(relevant),'first_known_positive_reciprocal_rank_at_5':next((1/(i+1) for i,block_id in enumerate(top) if block_id in relevant),0.0),'known_positive_blocks_in_fold':len(relevant),'known_hits_at_5':len(hits),'k':k,'unjudged_policy':'unjudged_not_negative','boundary':RETRIEVAL_BOUNDARY}


def retrieval_rows(reference,retrieval,query_predictions,approach):
    by_id={row['id']:row for row in retrieval}
    test_block_ids={p['source_block_id'] for p in reference['passages'] if p.get('split')=='test'}
    rows=[]
    for query in reference['retrieval_queries']:
        saved=by_id.get(query['id']);prediction=query_predictions.get(query['id'])
        if not saved or saved.get('status')!='complete' or not prediction or prediction.get('status')!='complete':
            raise ValueError('Recorded retrieval/query prediction missing or failed: '+query['id'])
        if saved['query']!=query['query'] or prediction.get('text_sha256')!=digest(query['query']):
            raise ValueError('Query differs from the recorded inference: '+query['id'])
        validate_entities(query['query'],prediction['entities'])
        if saved.get('query_entities')!=prediction['entities']:
            raise ValueError('Retrieval did not preserve its saved model query entities: '+query['id'])
        ranking=saved.get(approach)
        if not isinstance(ranking,list):raise ValueError('Missing recorded approach ranking: '+approach)
        metric=retrieval_metric(query,ranking,test_block_ids)
        split=query.get('split')
        if split not in ('test','development'):raise ValueError('Unknown retrieval split')
        prefix='test' if split=='test' else 'development'
        rows.append({
            'id':query['id'],'split':split,
            'inputs':{'query_id':query['id'],'query':query['query'],'split':split},
            'reference_outputs':{'known_relevant_block_ids':sorted(set(query['relevant_block_ids']).intersection(test_block_ids) if split=='test' else set(query['relevant_block_ids']).difference(test_block_ids)),'judgment_scope':query.get('judgment_scope'),'unjudged_policy':'unjudged_not_negative'},
            'outputs':{'approach':approach,'ranking':ranking,'query_entities':saved['query_entities'],'entity_boost_applied':saved.get('entity_boost_applied') if approach=='entity_assisted' else False,'recording_mode':'saved_retrieval_replay_no_search_or_inference'},
            'metric':metric,
            'feedback':{f'{prefix}_known_positive_recall_at_5':metric['known_positive_recall_at_5'],f'{prefix}_first_known_positive_reciprocal_rank_at_5':metric['first_known_positive_reciprocal_rank_at_5']},
            'feedback_comment':RETRIEVAL_BOUNDARY+' Test and development keys are separate; the experiment summary averages test queries only.',
        })
    return rows


def retrieval_test_summary(rows):
    test=[r for r in rows if r['split']=='test']
    if not test:raise ValueError('No held-out retrieval queries')
    return {'test_queries':len(test),'development_queries':len(rows)-len(test),'aggregation':'Mean over held-out test queries only. Development query feedback uses separate keys.','boundary':RETRIEVAL_BOUNDARY,'mean_test_known_positive_recall_at_5':sum(r['metric']['known_positive_recall_at_5'] for r in test)/len(test),'mean_test_first_known_positive_reciprocal_rank_at_5':sum(r['metric']['first_known_positive_reciprocal_rank_at_5'] for r in test)/len(test)}


def recovery_identity(directory,model,metadata,manifest,reference,status):
    """Read the separate repair identity without rewriting base-run provenance.

    The release importer performs the complete native-audit/prefix validation.
    Replay additionally binds the input artifacts it reads to this completed
    sidecar, so it cannot label changed retrieval as an unrepaired experiment.
    """
    path=directory/'recovery.json'
    if not path.exists():
        if status.get('recovery'):
            raise ValueError('Recovered status has no separate recovery identity')
        return None
    recovery=load_json(path);identity=recovery.get('identity',{})
    if (model['id']!='fastino/gliner2-base-v1' or recovery.get('schema_version')!=1
            or recovery.get('policy')!=RECOVERY_POLICY or recovery.get('status')!='complete'
            or not isinstance(identity,dict) or not identity
            or recovery.get('recovery_sha256')!=digest(identity)
            or recovery.get('remaining_failed_count')!=0):
        raise ValueError('Recovery identity is not a completed, self-consistent repair')
    expected={'model_id':model['id'],'revision':model['revision'],
              'base_configuration_sha256':metadata['configuration_sha256'],
              'base_code_sha256':metadata.get('code_sha256'),
              'corpus_snapshot_sha256':manifest['corpus_snapshot_sha256'],
              'corpus_content_sha256':manifest['corpus_content_sha256'],
              'reference_sha256':digest(reference)}
    if any(identity.get(key)!=value for key,value in expected.items()):
        raise ValueError('Recovery identity does not match pinned base inputs')
    if status.get('recovery',{}).get('recovery_sha256')!=recovery['recovery_sha256']:
        raise ValueError('Recovery status and sidecar identities differ')
    post_hashes=recovery.get('post_artifact_sha256',{})
    for name in ('metadata.json','reference-predictions.jsonl','query-predictions.jsonl','retrieval.json'):
        expected_hash=post_hashes.get(name)
        if not expected_hash or hashlib.sha256((directory/name).read_bytes()).hexdigest()!=expected_hash:
            raise ValueError('Replay artifact differs from completed recovery: '+name)
    return recovery['recovery_sha256']


def experiment_fingerprint(replay_fingerprint,model,experiment):
    """Keep legacy identities stable; separate every recorded repair producer."""
    payload={'replay':replay_fingerprint,'model_id':model['model_id'],'revision':model['revision'],'configuration_sha256':model['configuration_sha256'],'task':experiment['task'],'variant':experiment['variant'],'outputs':[(r['id'],r['outputs']) for r in experiment['rows']]}
    if model.get('recovery_sha256'):
        payload['recovery_sha256']=model['recovery_sha256']
    return digest(payload)


def build_plan(results,reference,tasks=('ner','retrieval')):
    manifest=load_json(results/'manifest.json')
    if manifest.get('corpus_snapshot_sha256')!=reference.get('corpus_snapshot_sha256'):
        raise ValueError('Manifest/reference corpus mismatch')
    fingerprint=digest({'protocol':PROTOCOL,'reference':reference,'corpus_content_sha256':manifest['corpus_content_sha256']})
    plan={'schema_version':1,'protocol':PROTOCOL,'fingerprint':fingerprint,'created_at':timestamp(),'recording_mode':'saved_outputs_only_no_inference','corpus_snapshot_sha256':manifest['corpus_snapshot_sha256'],'corpus_content_sha256':manifest['corpus_content_sha256'],'reference_sha256':digest(reference),'reference_status':reference.get('status'),'reference_review':reference.get('review'),'heldout_ner_passages':sum(p.get('split')=='test' for p in reference['passages']),'retrieval_queries':len(reference.get('retrieval_queries',[])),'heldout_retrieval_queries':sum(q.get('split')=='test' for q in reference.get('retrieval_queries',[])),'models':[],'pending_models':[]}
    for model in manifest['models']:
        directory=results/slug(model['id'])
        status=load_json(directory/'status.json') if (directory/'status.json').exists() else {}
        full_coverage=status.get('processed_blocks')==manifest.get('total_blocks') and status.get('processed_documents')==manifest.get('total_documents') and status.get('failed_blocks')==0
        if status.get('status')!='complete' or not full_coverage:
            plan['pending_models'].append({'model_id':model['id'],'reason':'Full inference model status is not complete'});continue
        metadata=load_json(directory/'metadata.json')
        if metadata.get('revision')!=model['revision'] or metadata.get('reference_sha256')!=digest(reference) or metadata.get('corpus_content_sha256')!=manifest['corpus_content_sha256']:
            raise ValueError('Saved model metadata does not match pinned inputs: '+model['id'])
        entry={'model_id':model['id'],'revision':model['revision'],'configuration_sha256':metadata['configuration_sha256'],'experiments':[]}
        recovery_sha=recovery_identity(directory,model,metadata,manifest,reference,status)
        if recovery_sha:entry['recovery_sha256']=recovery_sha
        if 'ner' in tasks:
            rows=heldout_ner_rows(reference,predictions_by_id(directory/'reference-predictions.jsonl'))
            entry['experiments'].append({'task':'ner','variant':'heldout','dataset_key':'ner-heldout','rows':rows,'summary':heldout_micro_summary(rows)})
        if 'retrieval' in tasks:
            saved_retrieval=load_json(directory/'retrieval.json')
            query_predictions=predictions_by_id(directory/'query-predictions.jsonl')
            for approach in ('lexical','entity_assisted'):
                rows=retrieval_rows(reference,saved_retrieval,query_predictions,approach)
                entry['experiments'].append({'task':'retrieval','variant':approach,'dataset_key':'retrieval-split-labelled','rows':rows,'summary':retrieval_test_summary(rows)})
        plan['models'].append(entry)
    plan['target_run_count']=sum(len(e['rows']) for m in plan['models'] for e in m['experiments'])
    plan['additional_evaluator_traces']=0
    return plan


def sanitized_error(error):
    # Error types are sufficient for the local checkpoint; SDK/request details
    # can contain headers. Never serialize exception text or environment values.
    return type(error).__name__


class ReplayUploader:
    def __init__(self,client,state_path,not_found):
        self.client,self.state_path,self.not_found=client,Path(state_path),not_found
        self.state=load_json(state_path) if Path(state_path).exists() else {'schema_version':1,'datasets':{},'experiments':{},'runs':{},'feedback':{},'updated_at':timestamp()}

    def save(self):
        self.state['updated_at']=timestamp();save_json(self.state_path,self.state)

    def dataset(self,key,fingerprint,rows,description):
        seed=key+':'+fingerprint
        if seed in self.state['datasets']:return self.state['datasets'][seed]
        name='workbench-'+key+'-'+fingerprint[:12]
        datasets=list(self.client.list_datasets(dataset_name=name))
        dataset=datasets[0] if datasets else self.client.create_dataset(dataset_name=name,description=description,metadata={'replay_protocol':PROTOCOL,'replay_fingerprint':fingerprint})
        output={'id':str(dataset.id),'name':name,'url':getattr(dataset,'url',None),'examples':{}}
        for row in rows:
            example_id=identity(seed,'example',row['id'])
            try:
                existing=self.client.read_example(example_id)
                if str(existing.dataset_id)!=output['id'] or existing.inputs!=row['inputs'] or existing.outputs!=row['reference_outputs']:
                    raise ValueError('Existing deterministic example differs from saved replay inputs')
            except self.not_found:
                self.client.create_example(inputs=row['inputs'],outputs=row['reference_outputs'],dataset_id=dataset.id,example_id=example_id,metadata={'split':row['split'],'replay_protocol':PROTOCOL},split=row['split'])
            output['examples'][row['id']]=example_id
        self.state['datasets'][seed]=output;self.save();return output

    def project(self,model,experiment,dataset,fingerprint):
        project_fingerprint=experiment_fingerprint(fingerprint,model,experiment)
        if project_fingerprint in self.state['experiments']:return self.state['experiments'][project_fingerprint]
        name=slug(model['model_id'])+'-'+experiment['task']+'-'+experiment['variant']+'-'+project_fingerprint[:10]
        recovery_metadata={'recovery_sha256':model['recovery_sha256']} if model.get('recovery_sha256') else {}
        try:
            project=self.client.read_project(project_name=name)
            if str(project.reference_dataset_id)!=dataset['id']:raise ValueError('Existing experiment points to a different dataset')
        except self.not_found:
            project=self.client.create_project(name,description='Saved-output replay; no inference or retrieval executed during upload. '+experiment['summary']['boundary'],reference_dataset_id=dataset['id'],num_examples=len(experiment['rows']),metadata={'replay_protocol':PROTOCOL,'replay_fingerprint':project_fingerprint,'model_id':model['model_id'],'checkpoint_revision':model['revision'],'configuration_sha256':model['configuration_sha256'],**recovery_metadata,'task':experiment['task'],'variant':experiment['variant'],'aggregation':experiment['summary']['aggregation'],'saved_output_replay':True})
        tenant_id=str(getattr(project,'tenant_id','') or '')
        url=f'https://smith.langchain.com/o/{tenant_id}/datasets/{dataset["id"]}/compare?selectedSessions={project.id}' if tenant_id else dataset.get('url')
        output={'id':str(project.id),'name':name,'dataset_id':dataset['id'],'dataset_name':dataset['name'],'fingerprint':project_fingerprint,**recovery_metadata,'tenant_id':tenant_id,'url':url,'task':experiment['task'],'variant':experiment['variant'],'rows':len(experiment['rows']),'summary':experiment['summary']}
        self.state['experiments'][project_fingerprint]=output;self.save();return output

    def feedback(self,project,run_id,key,score,comment):
        if score is None:return
        feedback_id=identity(project['fingerprint'],'feedback',run_id or 'summary',key)
        if feedback_id in self.state['feedback']:return
        try:self.client.read_feedback(feedback_id)
        except self.not_found:
            kwargs={'run_id':run_id,'session_id':project['id']} if run_id else {'project_id':project['id']}
            self.client.create_feedback(key=key,score=score,comment=comment,feedback_id=feedback_id,feedback_source_type='api',**kwargs)
        self.state['feedback'][feedback_id]={'key':key,'run_id':run_id,'project_id':project['id']};self.save()

    def row(self,project,dataset,row,model):
        run_id=identity(project['fingerprint'],'run',row['id'])
        if self.state['runs'].get(run_id,{}).get('complete'):return run_id
        try:
            run=self.client.read_run(run_id)
            if run.inputs!=row['inputs'] or run.outputs!=row['outputs']:
                raise ValueError('Existing deterministic replay run has different inputs or outputs')
        except self.not_found:
            instant=datetime.now(timezone.utc)
            # A supplied trace_id must have its matching hierarchy field, even
            # for this single root span. Use upload time, never model latency.
            dotted_order=instant.strftime('%Y%m%dT%H%M%S%fZ')+run_id
            recovery_metadata={'recovery_sha256':model['recovery_sha256']} if model.get('recovery_sha256') else {}
            self.client.create_run(name='Saved '+project['task']+' output replay',inputs=row['inputs'],outputs=row['outputs'],run_type='chain',project_name=project['name'],id=run_id,trace_id=run_id,dotted_order=dotted_order,reference_example_id=dataset['examples'][row['id']],start_time=instant,end_time=instant,extra={'metadata':{'saved_output_replay':True,'no_inference_performed':True,'split':row['split'],'model_id':model['model_id'],'checkpoint_revision':model['revision'],**recovery_metadata,'task':project['task'],'variant':project['variant'],'timing_boundary':'Replay timestamps are upload bookkeeping, not model latency. Original allocated inference timing remains in saved outputs where available.'}},tags=['saved-output-replay','split:'+row['split'],'task:'+project['task']])
        for key,value in row['feedback'].items():self.feedback(project,run_id,key,value,row['feedback_comment'])
        self.state['runs'][run_id]={'complete':True,'project_id':project['id'],'example_id':dataset['examples'][row['id']],'source_id':row['id']};self.save();return run_id


def comparison_links(experiments):
    from urllib.parse import quote
    links={}
    for task in ('ner','retrieval'):
        chosen=[e for e in experiments if e['task']==task]
        if chosen and chosen[0].get('tenant_id'):
            first=chosen[0]
            links[task]=f'https://smith.langchain.com/o/{first["tenant_id"]}/datasets/{first["dataset_id"]}/compare?selectedSessions='+quote(','.join(e['id'] for e in chosen),safe='')
    return links


def upload_plan(plan,uploader,status_path):
    status={'schema_version':1,'protocol':PROTOCOL,'status':'uploading','started_at':timestamp(),'fingerprint':plan['fingerprint'],'corpus_snapshot_sha256':plan['corpus_snapshot_sha256'],'corpus_content_sha256':plan['corpus_content_sha256'],'reference_sha256':plan['reference_sha256'],'no_inference_performed':True,'target_run_count':plan['target_run_count'],'additional_evaluator_traces':0,'pending_models':plan['pending_models'],'models':[]}
    save_json(status_path,status)
    completed=[]
    for model in plan['models']:
        result={'model_id':model['model_id'],'revision':model['revision'],'configuration_sha256':model['configuration_sha256'],'experiments':[]}
        if model.get('recovery_sha256'):result['recovery_sha256']=model['recovery_sha256']
        for experiment in model['experiments']:
            description=(NER_BOUNDARY+' Held-out test passages only; development excluded.' if experiment['task']=='ner' else RETRIEVAL_BOUNDARY+' All recorded query inputs are labelled development/test. Only test keys enter held-out experiment summaries.')
            dataset=uploader.dataset(experiment['dataset_key'],plan['fingerprint'],experiment['rows'],description)
            project=uploader.project(model,experiment,dataset,plan['fingerprint'])
            for row in experiment['rows']:uploader.row(project,dataset,row,model)
            if experiment['task']=='ner':
                for metric in ('precision','recall','f1'):
                    uploader.feedback(project,None,'heldout_entity_micro_'+metric,experiment['summary'][metric],NER_BOUNDARY+' Pooled entity counts, not average passage F1. Test passages: '+str(experiment['summary']['sample_passages']))
            else:
                for key in ('mean_test_known_positive_recall_at_5','mean_test_first_known_positive_reciprocal_rank_at_5'):
                    uploader.feedback(project,None,key,experiment['summary'][key],RETRIEVAL_BOUNDARY+' Test queries only; development excluded. Test count: '+str(experiment['summary']['test_queries']))
            result['experiments'].append(project);completed.append(project)
            save_json(status_path,{**status,'models':[*status['models'],result],'comparison_links':comparison_links(completed),'updated_at':timestamp()})
        status['models'].append(result)
    project_ids={project['id'] for project in completed}
    completed_runs=sum(run.get('complete') and run.get('project_id') in project_ids for run in uploader.state['runs'].values())
    status.update(status='complete' if not plan['pending_models'] else 'partial_pending_models',comparison_links=comparison_links(completed),finished_at=timestamp(),completed_replay_runs=completed_runs,reference_status=plan['reference_status'],reference_review=plan['reference_review'],heldout_ner_passages=plan['heldout_ner_passages'],retrieval_queries=plan['retrieval_queries'],heldout_retrieval_queries=plan['heldout_retrieval_queries'])
    save_json(status_path,status);return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',default='artifacts/local/ner');parser.add_argument('--reference',default='data/ner-reference-sample.json');parser.add_argument('--state-directory');parser.add_argument('--tasks',default='ner,retrieval');parser.add_argument('--upload',action='store_true',help='Write saved-output replay experiments to LangSmith. Omission performs local planning only.')
    args=parser.parse_args()
    results=Path(args.results).resolve();state_directory=Path(args.state_directory).resolve() if args.state_directory else results/'langsmith-replay'
    tasks=tuple(args.tasks.split(','))
    if not tasks or set(tasks)-{'ner','retrieval'}:parser.error('--tasks must be ner, retrieval or ner,retrieval')
    reference=load_json(args.reference)
    plan=build_plan(results,reference,tasks)
    save_json(state_directory/'plan.json',plan)
    print(json.dumps({'status':'plan_ready','eligible_models':len(plan['models']),'pending_models':len(plan['pending_models']),'heldout_ner_passages':plan['heldout_ner_passages'],'retrieval_queries':plan['retrieval_queries'],'heldout_retrieval_queries':plan['heldout_retrieval_queries'],'target_replay_runs':plan['target_run_count'],'no_inference_performed':True}),flush=True)
    if not args.upload:return 0
    if not plan['models']:raise ValueError('No completed model artifacts available for replay')
    if not os.environ.get('LANGSMITH_API_KEY'):raise RuntimeError('LANGSMITH_API_KEY must be provided through environment or Colab Secrets for observability. No model inference API is used.')
    from langsmith import Client
    from langsmith.utils import LangSmithNotFoundError
    uploader=ReplayUploader(Client(auto_batch_tracing=False),state_directory/'checkpoint.json',LangSmithNotFoundError)
    try:
        status=upload_plan(plan,uploader,state_directory/'status.json')
    except Exception as error:
        existing=load_json(state_directory/'status.json') if (state_directory/'status.json').exists() else {}
        save_json(state_directory/'status.json',{**existing,'status':'interrupted_or_failed','error_type':sanitized_error(error),'updated_at':timestamp(),'resume':'Rerun the same command. Deterministic run/feedback IDs and local checkpoint prevent duplicate records.'})
        print(json.dumps({'status':'interrupted_or_failed','error_type':sanitized_error(error),'no_inference_performed':True}),flush=True)
        return 2
    print(json.dumps({'status':status['status'],'completed_replay_runs':status['completed_replay_runs'],'comparison_links':status['comparison_links'],'no_inference_performed':True}),flush=True)
    return 0 if status['status']=='complete' else 2


if __name__=='__main__':
    try:sys.exit(main())
    except Exception as error:
        # No SDK/network exception string or stack trace can reveal credentials.
        print(json.dumps({'status':'failed_before_upload','error_type':sanitized_error(error),'no_inference_performed':True}),flush=True)
        sys.exit(2)
