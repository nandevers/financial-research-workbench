import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH=Path(__file__).resolve().parents[2]/'scripts/ner_langsmith_replay.py'
SPEC=importlib.util.spec_from_file_location('ner_langsmith_replay_for_test',PATH)
replay=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def entity(text,start=0):
    return {'text':text,'start':start,'end':start+len(text),'type':'ORGANIZATION','score':.9}


@pytest.fixture
def reference():
    return {'status':'agent_reviewed_not_human_gold','review':{'second_review_status':'completed'},'corpus_snapshot_sha256':'snapshot','passages':[
        {'id':'dev','document_id':'d1','source_block_id':'b-dev','split':'development','text':'Bank','entities':[{**entity('Bank'),'certainty':'clear'}]},
        {'id':'test1','document_id':'d2','source_block_id':'b-test1','split':'test','text':'Bank of America','entities':[{**entity('Bank of America'),'certainty':'clear'}]},
        {'id':'test2','document_id':'d2','source_block_id':'b-test2','split':'test','text':'Chubb','entities':[{**entity('Chubb'),'certainty':'clear'}]},
    ],'retrieval_queries':[
        {'id':'q-test','split':'test','query':'Bank','relevant_block_ids':['b-dev','b-test1'],'judgment_scope':'reference_sample_blocks_only'},
        {'id':'q-dev','split':'development','query':'Chubb','relevant_block_ids':['b-dev'],'judgment_scope':'reference_sample_blocks_only'},
    ]}


def saved_predictions(reference):
    return {p['id']:{'id':p['id'],'status':'complete','text_sha256':replay.digest(p['text']),'entities':[entity(p['text'])],'recorded_at':'2026-10-28T00:00:00Z'} for p in reference['passages']}


def saved_retrieval(reference):
    predictions={q['id']:{'id':q['id'],'status':'complete','text_sha256':replay.digest(q['query']),'entities':[entity(q['query'])]} for q in reference['retrieval_queries']}
    rows=[{'id':q['id'],'query':q['query'],'status':'complete','query_entities':predictions[q['id']]['entities'],'lexical':[{'block_id':'unjudged','document_id':'d0'},{'block_id':'b-test1','document_id':'d2'}],'entity_assisted':[{'block_id':'b-test1','document_id':'d2'}],'entity_boost_applied':True} for q in reference['retrieval_queries']]
    return rows,predictions


def test_only_heldout_rows_uploaded_and_micro_is_not_mean_passage_f1(reference):
    predicted=saved_predictions(reference)
    predicted['test1']['entities']=[]
    rows=replay.heldout_ner_rows(reference,predicted)
    assert [row['id'] for row in rows]==['test1','test2']
    assert all(row['split']=='test' for row in rows)
    micro=replay.heldout_micro_summary(rows)
    assert micro['tp']==1 and micro['fn']==1 and micro['fp']==0
    assert micro['f1']==pytest.approx(2/3)
    assert micro['f1']!=sum(row['feedback']['passage_exact_f1'] for row in rows)/len(rows)
    assert 'not human gold' in micro['boundary'].lower()
    assert 'local_inference_seconds_allocated' in rows[0]['outputs']


def test_failed_or_different_saved_input_is_not_replayed(reference):
    predicted=saved_predictions(reference);predicted['test1']['status']='failed'
    with pytest.raises(ValueError,match='missing or failed'):replay.heldout_ner_rows(reference,predicted)
    predicted=saved_predictions(reference);predicted['test1']['text_sha256']='different'
    with pytest.raises(ValueError,match='fingerprint'):replay.heldout_ner_rows(reference,predicted)
    predicted=saved_predictions(reference);predicted['test1']['entities'][0]['end']=100
    with pytest.raises(ValueError,match='exact input'):replay.heldout_ner_rows(reference,predicted)


def test_retrieval_fold_positive_scope_and_unjudged_policy(reference):
    rows,predictions=saved_retrieval(reference)
    lexical=replay.retrieval_rows(reference,rows,predictions,'lexical')
    assert lexical[0]['reference_outputs']['known_relevant_block_ids']==['b-test1']
    assert lexical[1]['reference_outputs']['known_relevant_block_ids']==['b-dev']
    assert lexical[0]['metric']['known_positive_recall_at_5']==1
    assert lexical[0]['metric']['first_known_positive_reciprocal_rank_at_5']==.5
    assert all('precision' not in key for key in lexical[0]['metric'])
    assert 'test_known_positive_recall_at_5' in lexical[0]['feedback']
    assert 'development_known_positive_recall_at_5' in lexical[1]['feedback']
    summary=replay.retrieval_test_summary(lexical)
    assert summary['test_queries']==1 and summary['development_queries']==1
    assert summary['mean_test_known_positive_recall_at_5']==1
    rows[0]['query_entities']=[]
    with pytest.raises(ValueError,match='query entities'):replay.retrieval_rows(reference,rows,predictions,'entity_assisted')


class NotFound(Exception):pass


class FakeClient:
    def __init__(self):
        self.datasets={};self.examples={};self.projects={};self.runs={};self.feedback={}
        self.created_runs=0;self.created_feedback=0;self.fail_feedback_once=False
    def list_datasets(self,dataset_name):return [d for d in self.datasets.values() if d.name==dataset_name]
    def create_dataset(self,dataset_name,**kwargs):
        obj=SimpleNamespace(id=replay.identity('dataset',dataset_name),name=dataset_name,url='https://smith.langchain.com/datasets/local');self.datasets[obj.id]=obj;return obj
    def read_example(self,id):
        if id not in self.examples:raise NotFound()
        return self.examples[id]
    def create_example(self,**kwargs):self.examples[kwargs['example_id']]=SimpleNamespace(**kwargs)
    def read_project(self,project_name):
        if project_name not in self.projects:raise NotFound()
        return self.projects[project_name]
    def create_project(self,name,**kwargs):
        obj=SimpleNamespace(id=replay.identity('project',name),name=name,tenant_id='tenant',**kwargs);self.projects[name]=obj;return obj
    def read_run(self,id):
        if id not in self.runs:raise NotFound()
        return self.runs[id]
    def create_run(self,**kwargs):
        assert kwargs['id'] not in self.runs
        assert kwargs['extra']['metadata']['no_inference_performed'] is True
        # Like native ingestion, reject an incomplete or inconsistent root
        # hierarchy instead of accepting fields the server will refuse.
        assert kwargs['trace_id']==kwargs['id']
        assert kwargs['dotted_order']==kwargs['start_time'].strftime('%Y%m%dT%H%M%S%fZ')+kwargs['id']
        self.runs[kwargs['id']]=SimpleNamespace(**kwargs);self.created_runs+=1
    def read_feedback(self,id):
        if id not in self.feedback:raise NotFound()
        return self.feedback[id]
    def create_feedback(self,**kwargs):
        if self.fail_feedback_once:self.fail_feedback_once=False;raise ConnectionError('Do not persist request details')
        assert kwargs['feedback_id'] not in self.feedback
        self.feedback[kwargs['feedback_id']]=kwargs;self.created_feedback+=1


def test_upload_resume_uses_deterministic_ids_and_does_not_duplicate_interrupted_run(tmp_path,reference):
    client=FakeClient();uploader=replay.ReplayUploader(client,tmp_path/'checkpoint.json',NotFound)
    rows=replay.heldout_ner_rows(reference,saved_predictions(reference))
    model={'model_id':'local/test','revision':'abc','configuration_sha256':'config'}
    experiment={'task':'ner','variant':'heldout','rows':rows,'summary':replay.heldout_micro_summary(rows)}
    dataset=uploader.dataset('ner-heldout','fingerprint',rows,'Agent-reviewed, not human gold')
    project=uploader.project(model,experiment,dataset,'fingerprint')
    client.fail_feedback_once=True
    with pytest.raises(ConnectionError):uploader.row(project,dataset,rows[0],model)
    assert client.created_runs==1
    resumed=replay.ReplayUploader(client,tmp_path/'checkpoint.json',NotFound)
    resumed.row(project,dataset,rows[0],model)
    resumed.row(project,dataset,rows[0],model)
    assert client.created_runs==1 and client.created_feedback==3
    assert len(resumed.state['runs'])==1
    # Also recover from a missing local checkpoint using the same server IDs.
    fresh=replay.ReplayUploader(client,tmp_path/'new-checkpoint.json',NotFound)
    fresh.row(project,dataset,rows[0],model)
    assert client.created_runs==1 and client.created_feedback==3


def test_replay_root_hierarchy_uses_existing_id_and_upload_instant_without_changing_saved_outputs(tmp_path,reference):
    client=FakeClient();uploader=replay.ReplayUploader(client,tmp_path/'checkpoint.json',NotFound)
    rows=replay.heldout_ner_rows(reference,saved_predictions(reference))
    model={'model_id':'local/test','revision':'abc','configuration_sha256':'config'}
    experiment={'task':'ner','variant':'heldout','rows':rows,'summary':replay.heldout_micro_summary(rows)}
    dataset=uploader.dataset('ner-heldout','fingerprint',rows,'Agent-reviewed sample')
    project=uploader.project(model,experiment,dataset,'fingerprint')
    expected_id=replay.identity(project['fingerprint'],'run',rows[0]['id'])
    run_id=uploader.row(project,dataset,rows[0],model)
    run=client.runs[run_id]
    assert run_id==expected_id==run.id==run.trace_id
    assert run.dotted_order==run.start_time.strftime('%Y%m%dT%H%M%S%fZ')+expected_id
    assert run.start_time==run.end_time and run.start_time.tzinfo is not None
    assert run.inputs==rows[0]['inputs'] and run.outputs==rows[0]['outputs']
    assert run.outputs['recording_mode']=='saved_prediction_replay_no_inference'
    fresh=replay.ReplayUploader(client,tmp_path/'new-checkpoint.json',NotFound)
    assert fresh.row(project,dataset,rows[0],model)==expected_id
    assert client.created_runs==1 and client.runs[run_id] is run


def test_build_plan_never_treats_partial_corpus_as_upload_ready(tmp_path,reference):
    replay.save_json(tmp_path/'manifest.json',{'corpus_snapshot_sha256':'snapshot','corpus_content_sha256':'content','total_blocks':3,'total_documents':2,'models':[{'id':'local/test','revision':'abc'}]})
    directory=tmp_path/'local--test'
    replay.save_json(directory/'status.json',{'status':'complete','processed_blocks':2,'processed_documents':2,'failed_blocks':0})
    plan=replay.build_plan(tmp_path,reference)
    assert plan['target_run_count']==0 and len(plan['pending_models'])==1
    assert plan['additional_evaluator_traces']==0


def test_read_only_prediction_loader_refuses_truncated_tail(tmp_path):
    path=tmp_path/'predictions.jsonl';raw=b'{"id":"good"}\n{"id":"unfinished'
    path.write_bytes(raw)
    with pytest.raises(ValueError):replay.predictions_by_id(path)
    assert path.read_bytes()==raw


def test_error_reporting_never_serializes_exception_request_or_credentials():
    error=RuntimeError('Bearer not-a-real-secret')
    assert replay.sanitized_error(error)=='RuntimeError'


def test_complete_saved_plan_uploads_three_native_experiments_without_repeating_records(tmp_path,reference):
    import json
    manifest={'corpus_snapshot_sha256':'snapshot','corpus_content_sha256':'content','total_blocks':3,'total_documents':2,'models':[{'id':'local/test','revision':'abc'}]}
    replay.save_json(tmp_path/'manifest.json',manifest)
    directory=tmp_path/'local--test'
    replay.save_json(directory/'status.json',{'status':'complete','processed_blocks':3,'processed_documents':2,'failed_blocks':0})
    replay.save_json(directory/'metadata.json',{'revision':'abc','reference_sha256':replay.digest(reference),'corpus_content_sha256':'content','configuration_sha256':'configuration'})
    reference_predictions=saved_predictions(reference)
    rows,query_predictions=saved_retrieval(reference)
    (directory/'reference-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in reference_predictions.values()),encoding='utf-8')
    (directory/'query-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in query_predictions.values()),encoding='utf-8')
    replay.save_json(directory/'retrieval.json',rows)
    plan=replay.build_plan(tmp_path,reference)
    assert plan['target_run_count']==6 and plan['additional_evaluator_traces']==0
    assert plan['heldout_ner_passages']==2 and plan['heldout_retrieval_queries']==1
    client=FakeClient();uploader=replay.ReplayUploader(client,tmp_path/'replay-checkpoint.json',NotFound)
    status=replay.upload_plan(plan,uploader,tmp_path/'replay-status.json')
    assert status['status']=='complete'
    assert status['corpus_snapshot_sha256']=='snapshot'
    assert status['corpus_content_sha256']=='content'
    assert status['reference_sha256']==replay.digest(reference)
    assert status['models'][0]['configuration_sha256']=='configuration'
    assert status['completed_replay_runs']==6 and client.created_runs==6
    assert len(client.projects)==3 and len(client.datasets)==2
    assert set(status['comparison_links'])=={'ner','retrieval'}
    feedback_count=client.created_feedback
    resumed=replay.ReplayUploader(client,tmp_path/'replay-checkpoint.json',NotFound)
    again=replay.upload_plan(plan,resumed,tmp_path/'replay-status.json')
    assert again['completed_replay_runs']==6 and client.created_runs==6
    assert client.created_feedback==feedback_count
    assert any(f['key']=='heldout_entity_micro_f1' and 'project_id' in f for f in client.feedback.values())


def test_recovery_fingerprint_separates_same_outputs_without_changing_unrepaired_identity(reference):
    rows=replay.heldout_ner_rows(reference,saved_predictions(reference))
    model={'model_id':'fastino/gliner2-base-v1','revision':'abc','configuration_sha256':'config'}
    experiment={'task':'ner','variant':'heldout','rows':rows}
    original=replay.digest({'replay':'fingerprint','model_id':model['model_id'],'revision':'abc','configuration_sha256':'config','task':'ner','variant':'heldout','outputs':[(r['id'],r['outputs']) for r in rows]})
    assert replay.experiment_fingerprint('fingerprint',model,experiment)==original
    assert replay.experiment_fingerprint('fingerprint',{**model,'recovery_sha256':None},experiment)==original
    recovered=replay.experiment_fingerprint('fingerprint',{**model,'recovery_sha256':'a'*64},experiment)
    assert recovered!=original
    assert recovered!=replay.experiment_fingerprint('fingerprint',{**model,'recovery_sha256':'b'*64},experiment)


def test_recovery_is_native_project_and_run_provenance_and_cannot_reuse_legacy_session(tmp_path,reference):
    rows=replay.heldout_ner_rows(reference,saved_predictions(reference))
    model={'model_id':'fastino/gliner2-base-v1','revision':'abc','configuration_sha256':'config'}
    recovered={**model,'recovery_sha256':'a'*64}
    experiment={'task':'ner','variant':'heldout','rows':rows,'summary':replay.heldout_micro_summary(rows)}
    client=FakeClient();uploader=replay.ReplayUploader(client,tmp_path/'checkpoint.json',NotFound)
    dataset=uploader.dataset('ner-heldout','fingerprint',rows,'Agent-reviewed sample')
    original=uploader.project(model,experiment,dataset,'fingerprint')
    repaired=uploader.project(recovered,experiment,dataset,'fingerprint')
    assert repaired['id']!=original['id']
    assert 'recovery_sha256' not in original
    assert repaired['recovery_sha256']=='a'*64
    assert client.projects[repaired['name']].metadata['recovery_sha256']=='a'*64
    original_run=uploader.row(original,dataset,rows[0],model)
    repaired_run=uploader.row(repaired,dataset,rows[0],recovered)
    assert original_run!=repaired_run
    assert client.runs[repaired_run].extra['metadata']['recovery_sha256']=='a'*64


def test_build_plan_carries_only_complete_matching_recovery_identity(tmp_path,reference):
    import hashlib
    import json
    model={'id':'fastino/gliner2-base-v1','revision':'abc'}
    manifest={'corpus_snapshot_sha256':'snapshot','corpus_content_sha256':'content','total_blocks':3,'total_documents':2,'models':[model]}
    replay.save_json(tmp_path/'manifest.json',manifest)
    directory=tmp_path/replay.slug(model['id'])
    metadata={'revision':'abc','reference_sha256':replay.digest(reference),'corpus_content_sha256':'content','configuration_sha256':'configuration','code_sha256':'base-code'}
    replay.save_json(directory/'metadata.json',metadata)
    rows,query_predictions=saved_retrieval(reference)
    for name,predictions in [('reference-predictions.jsonl',saved_predictions(reference)),('query-predictions.jsonl',query_predictions)]:
        (directory/name).write_text(''.join(json.dumps(r)+'\n' for r in predictions.values()),encoding='utf-8')
    replay.save_json(directory/'retrieval.json',rows)
    identity={'model_id':model['id'],'revision':'abc','base_configuration_sha256':'configuration','base_code_sha256':'base-code','corpus_snapshot_sha256':'snapshot','corpus_content_sha256':'content','reference_sha256':replay.digest(reference)}
    recovery={'schema_version':1,'policy':replay.RECOVERY_POLICY,'status':'complete','identity':identity,'recovery_sha256':replay.digest(identity),'remaining_failed_count':0,'post_artifact_sha256':{name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in ('metadata.json','reference-predictions.jsonl','query-predictions.jsonl','retrieval.json')}}
    status={'status':'complete','processed_blocks':3,'processed_documents':2,'failed_blocks':0,'recovery':{'recovery_sha256':recovery['recovery_sha256']}}
    replay.save_json(directory/'status.json',status);replay.save_json(directory/'recovery.json',recovery)
    plan=replay.build_plan(tmp_path,reference)
    assert plan['models'][0]['recovery_sha256']==recovery['recovery_sha256']
    client=FakeClient();uploader=replay.ReplayUploader(client,tmp_path/'checkpoint.json',NotFound)
    uploaded=replay.upload_plan(plan,uploader,tmp_path/'uploaded.json')
    assert uploaded['models'][0]['recovery_sha256']==recovery['recovery_sha256']
    assert all(e['recovery_sha256']==recovery['recovery_sha256'] for e in uploaded['models'][0]['experiments'])
    for change in ({'status':'recovering'},{'recovery_sha256':'different'},{'remaining_failed_count':1}):
        replay.save_json(directory/'recovery.json',{**recovery,**change})
        with pytest.raises(ValueError,match='completed, self-consistent'):replay.build_plan(tmp_path,reference)
    replay.save_json(directory/'recovery.json',recovery)
    (directory/'retrieval.json').write_text('[]',encoding='utf-8')
    with pytest.raises(ValueError,match='Replay artifact differs'):replay.build_plan(tmp_path,reference)
    (directory/'recovery.json').unlink()
    with pytest.raises(ValueError,match='no separate recovery'):replay.build_plan(tmp_path,reference)
