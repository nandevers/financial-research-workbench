import json
from pathlib import Path

import pytest

from workbench.ner import (EntityRetrieval, append_jsonl, corpus_fingerprints, digest, enforce_resume, entity_index, evaluate_reference, export_bundle, known_positive_retrieval, latest_records, merge_entities, normalize_entities, read_jsonl, save_json, slug, span_counts, token_windows, validate_reference)


def character_tokenizer(text, **kwargs):
    return {'offset_mapping':[(i,i+1) for i,c in enumerate(text) if not c.isspace()]}


@pytest.fixture
def corpus():
    return {'documents':[{'id':'doc-a','ticker':'BAC','url':'https://example.org/a','sha256':'a'},{'id':'doc-b','ticker':'JPM','url':'https://example.org/b','sha256':'b'}], 'blocks':[
        {'id':'a1','document_id':'doc-a','text':'Bank of America manages liquidity in New York.','ticker':'BAC','fiscal_period':'2025','source_url':'https://example.org/a'},
        {'id':'b1','document_id':'doc-b','text':'JPMorgan Chase manages liquidity and capital.','ticker':'JPM','fiscal_period':'2025','source_url':'https://example.org/b'},
        {'id':'b2','document_id':'doc-b','text':'Bank of America also appears in this reference.','ticker':'JPM','fiscal_period':'2025','source_url':'https://example.org/b'},
    ]}


def entity(start,end,text,type_='ORGANIZATION',score=.9):
    return {'start':start,'end':end,'text':text,'type':type_,'score':score}


def test_all_characters_covered_with_unicode_and_long_blocks():
    text='  '+('Bank 😀 in New York; '*1500)+'  '
    windows=token_windows(text,character_tokenizer,max_tokens=64,overlap_tokens=12)
    covered=set()
    for window in windows:
        assert window['text']==text[window['start']:window['end']]
        assert window['tokens']<=64
        covered.update(range(window['start'],window['end']))
    assert covered==set(range(len(text)))
    assert token_windows('   ',character_tokenizer)[0]['text']=='   '
    with pytest.raises(ValueError):token_windows(text,character_tokenizer,max_tokens=10,overlap_tokens=10)


def test_offsets_type_normalization_window_duplicates_and_nuner_merge():
    normalized=normalize_entities('Bank 😀',[{'start':0,'end':4,'label':'ORG','score':.8}],offset=7)
    assert normalized==[entity(7,11,'Bank',score=.8)]
    assert normalize_entities('foo',[{'start':0,'end':3,'label':'MISC','score':.8}])==[]
    with pytest.raises(ValueError):normalize_entities('foo',[{'start':0,'end':8,'label':'PER'}])
    text='Bank of America'
    complete=entity(0,15,text);clipped=entity(8,15,'America',score=.95)
    assert merge_entities(text,[complete,complete,clipped])==[complete]
    words=[entity(0,4,'Bank'),entity(5,7,'of'),entity(8,15,'America')]
    assert merge_entities(text,words,merge_adjacent=True)==[complete]


def test_exact_resume_and_incomplete_tail_recovery(tmp_path):
    enforce_resume(tmp_path,{'configuration_sha256':'original'})
    enforce_resume(tmp_path,{'configuration_sha256':'original'})
    with pytest.raises(ValueError):enforce_resume(tmp_path,{'configuration_sha256':'different'})
    path=tmp_path/'predictions.jsonl'
    append_jsonl(path,[{'id':'a','status':'failed'},{'id':'a','status':'complete'}])
    with path.open('ab') as f:f.write(b'{"id":"unfinished')
    assert latest_records(path)=={'a':{'id':'a','status':'complete'}}
    assert path.read_bytes().endswith(b'\n')
    path.write_text('bad\n{}\n',encoding='utf-8')
    with pytest.raises(ValueError):read_jsonl(path,recover_tail=True)


def test_text_sensitive_fingerprint_detects_same_id_changed_text(corpus):
    old=corpus_fingerprints(corpus)
    corpus['blocks'][0]['text']+=' changed'
    changed=corpus_fingerprints(corpus)
    assert old['corpus_snapshot_sha256']==changed['corpus_snapshot_sha256']
    assert old['corpus_content_sha256']!=changed['corpus_content_sha256']


def test_entity_postings_include_all_successful_blocks_and_no_failed_predictions(corpus):
    rows={b['id']:{'id':b['id'],'status':'complete','entities':[entity(0,15,'Bank of America')]} for b in corpus['blocks'] if b['id']!='b1'}
    rows['b1']={'id':'b1','status':'failed','entities':[entity(0,8,'invented')]}
    index=entity_index(corpus,rows)
    assert len(index)==1
    assert index[0]['block_ids']==['a1','b2']
    assert index[0]['document_ids']==['doc-a','doc-b']
    assert index[0]['mention_count']==2
    assert index[0]['normalized']=='bank of america'


def test_actual_entity_postings_can_change_retrieval_without_faking_new_query_entities(corpus):
    entries=[{'normalized':'bank of america','block_ids':['a1','b2']}]
    retrieval=EntityRetrieval(corpus,entries)
    empty=retrieval.rank('liquidity',[])
    assert empty['lexical']==empty['entity_assisted']
    assert not empty['entity_boost_applied']
    boosted=retrieval.rank('capital',[entity(0,15,'Bank of America')],entity_weight=.9)
    assert boosted['lexical'][0]['block_id']=='b1'
    assert boosted['entity_assisted'][0]['block_id'] in ['a1','b2']
    assert boosted['entity_boost_applied']
    assert retrieval.rank('liquidity',[],allowed={'a1'})['lexical'][0]['block_id']=='a1'


def make_reference(corpus):
    return {'status':'agent_authored_pending_second_review','corpus_snapshot_sha256':corpus_fingerprints(corpus)['corpus_snapshot_sha256'],'passages':[{'id':'p1','document_id':'doc-a','source_block_id':'a1','source_text_start':0,'source_text_end':15,'text':'Bank of America','split':'test','entities':[{'start':0,'end':15,'text':'Bank of America','type':'ORGANIZATION','certainty':'clear'}]}]}


def test_reference_is_exact_source_and_scores_only_actual_predictions(corpus):
    reference=make_reference(corpus);validate_reference(corpus,reference)
    result=evaluate_reference(reference,{'p1':{'status':'complete','entities':[entity(0,15,'Bank of America')]}})
    assert result['test_passages']==1
    assert result['metrics']['f1']==1
    assert result['reference_status']=='agent_authored_pending_second_review'
    missing=evaluate_reference(reference,{})
    assert missing['metrics'] is None and missing['missing_predictions']==1
    reference['passages'][0]['text']='Not the original'
    with pytest.raises(ValueError):validate_reference(corpus,reference)


def test_uncertain_reference_is_excluded_and_unjudged_retrieval_not_negative():
    expected=[{'start':0,'end':4,'type':'ORG','certainty':'clear'},{'start':5,'end':10,'type':'LOC','certainty':'uncertain'}]
    assert span_counts(expected,[entity(0,4,'Bank'),entity(5,10,'Place','LOCATION')])=={'tp':1,'fp':0,'fn':0,'uncertain_reference_entities':1}
    metric=known_positive_retrieval({'relevant_block_ids':['a1']},[{'block_id':'unjudged'},{'block_id':'a1'}])
    assert metric['known_positive_recall_at_k']==1
    assert metric['first_known_positive_reciprocal_rank']==.5
    assert 'precision' not in metric


def test_public_export_has_complete_postings_but_bounded_text_and_explicit_missing_model(tmp_path,corpus):
    manifest={'models':[{'id':'local/test','name':'Test','license':'MIT','revision':'abc','profile':'test'},{'id':'local/missing','name':'Missing','license':'MIT','revision':'def','profile':'test'}]}
    directory=tmp_path/slug('local/test')
    save_json(directory/'status.json',{'status':'complete','processed_blocks':3})
    save_json(directory/'entity-index.json',[{'normalized':'bank of america','block_ids':['a1','b2'],'mentions':[{'block_id':'a1','start':0,'end':15}]}])
    bundle=export_bundle(corpus,manifest,tmp_path,tmp_path/'public.json',excerpt_limit=1,excerpt_chars=8)
    assert bundle['source_documents']==2 and bundle['source_blocks']==3
    assert bundle['complete_models']==1 and bundle['completion_status']=='partial_or_failed'
    assert bundle['models'][1]['status']['status']=='not_run'
    assert bundle['models'][0]['entities'][0]['block_ids']==['a1','b2']
    assert len(bundle['excerpts'])==1
    assert next(iter(bundle['excerpts'].values()))['text']=='Bank of '
    assert next(iter(bundle['excerpts'].values()))['truncated'] is True


def test_adaptive_batch_retries_keep_every_window_and_isolate_single_input_failures():
    import importlib.util
    from types import SimpleNamespace
    worker_path=Path(__file__).resolve().parents[2]/'scripts/ner_worker.py'
    spec=importlib.util.spec_from_file_location('ner_worker_for_test',worker_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class FakeTagger:
        torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False))
        def predict(self,texts):
            if len(texts)>2:raise RuntimeError('Simulated batch OOM')
            if 'bad input' in texts:raise ValueError('Simulated individual failure')
            return [[{'text':text}] for text in texts]
    results=module.adaptive_predict(FakeTagger(),['a','bad input','c','d'],batch_size=4)
    assert len(results)==4
    assert results[0]['entities'][0]['text']=='a'
    assert 'ValueError' in results[1]['error']
    assert results[2]['entities'][0]['text']=='c'
    assert results[3]['entities'][0]['text']=='d'


def test_colab_profile_uses_base_pip_without_ensurepip(tmp_path,monkeypatch):
    import importlib.util
    import sys
    script=Path(__file__).resolve().parents[2]/'scripts/ner_colab_run.py'
    spec=importlib.util.spec_from_file_location('ner_orchestrator_for_test',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    observed={}
    class FakeBuilder:
        def __init__(self,**kwargs):observed['builder']=kwargs
        def create(self,path):observed['directory']=path
    monkeypatch.setattr(module.venv,'EnvBuilder',FakeBuilder)
    monkeypatch.setattr(module.subprocess,'run',lambda command,**kwargs:observed.update(command=command,options=kwargs))
    target=module.prepare_profile('transformers',tmp_path)
    assert observed['builder']=={'with_pip':False,'system_site_packages':True}
    assert observed['command'][:6]==[sys.executable,'-m','pip','--python',str(target),'install']
    assert 'transformers==4.57.1' in observed['command']
    assert observed['options']['check'] is True


def test_completed_second_agent_review_is_described_without_human_gold_claim(corpus):
    reference=make_reference(corpus)
    reference.update(status='agent_reviewed_not_human_gold',review={'second_review_status':'completed','second_reviewer':'root agent'})
    result=evaluate_reference(reference,{'p1':{'status':'complete','entities':[entity(0,15,'Bank of America')]}})
    assert 'separately agent-reviewed' in result['boundary']
    assert 'pending' not in result['boundary']
    assert 'not human gold' in result['boundary']
    assert result['reference_review']['second_reviewer']=='root agent'
