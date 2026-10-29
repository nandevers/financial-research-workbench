import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

PATH=Path(__file__).resolve().parents[2]/'scripts/ner_recover_gliner2.py'
SPEC=importlib.util.spec_from_file_location('ner_recovery_for_test',PATH)
recovery=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(recovery)


def native(text,start,end):
    return {'organization':[{'text':text,'start':start,'end':end,'confidence':.9}]}


def test_proved_synthetic_terminal_period_projects_only_non_source_suffix():
    original='Federal Deposit Insurance Corp'
    entities,decisions=recovery.project_native_entities(original,native(original+'.',0,len(original)+1))
    assert entities==[{'start':0,'end':len(original),'text':original,'label':'ORGANIZATION','score':.9}]
    assert decisions[0]['action']=='project_synthetic_period'
    assert decisions[0]['native_text']==original+'.'
    assert decisions[0]['native_end']==len(original)+1
    assert decisions[0]['projected_text']==original


def test_inbound_spans_and_genuine_terminal_period_are_not_changed():
    original='Bank of America and Chubb'
    entities,decisions=recovery.project_native_entities(original,native('Bank of America',0,15))
    assert entities[0]['start']==0 and entities[0]['end']==15
    assert entities[0]['text']=='Bank of America'
    assert decisions[0]['action']=='unchanged_original_span'
    with_period='Chubb Corp.'
    entities,decisions=recovery.project_native_entities(with_period,native(with_period,0,len(with_period)))
    assert entities[0]['end']==len(with_period)
    assert decisions[0]['action']=='unchanged_original_span'


def test_synthetic_only_prediction_is_omitted_with_explicit_audit():
    for original in ('Acme',''):
        entities,decisions=recovery.project_native_entities(original,native('.',len(original),len(original)+1))
        assert entities==[]
        assert decisions[0]['action']=='omit_synthetic_only'
        assert decisions[0]['projected_start'] is None and decisions[0]['projected_end'] is None
        assert decisions[0]['projected_text']==''


@pytest.mark.parametrize('original,payload',[
    ('Acme',native('Acme?',0,5)),
    ('Acme',native('Acme.',0,6)),
    ('Acme.',native('Acme..',0,6)),
    ('Acme',native('Wrong',0,4)),
    ('Acme',native('Acme',1,4)),
    ('Acme',native('Acme.',-1,5)),
    ('Acme',native('Acme.',False,5)),
])
def test_unknown_surface_offset_or_non_synthetic_suffix_is_rejected(original,payload):
    with pytest.raises(ValueError):recovery.project_native_entities(original,payload)


def test_mixed_native_labels_keep_original_coordinates_and_validate_confidence():
    original='Chubb in New York'
    payload={**native('Chubb',0,5),'location':[{'text':'New York.','start':9,'end':18,'confidence':.8}]}
    entities,decisions=recovery.project_native_entities(original,payload)
    assert [(e['start'],e['end'],e['label']) for e in entities]==[(0,5,'ORGANIZATION'),(9,17,'LOCATION')]
    assert [d['action'] for d in decisions]==['unchanged_original_span','project_synthetic_period']
    payload['organization'][0]['confidence']=float('nan')
    with pytest.raises(ValueError,match='confidence'):recovery.project_native_entities(original,payload)


def test_frozen_prediction_prefix_and_metadata_remain_bound_to_recovery_identity(tmp_path):
    original=b'{"id":"failed","status":"failed"}\n'
    (tmp_path/'predictions.jsonl').write_bytes(original)
    (tmp_path/'metadata.json').write_text('{"configuration_sha256":"base"}\n',encoding='utf-8')
    identity={'original_predictions_bytes':len(original),'original_artifact_sha256':{'predictions.jsonl':hashlib.sha256(original).hexdigest(),'metadata.json':recovery.file_sha(tmp_path/'metadata.json')}}
    manifest={'identity':identity,'recovery_sha256':recovery.digest(identity)}
    recovery.append_record(tmp_path/'predictions.jsonl',{'id':'failed','status':'complete'})
    recovery.verify_prefix(tmp_path,manifest)
    assert (tmp_path/'predictions.jsonl').read_bytes().startswith(original)
    (tmp_path/'metadata.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='metadata'):recovery.verify_prefix(tmp_path,manifest)


def test_normalization_is_append_only_not_global_offset_shift():
    examples=['Bank','Bank.','Bank!','Bank?','Bank ','😀 Corp','']
    for original in examples:
        normalized=recovery.stock_normalized_text(original)
        assert normalized.startswith(original)
        assert normalized in (original,original+'.')
        for i,char in enumerate(original):assert normalized[i]==char


def test_diagnostic_reports_own_validation_reason_and_redacts_external_credentials():
    assert recovery.safe_error_message(ValueError('Frozen source fingerprint changed'))=='Frozen source fingerprint changed'
    assert 'secret request headers' not in recovery.safe_error_message(RuntimeError('secret request headers'))
    token='hf_'+'A'*40
    message=recovery.safe_error_message(ValueError('Bearer '+token+' api_key=privatevalue'))
    assert token not in message and 'privatevalue' not in message
    assert '[REDACTED]' in message
