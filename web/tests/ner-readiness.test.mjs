import test from 'node:test';
import assert from 'node:assert/strict';
import {NER_EXPECTED,nerResultsProblem,nerRecoveryProblem,loadNerResults,langsmithResultLink,entityIndexProblem,predictedSegments} from '../src/lib/ner-results.ts';

// Synthetic structure fixtures exercise the publication gate; they are never exported as measurements.
const snapshot='a'.repeat(64),content='b'.repeat(64),org='11111111-1111-1111-1111-111111111111';
function fixture(){return {schema_version:1,completion_status:'complete',complete_models:5,source_documents:30,source_blocks:26058,corpus_snapshot_sha256:snapshot,corpus_content_sha256:content,documents:Array.from({length:30},(_,i)=>({id:`fixture-doc-${i}`})),excerpts:{},models:NER_EXPECTED.modelIds.map(id=>({id,status:{status:'complete',processed_blocks:26058,total_blocks:26058,processed_documents:30,total_documents:30,failed_blocks:0,reference_failures:0,query_failures:0},entities:[],retrieval:[],evaluation:{test_passages:32,missing_predictions:0,reference_status:'agent_reviewed_not_human_gold',metrics:{precision:0,recall:0,f1:0}}}))}}

test('measured view requires five distinct expected checkpoints, not a declared count alone',()=>{
 const result=fixture();assert.equal(nerResultsProblem(result,snapshot,result.documents.map(d=>d.id)),null);
 const duplicate=fixture();duplicate.models[4]=duplicate.models[0];assert.match(nerResultsProblem(duplicate),/unique/);
 const short=fixture();short.models.pop();assert.notEqual(nerResultsProblem(short),null);
 const unexpected=fixture();unexpected.models[4].id='unrecorded-model';assert.match(nerResultsProblem(unexpected),/expected/);
});
test('partial coverage, stale corpus, missing predictions and invalid scores never activate measured results',()=>{
 for(const mutate of [r=>{r.models[0].status.processed_blocks=26057},r=>{r.models[0].status.processed_documents=29},r=>{r.models[0].status.failed_blocks=1},r=>{r.models[0].status.reference_failures=1},r=>{r.models[0].status.query_failures=1},r=>{r.models[0].evaluation.test_passages=40},r=>{r.models[0].evaluation.missing_predictions=1},r=>{r.models[0].evaluation.metrics.f1=NaN},r=>{r.documents[0]=null}]){
  const result=fixture();mutate(result);assert.notEqual(nerResultsProblem(result),null);
 }
 assert.match(nerResultsProblem(fixture(),'c'.repeat(64)),/collection/);
 assert.match(nerResultsProblem(fixture(),snapshot,['other']),/documents/);
});
test('a missing build artifact is distinct from a network error, malformed JSON or incomplete result',async()=>{
 const missing=await loadNerResults('/fixture',snapshot,undefined,async()=>new Response('',{status:404}));assert.equal(missing.status,'missing');assert.equal(missing.results,null);
 const server=await loadNerResults('/fixture',snapshot,undefined,async()=>new Response('',{status:503}));assert.equal(server.status,'error');assert.match(server.message,/503/);
 const malformed=await loadNerResults('/fixture',snapshot,undefined,async()=>new Response('<html>not results</html>',{status:200}));assert.equal(malformed.status,'error');
 const partial=fixture();partial.models[0].status.processed_blocks=100;
 assert.equal((await loadNerResults('/fixture',snapshot,undefined,async()=>Response.json(partial))).status,'error');
 assert.equal((await loadNerResults('/fixture',snapshot,undefined,async()=>Response.json(fixture()))).status,'ready');
});
test('entity index provenance and prediction structure are checked before search',()=>{
 const modelId=NER_EXPECTED.modelIds[0],index={schema_version:1,model_id:modelId,corpus_content_sha256:content,entities:[]};
 assert.equal(entityIndexProblem(index,modelId,content),null);
 assert.notEqual(entityIndexProblem({...index,model_id:'other'},modelId,content),null);
 assert.notEqual(entityIndexProblem({...index,entities:null},modelId,content),null);
 assert.notEqual(entityIndexProblem({...index,entities:[{text:'name',normalized:'name',types:['PERSON'],mentions:[{block_id:'fixture',document_id:'fixture',start:0,end:4,score:'bad'}]}]},modelId,content),null);
});
test('native LangSmith links identify exported records and never invent an experiment session',()=>{
 assert.equal(langsmithResultLink(null,org),null);
 const recorded={dataset_id:'22222222-2222-2222-2222-222222222222',experiment_name:'fixture-experiment'};
 const link=langsmithResultLink(recorded,org);assert.match(link.label,/dataset/);assert.ok(link.url.endsWith(recorded.dataset_id));assert.ok(!link.url.includes(recorded.experiment_name));
 assert.equal(langsmithResultLink({experiment_url:'javascript:alert(1)'},org),null);
 assert.equal(langsmithResultLink({experiment_url:'https://example.com/not-langsmith'},org),null);
 assert.equal(langsmithResultLink({experiment_url:'https://secret@smith.langchain.com/example'},org),null);
 assert.equal(langsmithResultLink({experiment_url:'https://smith.langchain.com/example'},org).label,'Open this experiment in LangSmith');
});
test('displayed source highlights preserve Python Unicode offsets and reject truncated spans',()=>{
 const text='😀 New York Bank',mention={start:2,end:10,type:'LOCATION',score:.9};
 const pieces=predictedSegments(text,[mention,{...mention,start:2,end:30}]);
 assert.equal(pieces.map(p=>p.text).join(''),text);
 assert.deepEqual(pieces.filter(p=>p.mention).map(p=>p.text),['New York']);
});
test('targeted repair disclosure requires complete passage accounting and matching replay identity',()=>{
 const modelId=NER_EXPECTED.modelIds[4],revision='fixture-revision',hash='c'.repeat(64);
 const ids=Array.from({length:20},(_,i)=>`fixture-repair-${i}`);
 const recovery={schema_version:1,policy:'gliner2-2.1.0-synthetic-terminal-period-v1',status:'complete',recovery_sha256:hash,identity:{model_id:modelId,revision,base_configuration_sha256:'d'.repeat(64),corpus_snapshot_sha256:snapshot,corpus_content_sha256:content,failed_block_ids:ids},append_records:ids.map(block_id=>({block_id,parent_prediction_sha256:hash,appended_prediction_sha256:hash,native_audit_record_sha256:hash})),original_success_count:26038,remaining_failed_count:0};
 const problem=value=>nerRecoveryProblem(value,modelId,revision,26058,content,snapshot);
 assert.equal(problem(recovery),null);
 assert.notEqual(problem({...recovery,status:'recovering'}),null);
 assert.notEqual(problem({...recovery,remaining_failed_count:1}),null);
 assert.notEqual(problem({...recovery,original_success_count:26039}),null);
 assert.notEqual(problem({...recovery,identity:{...recovery.identity,corpus_content_sha256:'other'}}),null);
 assert.notEqual(problem({...recovery,append_records:recovery.append_records.slice(1)}),null);
 const result=fixture();Object.assign(result.models[4],{revision,recovery});
 assert.equal(nerResultsProblem(result),null);
 result.models[4].langsmith={replay:{recovery_sha256:'e'.repeat(64),experiments:[]}};
 assert.match(nerResultsProblem(result),/replay experiment/);
 result.models[4].langsmith.replay.recovery_sha256=hash;
 assert.equal(nerResultsProblem(result),null);
});
