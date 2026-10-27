import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {editorialSpans,findEditorialExample,draftNerProtocol} from '../src/lib/named-entities.ts';
const data=JSON.parse(readFileSync(new URL('../public/data/workbench.json',import.meta.url),'utf8'));
const registry=JSON.parse(readFileSync(new URL('../src/data/ner-models.json',import.meta.url),'utf8'));
test('NER registry contains five open specialists, common labels and no invented scores',()=>{
 assert.equal(registry.models.length,5);assert.equal(new Set(registry.models.map(m=>m.id)).size,5);
 assert.equal(registry.prediction_runs,0);assert.equal(registry.gold_passages,0);
 for(const model of registry.models){
  assert.ok(['MIT','Apache-2.0'].includes(model.license));assert.equal(model.language,'English');
  assert.deepEqual(model.common_labels,['PERSON','ORGANIZATION','LOCATION']);
  assert.equal(new URL(model.source).hostname,'huggingface.co');
  assert.ok(Object.values(model.results).every(v=>v===null));
 }
});
test('every editorial illustration points to real exported text and preserves exact non-overlapping spans',()=>{
 for(const example of registry.examples){
  assert.equal(example.editorial,true);const found=findEditorialExample(data,example);assert.ok(found);assert.ok(found.spans.length>0);
  for(const span of found.spans)assert.equal(found.evidence.text.slice(span.start,span.end),span.surface);
  for(let i=1;i<found.spans.length;i++)assert.ok(found.spans[i-1].end<=found.spans[i].start);
 }
 assert.equal(findEditorialExample(data,{...registry.examples[0],evidence_id:'missing'}),null);
});
test('editorial highlighting prefers full organization names over contained aliases and rejects absent strings',()=>{
 const spans=editorialSpans('Federal Reserve System (Federal Reserve)',[{surface:'Federal Reserve System',type:'ORGANIZATION',explanation:'Full name'},{surface:'Federal Reserve',type:'ORGANIZATION',explanation:'Alias'},{surface:'Invented Person',type:'PERSON',explanation:'Absent'}]);
 assert.deepEqual(spans.map(s=>s.surface),['Federal Reserve System','Federal Reserve']);
 assert.deepEqual(spans.map(s=>s.start),[0,24]);
 assert.equal(editorialSpans('text',[{surface:'',type:'PERSON',explanation:'Empty'}]).length,0);
});
test('draft benchmark stays unexecuted and separates unsupported extra categories',()=>{
 const draft=draftNerProtocol(data,registry.models);assert.equal(draft.status,'draft_not_executed');assert.equal(draft.prediction_runs,0);
 assert.equal(draft.corpus_snapshot_sha256,data.comparison_cohort.corpus_snapshot_sha256);
 assert.equal(draft.extended_track.model_ids.length,3);
 assert.ok(!draft.extended_track.model_ids.includes('dslim/bert-base-NER'));
 assert.equal(draft.extended_track.fixed_label_models,'not_applicable');
 assert.match(draft.reference_policy,/independently/i);assert.match(draft.split_policy,/held-out/);
});
