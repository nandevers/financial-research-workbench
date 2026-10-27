import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {allAttempts,attemptFor,parseRoute,formatRoute} from '../src/lib/case-files.ts';
import {MODEL_IDS,experimentURL,metricValue,pairedDelta,promptPreview} from '../src/lib/evaluation.ts';
const data=JSON.parse(readFileSync(new URL('../public/data/workbench.json',import.meta.url),'utf8'));
const provenance=JSON.parse(readFileSync(new URL('../src/data/evaluation-provenance.json',import.meta.url),'utf8'));
const banking='liquidity-2024-2025-pair',spark=MODEL_IDS[2];
test('five arms map uniquely to five verified sessions, not five approach variants',()=>{
 assert.equal(MODEL_IDS.length,5);assert.equal(new Set(MODEL_IDS).size,5);
 assert.deepEqual(provenance.experiments.map(e=>e.model_id),MODEL_IDS);
 const url=new URL(experimentURL(provenance));assert.equal(url.searchParams.get('selectedSessions').split(',').length,5);
 assert.equal(new URL(experimentURL(provenance,spark)).searchParams.get('selectedSessions'),provenance.experiments[2].session_id);
 assert.equal(allAttempts(data).length,20);
});
test('failed generation telemetry is missing, never an invented zero; real zero is retained',()=>{
 const failed=attemptFor(data,banking,MODEL_IDS[1],'workflow');
 assert.equal(failed.state,'failed');assert.equal(metricValue(failed,'generation_seconds'),null);
 const good=structuredClone(attemptFor(data,banking,spark,'workflow'));
 good.run.generation.generation_seconds=0;assert.equal(metricValue(good,'generation_seconds'),0);
 good.run.generation.generation_seconds=NaN;assert.equal(metricValue(good,'generation_seconds'),null);
});
test('paired deltas require matching case, checkpoint, approach contrast and provenance',()=>{
 const a=attemptFor(data,banking,spark,'baseline'),b=attemptFor(data,banking,spark,'workflow');
 assert.equal(pairedDelta(a,b,'generation_seconds'),b.run.generation.generation_seconds-a.run.generation.generation_seconds);
 for(const field of ['revision','quantization','corpus_snapshot_sha256','workflow_sha256']){
  const corrupted=structuredClone(b);corrupted.run.hardware[field]='other';assert.equal(pairedDelta(a,corrupted,'generation_seconds'),null);
 }
 assert.equal(pairedDelta(a,a,'generation_seconds'),null);
 assert.equal(pairedDelta(a,attemptFor(data,banking,MODEL_IDS[4],'workflow'),'generation_seconds'),null);
});
test('prompt reconstruction uses exported question and bounded evidence, and refuses stale source versions',()=>{
 const run=attemptFor(data,banking,spark,'workflow').run;
 const preview=promptPreview(data,run,provenance);assert.ok(preview.includes(data.cases.find(c=>c.id===banking).query));
 assert.ok(preview.includes(run.evidence[0].id));assert.ok(!preview.includes('{{evidence}}'));
 assert.equal(promptPreview(data,run,{...provenance,workflow_sha256:'other'}),null);
 assert.equal(createHash('sha256').update(provenance.prompt_template).digest('hex'),provenance.prompt_sha256);
 const probe=structuredClone(run);probe.evidence=[{...probe.evidence[0],text:'😀'.repeat(1260)}];
 const reconstructed=promptPreview(data,probe,provenance);assert.equal((reconstructed.match(/😀/gu)||[]).length,1250);
});
test('source code snapshot matches the actual recorded workflow with Colab line endings',()=>{
 const source=['research.py','retrieval.py','contracts.py'].map(n=>readFileSync(new URL('../../engine/src/workbench/'+n,import.meta.url),'utf8').replaceAll('\r\n','\n')).join('');
 assert.equal(createHash('sha256').update(source).digest('hex'),provenance.workflow_sha256);
 assert.equal(provenance.native_spans_exported,false);assert.equal(provenance.prompt_hub_history_exported,false);
});
test('evaluation navigation preserves selected case and approach',()=>{
 const r={view:'evaluation',caseId:banking,modelId:spark,tab:'brief',approach:'baseline',sourceId:''};
 assert.deepEqual(parseRoute(formatRoute(r)),r);
 const embedded={...r,view:'research',tab:'evaluation'};assert.deepEqual(parseRoute(formatRoute(embedded)),embedded);
});
