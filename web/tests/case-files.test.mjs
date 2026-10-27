import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {allAttempts,attemptFor,casePair,currentRuns,featuredCaseIds,preferredModel,parseRoute,formatRoute,normalizeRoute} from '../src/lib/case-files.ts';
const data=JSON.parse(readFileSync(new URL('../public/data/workbench.json',import.meta.url),'utf8'));
const banking='liquidity-2024-2025-pair',spark='XHToken/Spark-X2.6-4B',mimo='XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B';

test('outcome river accounts for all twenty current-cohort attempts exactly once',()=>{
 const attempts=allAttempts(data);
 assert.equal(attempts.length,20);assert.equal(new Set(attempts.map(a=>a.id)).size,20);
 assert.equal(attempts.filter(a=>a.state==='recorded').length,14);
 assert.equal(attempts.filter(a=>a.state==='failed').length,6);
 assert.equal(featuredCaseIds(data).length,2);
 assert.ok(attempts.every(a=>a.run?.mode!=='reference'));
});
test('paired comparison requires matching source/workflow, model revision and precision',()=>{
 const pair=casePair(data,banking,spark);assert.equal(pair.compatible,true);
 assert.equal(pair.baseline.run.baseline,true);assert.equal(pair.workflow.run.baseline,false);
 const changed=structuredClone(data);const run=changed.runs.find(r=>r.id===pair.baseline.run.id);
 run.hardware.revision='different-revision';assert.equal(casePair(changed,banking,spark).compatible,false);
 run.hardware.corpus_snapshot_sha256='different-source';assert.equal(attemptFor(changed,banking,spark,'baseline').state,'not-recorded');
});
test('failed or unrecorded model attempts never fall back to extractive answers',()=>{
 const failed=attemptFor(data,banking,mimo,'workflow');assert.equal(failed.state,'failed');assert.equal(failed.run,undefined);
 const absent=attemptFor(data,'credit-2025-single',spark,'workflow');assert.equal(absent.state,'not-recorded');assert.equal(absent.run,undefined);
 assert.equal(currentRuns(data).length,14);
});
test('default checkpoint chooses a complete recorded pair and preserves evidence-first cases',()=>{
 assert.equal(preferredModel(data,banking),spark);
 assert.equal(preferredModel(data,'credit-2025-single'),'reference');
 assert.equal(normalizeRoute(parseRoute('#/research/credit-2025-single'),data).modelId,'reference');
});
test('old routes and exact-view routes restore checkpoint, approach and its own passage',()=>{
 const route={view:'research',caseId:banking,modelId:spark,tab:'compare',approach:'baseline',sourceId:'bac-source-123'};
 assert.deepEqual(parseRoute(formatRoute(route)),route);
 assert.equal(normalizeRoute(parseRoute('#/research/'+banking),data).caseId,banking);
 assert.equal(parseRoute('#/models').view,'models');
 assert.equal(normalizeRoute(parseRoute('#/research/missing?model=missing&tab=invalid'),data).modelId,spark);
});

test('the entry route explains the study and the entity section has its own stable address',()=>{
 assert.equal(parseRoute('').view,'overview');
 assert.equal(parseRoute('#/').view,'overview');
 for(const view of ['overview','entities']){
  const route={view,caseId:banking,modelId:spark,tab:'brief',approach:'workflow',sourceId:''};
  assert.deepEqual(parseRoute(formatRoute(route)),route);
 }
 assert.equal(parseRoute('#/research/'+banking).view,'research');
});
