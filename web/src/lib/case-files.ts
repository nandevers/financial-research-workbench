import type {Data, Run, FailureSummary} from '../types';

export type Approach = 'workflow' | 'baseline';
export type CaseTab = 'brief' | 'compare' | 'evidence' | 'execution' | 'outcomes' | 'evaluation';
export type Route = {view:'research'|'models'|'corpus'|'method'|'evaluation'|'overview'|'entities';caseId:string;modelId:string;tab:CaseTab;approach:Approach;sourceId:string};
export type Attempt = {id:string;caseId:string;modelId:string;approach:Approach;state:'recorded'|'failed'|'not-recorded';run?:Run;failure?:FailureSummary};
const tabs:CaseTab[]=['brief','compare','evidence','execution','outcomes','evaluation'];
export const duration = (n:number) => n<1000?`${Math.round(n)} ms`:`${(n/1000).toFixed(1)} s`;

export function currentRuns(data:Data):Run[] {
  const c=data.comparison_cohort;
  // Missing version metadata cannot establish membership in a versioned cohort.
  return data.runs.filter(r=>r.mode==='model'&&(!c||(r.hardware?.corpus_snapshot_sha256===c.corpus_snapshot_sha256&&r.hardware?.workflow_sha256===c.workflow_sha256)));
}
export function featuredCaseIds(data:Data):string[] {
  return [...new Set([...currentRuns(data).map(r=>r.case_id),...data.models.flatMap(m=>(m.failures||[]).flatMap(f=>f.case_id?[f.case_id]:[]))])].filter(id=>data.cases.some(c=>c.id===id));
}
export function attemptFor(data:Data,caseId:string,modelId:string,approach:Approach):Attempt {
  const run=currentRuns(data).filter(r=>r.case_id===caseId&&r.model_id===modelId&&Boolean(r.baseline)===(approach==='baseline')).sort((a,b)=>Date.parse(b.recorded_at)-Date.parse(a.recorded_at))[0];
  const failure=data.models.find(m=>m.id===modelId)?.failures?.find(f=>f.case_id===caseId&&Boolean(f.baseline)===(approach==='baseline'));
  return {id:`${caseId}:${modelId}:${approach}`,caseId,modelId,approach,state:run?'recorded':failure?'failed':'not-recorded',run,failure};
}
export function casePair(data:Data,caseId:string,modelId:string) {
  const baseline=attemptFor(data,caseId,modelId,'baseline'),workflow=attemptFor(data,caseId,modelId,'workflow');
  const a=baseline.run?.hardware,b=workflow.run?.hardware;
  const compatible=Boolean(a&&b&&a.revision&&b.revision&&a.revision===b.revision&&a.quantization&&a.quantization===b.quantization);
  return {baseline,workflow,compatible};
}
export function allAttempts(data:Data):Attempt[] {
  return featuredCaseIds(data).flatMap(caseId=>data.models.flatMap(m=>(['baseline','workflow'] as const).map(a=>attemptFor(data,caseId,m.id,a)))).filter(a=>a.state!=='not-recorded');
}
export function referenceRun(data:Data,caseId:string):Run|undefined {
  return data.runs.filter(r=>r.case_id===caseId&&r.mode==='reference').sort((a,b)=>Date.parse(b.recorded_at)-Date.parse(a.recorded_at))[0];
}
export function preferredModel(data:Data,caseId:string):string {
  if(!featuredCaseIds(data).includes(caseId))return 'reference';
  const complete=data.models.filter(m=>casePair(data,caseId,m.id).compatible);
  return complete.find(m=>m.family==='Spark')?.id||complete[0]?.id||data.models[0]?.id||'reference';
}
export function parseRoute(hash:string):Route {
  const [path,query='']=hash.replace(/^#\/?/,'').split('?'),parts=path.split('/'),p=new URLSearchParams(query);
  const view=parts[0]==='research'?'research':['models','corpus','method','evaluation','overview','entities'].includes(parts[0])?parts[0] as Route['view']:'overview';
  const candidate=p.get('tab');
  let pathCase='';try{pathCase=decodeURIComponent(parts[1]||'')}catch{}
  return {view,caseId:view==='research'?pathCase:p.get('case')||'',modelId:p.get('model')||'',tab:tabs.includes(candidate as CaseTab)?candidate as CaseTab:'brief',approach:p.get('approach')==='baseline'?'baseline':'workflow',sourceId:p.get('source')||''};
}
export function normalizeRoute(route:Route,data:Data):Route {
  const caseId=data.cases.some(c=>c.id===route.caseId)?route.caseId:data.cases.find(c=>c.id==='liquidity-2024-2025-pair')?.id||data.cases[0]?.id||'';
  const modelId=route.modelId==='reference'||data.models.some(m=>m.id===route.modelId)?route.modelId:preferredModel(data,caseId);
  return {...route,caseId,modelId};
}
export function formatRoute(r:Route):string {
  const p=new URLSearchParams();
  if(r.view!=='research'&&r.caseId)p.set('case',r.caseId);
  if(r.modelId)p.set('model',r.modelId);
  if(r.view==='evaluation'&&r.approach==='baseline')p.set('approach',r.approach);
  if(r.view==='research') {if(r.tab!=='brief')p.set('tab',r.tab);if(r.approach==='baseline')p.set('approach',r.approach);if(r.sourceId)p.set('source',r.sourceId);}
  return `#/${r.view}${r.view==='research'?`/${encodeURIComponent(r.caseId)}`:''}${p.size?'?'+p.toString():''}`;
}
