import type {Data, Run} from '../types';
import type {Attempt} from './case-files';

export const MODEL_IDS = ['Qwen/Qwen3.5-9B','XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B','XHToken/Spark-X2.6-4B','mistralai/Ministral-3-3B-Instruct-2512','ibm-granite/granite-4.1-8b'];
export type Metric = 'generation_seconds'|'tokens_per_second'|'peak_gpu_memory_gb'|'input_tokens'|'output_tokens';
export const METRICS:Record<Metric,{label:string;unit:string}> = {
  generation_seconds:{label:'Generation time',unit:'s'},tokens_per_second:{label:'Generation throughput',unit:'tokens/s'},peak_gpu_memory_gb:{label:'Peak GPU memory',unit:'GiB'},input_tokens:{label:'Input tokens',unit:'tokens'},output_tokens:{label:'Output tokens',unit:'tokens'},
};
export const mean = (values:number[]) => values.length?values.reduce((a,b)=>a+b,0)/values.length:null;
export function metricValue(attempt:Attempt, metric:Metric):number|null {
  const value = attempt.run?.generation?.[metric];
  return typeof value==='number'&&Number.isFinite(value)?value:null;
}
export function experimentURL(p:{org_id:string;dataset_id:string;experiments:{model_id:string;session_id:string}[]},modelId?:string):string {
  const selected=modelId?p.experiments.filter(e=>e.model_id===modelId):p.experiments;
  return `https://smith.langchain.com/o/${p.org_id}/datasets/${p.dataset_id}/compare?selectedSessions=${encodeURIComponent(selected.map(e=>e.session_id).join(','))}`;
}
export function promptPreview(data:Data,run:Run,p:{workflow_sha256:string;prompt_template:string;output_schema:unknown}):string|null {
  if(run.hardware?.workflow_sha256!==p.workflow_sha256)return null;
  const c=data.cases.find(c=>c.id===run.case_id);
  if(!c)return null;
  const variables:Record<string,string>={
    schema:JSON.stringify(p.output_schema),question:c.query,
    scope:JSON.stringify({tickers:c.tickers,years:c.years,as_of:c.as_of??null}),
    coverage_gaps:JSON.stringify(run.coverage_gaps||[]),
    evidence:JSON.stringify(run.evidence.map(e=>({id:e.id,ticker:e.ticker,period:e.fiscal_period,text:Array.from(e.text).slice(0,1250).join('')}))),
  };
  return p.prompt_template.replace(/\{\{(\w+)\}\}/g,(_,key)=>variables[key]??`{{${key}}}`);
}
export function pairedDelta(a:Attempt,b:Attempt,metric:Metric):number|null {
  if(a.caseId!==b.caseId||a.modelId!==b.modelId||a.approach===b.approach)return null;
  const x=a.run?.hardware,y=b.run?.hardware;
  if(!x?.revision||x.revision!==y?.revision||!x.quantization||x.quantization!==y.quantization||!x.corpus_snapshot_sha256||x.corpus_snapshot_sha256!==y.corpus_snapshot_sha256||!x.workflow_sha256||x.workflow_sha256!==y.workflow_sha256)return null;
  const av=metricValue(a,metric),bv=metricValue(b,metric);
  return av===null||bv===null?null:bv-av;
}
