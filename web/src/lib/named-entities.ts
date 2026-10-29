import type {Data, Evidence, Run} from '../types';

export type EntityType='PERSON'|'ORGANIZATION'|'LOCATION'|'DATE'|'MONEY'|'PERCENTAGE';
export type EditorialAnnotation={surface:string;type:string;explanation:string};
export type EntitySpan={start:number;end:number;surface:string;type:string;explanation:string};
export type NerModel={id:string;name:string;license:string;language:string;size:string;method:string;role:string;flexible_labels:boolean;common_labels:string[];native_labels:string[];domain_limit:string;source:string;results:{precision:null;recall:null;f1:null;latency_ms:null}};
export type EditorialExample={id:string;title:string;evidence_id:string;editorial:boolean;annotations:EditorialAnnotation[]};

/** Exact literal matches for an explicitly editorial illustration, never model output. */
export function editorialSpans(text:string,annotations:EditorialAnnotation[]):EntitySpan[] {
  const candidates:EntitySpan[]=[];
  for(const item of annotations){
    if(!item.surface)continue;
    let from=0;
    while(from<text.length){
      const start=text.indexOf(item.surface,from);
      if(start<0)break;
      candidates.push({...item,start,end:start+item.surface.length});from=start+item.surface.length;
    }
  }
  // Prefer a full formal name over an alias contained inside it.
  const accepted:EntitySpan[]=[];
  for(const span of candidates.sort((a,b)=>(b.end-b.start)-(a.end-a.start)||a.start-b.start)){
    if(!accepted.some(a=>span.start<a.end&&span.end>a.start))accepted.push(span);
  }
  return accepted.sort((a,b)=>a.start-b.start);
}
export function findEditorialExample(data:Data,example:EditorialExample):{run:Run;evidence:Evidence;spans:EntitySpan[]}|null {
  for(const run of data.runs){
    const evidence=run.evidence.find(e=>e.id===example.evidence_id);
    if(evidence)return {run,evidence,spans:editorialSpans(evidence.text,example.annotations)};
  }
  return null;
}
export function draftNerProtocol(data:Data,models:NerModel[]) {
  return {
    status:'draft_not_executed',prediction_runs:0,execution_boundary:'No predictions are included in this downloadable outline. This is not the status of inference running elsewhere.',reference_annotations_status:'agent_reviewed_not_human_gold',reference_passages:40,held_out_test_passages:32,development_passages:8,
    corpus_snapshot_sha256:data.comparison_cohort?.corpus_snapshot_sha256||null,
    task:'Find exact character spans for PERSON, ORGANIZATION and LOCATION in existing report passages.',
    model_ids:models.map(m=>m.id),common_labels:['PERSON','ORGANIZATION','LOCATION'],
    extended_track:{labels:['DATE','MONEY','PERCENTAGE'],model_ids:models.filter(m=>m.flexible_labels).map(m=>m.id),fixed_label_models:'not_applicable'},
    source_policy:'Existing corpus only; no new reports. Editorial illustrations are not gold test labels.',
    reference_policy:'Use the frozen 40-snippet agent-authored reference, independently checked by a second agent before inference; not human gold or professional validation. Primary scores use only the 32 held-out test snippets.',
    split_policy:'Separate development and held-out test by source document; keep overlapping or near-duplicate passages together.',
    metrics:['Exact-span-and-type entity precision','Exact-span-and-type entity recall','Exact-span-and-type entity F1','Per-type counts','Measured warmed latency and memory on the same hardware'],
    controls:['Pin model and tokenizer revisions','Freeze thresholds on development examples only','Use the same source passages and documented window/overlap policy','Preserve character offsets and merge duplicate predictions consistently','Keep loading time separate from warmed inference'],
    observability:'Record a separate NER experiment series in LangSmith only when real local or Colab runs are executed. No paid model API or LLM judge is required.',
  };
}
