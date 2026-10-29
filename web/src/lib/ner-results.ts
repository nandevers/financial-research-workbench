export type NerMention={block_id:string;document_id:string;start:number;end:number;type:string;score:number};
export type NerEntity={text:string;normalized:string;types:string[];mention_count:number;document_ids:string[];block_ids:string[];mentions:NerMention[]};
export type NerHit={block_id:string;document_id:string;score:number;lexical_score:number;entity_overlap:number};
export type NerQuery={id:string;query:string;query_entities:{start:number;end:number;text:string;type:string;score:number}[];lexical:NerHit[];entity_assisted:NerHit[];entity_boost_applied:boolean;query_entity_keys:string[];known_positive_metrics?:{lexical:Record<string,number|null>;entity_assisted:Record<string,number|null>}};
export type NerReplayExperiment={id:string;name:string;dataset_id:string;dataset_name:string;url:string;task:'ner'|'retrieval';variant:'heldout'|'lexical'|'entity_assisted';rows:number;recovery_sha256?:string;summary?:Record<string,unknown>};
export type NerLangSmith={dataset_id?:string;dataset_name?:string;experiment_name?:string;experiment_url?:string;dataset_url?:string;source?:string;evaluation?:unknown;replay?:{schema_version?:number;status?:string;model_id?:string;recovery_sha256?:string;experiments?:NerReplayExperiment[]}};
export type NerRecovery={schema_version:1;policy:string;status:'complete';recovery_sha256:string;identity:{model_id:string;revision:string;base_configuration_sha256:string;corpus_snapshot_sha256:string;corpus_content_sha256:string;failed_block_ids:string[];[key:string]:unknown};append_records:{block_id:string;parent_prediction_sha256:string;appended_prediction_sha256:string;native_audit_record_sha256:string}[];original_success_count:number;remaining_failed_count:number;[key:string]:unknown};
export type NerModelResult={id:string;name:string;license:string;revision:string;profile:string;status:{status:string;processed_blocks:number;total_blocks:number;processed_documents:number;total_documents:number;failed_blocks:number;reference_failures?:number;query_failures?:number;langsmith_error?:string;model_load_seconds?:number;corpus_inference_seconds?:number};metadata:Record<string,unknown>;entities:NerEntity[];entity_index_url?:string;entity_count?:number;entity_index_sha256?:string;retrieval:NerQuery[];langsmith?:NerLangSmith|null;recovery?:NerRecovery;evaluation:{reference_status:string;boundary:string;test_passages:number;missing_predictions:number;metrics:{precision:number;recall:number;f1:number;tp:number;fp:number;fn:number}|null;per_type:unknown;passages:unknown[]}};
export type NerResults={schema_version:number;prepared_at:string;corpus_snapshot_sha256:string;corpus_content_sha256:string;source_documents:number;source_blocks:number;complete_models:number;completion_status:string;documents:{id:string;issuer:string;ticker:string;fiscal_period:string;filing_date:string|null;url:string;sha256:string}[];excerpts:Record<string,{id:string;document_id:string;ticker:string;fiscal_period:string;source_url:string;section:string;kind:string;page?:number;text:string;full_text_length:number;truncated:boolean}>;models:NerModelResult[];method:Record<string,unknown>};
export const NER_EXPECTED={documents:30,blocks:26058,referencePassages:40,testPassages:32,developmentPassages:8,modelIds:['dslim/distilbert-NER','dslim/bert-base-NER','urchade/gliner_medium-v2.1','numind/NuNER_Zero','fastino/gliner2-base-v1']};
export type NerResultState={status:'loading'|'missing'|'error';results:null;message:string}|{status:'ready';results:NerResults;message:string};
/** Structural display check; the offline importer verifies hashes and native audit. */
export function nerRecoveryProblem(value:unknown,modelId:string,revision:string,sourceBlocks:number,corpusContent:string,corpusSnapshot:string):string|null{
  const r=value as NerRecovery|null,i=r?.identity,hash=/^[a-f0-9]{64}$/;
  if(!r||r.schema_version!==1||r.policy!=='gliner2-2.1.0-synthetic-terminal-period-v1'||r.status!=='complete'||modelId!=='fastino/gliner2-base-v1'||!hash.test(r.recovery_sha256||''))return 'The targeted processing repair has no valid completed identity.';
  if(!i||i.model_id!==modelId||i.revision!==revision||i.corpus_content_sha256!==corpusContent||i.corpus_snapshot_sha256!==corpusSnapshot||!hash.test(i.base_configuration_sha256||''))return 'The targeted processing repair does not match this model and collection.';
  const failed=i.failed_block_ids,appended=r.append_records;
  if(!Array.isArray(failed)||!failed.length||failed.some(id=>typeof id!=='string')||new Set(failed).size!==failed.length||!Array.isArray(appended)||appended.length!==failed.length||appended.some(row=>!row||typeof row.block_id!=='string'||!hash.test(row.parent_prediction_sha256||'')||!hash.test(row.appended_prediction_sha256||'')||!hash.test(row.native_audit_record_sha256||''))||[...new Set(appended.map(row=>row.block_id))].sort().join('|')!==[...failed].sort().join('|')||r.remaining_failed_count!==0||!Number.isInteger(r.original_success_count)||r.original_success_count+failed.length!==sourceBlocks)return 'The targeted processing repair does not account for every reprocessed passage.';
  return null;
}
/** A complete-model count alone cannot prove five distinct checkpoints finished. */
export function nerResultsProblem(value:unknown,expectedSnapshot?:string,expectedDocumentIds?:string[]):string|null {
  if(!value||typeof value!=='object')return 'The result file is not a recorded NER result bundle.';
  const r=value as Partial<NerResults>;
  if(r.schema_version!==1||r.completion_status!=='complete'||r.complete_models!==5)return 'The published result bundle does not establish completion of all five models.';
  if(r.source_documents!==NER_EXPECTED.documents||r.source_blocks!==NER_EXPECTED.blocks||!Array.isArray(r.documents)||r.documents.length!==NER_EXPECTED.documents||r.documents.some(d=>!d||typeof d.id!=='string')||new Set(r.documents.map(d=>d.id)).size!==NER_EXPECTED.documents)return 'The result bundle does not cover the expected 30 reports and 26,058 passages.';
  if(!/^[a-f0-9]{64}$/.test(r.corpus_content_sha256||'')||(expectedSnapshot&&r.corpus_snapshot_sha256!==expectedSnapshot))return 'The result bundle does not match this research collection.';
  if(expectedDocumentIds&&[...expectedDocumentIds].sort().join('|')!==r.documents.map(d=>d.id).sort().join('|'))return 'The result documents do not match this research collection.';
  if(!r.excerpts||typeof r.excerpts!=='object'||Array.isArray(r.excerpts)||!Array.isArray(r.models)||r.models.length!==5)return 'The result bundle is missing its recorded models or source excerpts.';
  const ids=r.models.map(m=>m?.id);
  if(new Set(ids).size!==5||NER_EXPECTED.modelIds.some(id=>!ids.includes(id)))return 'Five unique expected specialist models are required before measured results can be shown.';
  for(const m of r.models){
    const s=m?.status,e=m?.evaluation;
    if(s?.status!=='complete'||s.processed_blocks!==NER_EXPECTED.blocks||s.total_blocks!==NER_EXPECTED.blocks||s.processed_documents!==NER_EXPECTED.documents||s.total_documents!==NER_EXPECTED.documents||s.failed_blocks!==0||s.reference_failures!==0||s.query_failures!==0)return 'At least one model has incomplete corpus, reference or query processing.';
    if(!Array.isArray(m.entities)||!Array.isArray(m.retrieval)||!e||e.test_passages!==NER_EXPECTED.testPassages||e.missing_predictions!==0||!e.metrics||e.reference_status!=='agent_reviewed_not_human_gold')return 'The result bundle is missing the reviewed held-out reference evaluation.';
    if(['precision','recall','f1'].some(k=>{const n=e.metrics?.[k as 'precision'];return typeof n!=='number'||!Number.isFinite(n)||n<0||n>1}))return 'The result bundle contains invalid reference scores.';
    if(m.recovery){
      const problem=nerRecoveryProblem(m.recovery,m.id,m.revision,NER_EXPECTED.blocks,r.corpus_content_sha256!,r.corpus_snapshot_sha256!);if(problem)return problem;
      if(m.langsmith?.replay&&m.langsmith.replay.recovery_sha256!==m.recovery.recovery_sha256)return 'The replay experiment does not identify the recorded processing repair.';
    }
  }
  return null;
}
export async function loadNerResults(url:string,expectedSnapshot?:string,expectedDocumentIds?:string[],fetcher:typeof fetch=fetch,signal?:AbortSignal):Promise<NerResultState>{
  try{
    const response=await fetcher(url,{signal});
    if(response.status===404)return {status:'missing',results:null,message:'A complete measured NER bundle is not included in this build.'};
    if(!response.ok)throw Error(`The saved NER results could not be loaded (HTTP ${response.status}).`);
    const result:unknown=await response.json(),problem=nerResultsProblem(result,expectedSnapshot,expectedDocumentIds);
    if(problem)return {status:'error',results:null,message:problem};
    return {status:'ready',results:result as NerResults,message:'All five expected model results are available.'};
  }catch(error){return {status:'error',results:null,message:error instanceof SyntaxError?'The saved NER result file could not be read.':error instanceof Error?error.message:'The saved NER results could not be loaded.'}}
}
export function langsmithResultLink(record:NerLangSmith|null|undefined,orgId:string):{url:string;label:string}|null{
  for(const [candidate,label] of [[record?.experiment_url,'Open this experiment in LangSmith'],[record?.dataset_url,'Open the reference dataset in LangSmith']] as const){
    if(candidate){try{const u=new URL(candidate);if(u.protocol==='https:'&&u.hostname==='smith.langchain.com'&&!u.username&&!u.password)return {url:u.href,label}}catch{}}
  }
  if(record?.dataset_id&&/^[a-f0-9-]{36}$/.test(record.dataset_id)&&/^[a-f0-9-]{36}$/.test(orgId))return {url:`https://smith.langchain.com/o/${orgId}/datasets/${record.dataset_id}`,label:'Open the reference dataset in LangSmith'};
  return null;
}
export function entityIndexProblem(value:unknown,modelId:string,corpusContent:string):string|null{
  const r=value as {schema_version?:number;model_id?:string;corpus_content_sha256?:string;entities?:NerEntity[]}|null;
  if(!r||r.schema_version!==1||r.model_id!==modelId||r.corpus_content_sha256!==corpusContent)return 'Entity index provenance does not match these results.';
  if(!Array.isArray(r.entities)||r.entities.some(e=>!e||typeof e.text!=='string'||typeof e.normalized!=='string'||!Array.isArray(e.types)||e.types.some(t=>typeof t!=='string')||!Array.isArray(e.mentions)||e.mentions.some(m=>!m||typeof m.block_id!=='string'||typeof m.document_id!=='string'||typeof m.type!=='string'||!Number.isInteger(m.start)||!Number.isInteger(m.end)||m.start<0||m.end<=m.start||typeof m.score!=='number'||!Number.isFinite(m.score))))return 'The saved entity index is missing valid model predictions.';
  return null;
}
/** Python model offsets count Unicode code points, rather than JS UTF-16 units. */
export function predictedSegments(text:string,mentions:NerMention[]):{text:string;mention?:NerMention}[]{
  const characters=Array.from(text),accepted:NerMention[]=[];
  for(const m of [...mentions].filter(m=>m.start>=0&&m.end<=characters.length&&m.end>m.start).sort((a,b)=>(b.end-b.start)-(a.end-a.start))){if(!accepted.some(s=>m.start<s.end&&m.end>s.start))accepted.push(m)}
  const pieces:{text:string;mention?:NerMention}[]=[];let offset=0;
  for(const m of accepted.sort((a,b)=>a.start-b.start)){pieces.push({text:characters.slice(offset,m.start).join('')},{text:characters.slice(m.start,m.end).join(''),mention:m});offset=m.end}
  pieces.push({text:characters.slice(offset).join('')});return pieces;
}
export const normalizeEntityQuery=(value:string)=>value.normalize('NFKC').toLowerCase().replace(/\s+/g,' ').trim();
export function searchEntityIndex(model:NerModelResult,query:string,kind='all',ticker='all',documents:NerResults['documents']=[]) {
  const q=normalizeEntityQuery(query),docMap=new Map(documents.map(d=>[d.id,d]));
  if(q.length<2)return [];
  const found=model.entities.filter(e=>e.normalized.length>=2&&(kind==='all'||e.types.includes(kind))&&(e.normalized.includes(q)||q.includes(e.normalized)));
  const blocks=new Map<string,{blockId:string;documentId:string;entities:NerEntity[];mentions:NerMention[];exactMatches:number}>();
  for(const e of found)for(const m of e.mentions){
    if(ticker!=='all'&&docMap.get(m.document_id)?.ticker!==ticker)continue;
    if(kind!=='all'&&m.type!==kind)continue;
    const hit=blocks.get(m.block_id)||{blockId:m.block_id,documentId:m.document_id,entities:[],mentions:[],exactMatches:0};
    if(!hit.entities.some(v=>v.normalized===e.normalized)){hit.entities.push(e);if(e.normalized===q)hit.exactMatches++}
    hit.mentions.push(m);blocks.set(m.block_id,hit);
  }
  return [...blocks.values()].sort((a,b)=>b.exactMatches-a.exactMatches||b.entities.length-a.entities.length||b.mentions.length-a.mentions.length||a.blockId.localeCompare(b.blockId));
}
