import {useMemo,useState} from 'react';
import type {Data} from '../types';
import {allAttempts,type Attempt} from '../lib/case-files';

export default function OutcomeRiver({data,caseId,onOpen}:{data:Data;caseId?:string;onOpen:(attempt:Attempt)=>void}) {
  const [model,setModel]=useState('all'),[selected,setSelected]=useState('');
  const attempts=useMemo(()=>allAttempts(data).filter(a=>(!caseId||a.caseId===caseId)&&(model==='all'||a.modelId===model)),[data,caseId,model]);
  const valid=attempts.filter(a=>a.state==='recorded').length,failed=attempts.filter(a=>a.state==='failed').length;
  const models=data.models.filter(m=>attempts.some(a=>a.modelId===m.id));
  const chosen=attempts.find(a=>a.id===selected);
  const count=(approach:string)=>attempts.filter(a=>a.approach===approach).length;
  if(!attempts.length)return <section className="cf-missing"><h2>No current-cohort model attempts.</h2><p>This investigation has an extractive reference. Its model comparison has not been recorded.</p></section>;
  return <section className="cf-river" aria-label="Recorded experiment outcome river">
    <div className="cf-section-title"><div><span className="cf-kicker">Every attempt accounted for</span><h2>Follow the outcomes.</h2></div><label>Checkpoint<select aria-label="Filter outcome river checkpoint" value={model} onChange={e=>{setModel(e.target.value);setSelected('')}}><option value="all">All five checkpoints</option>{data.models.map(m=><option key={m.id} value={m.id}>{m.name}</option>)}</select></label></div>
    <p className="cf-caption">{attempts.length} recorded attempts · {valid} schema-valid outputs · {failed} schema failures. Select a stream to inspect its case.</p>
    <div className="cf-plot-scroll"><svg viewBox="0 0 850 370" role="group" aria-label={`Outcome river: ${attempts.length} attempts, ${valid} schema-valid, ${failed} failed. Select an attempt using the list below or a stream.`}>
      <text x="24" y="25" className="cf-axis-title">Checkpoint</text><text x="375" y="25" className="cf-axis-title">Approach</text><text x="690" y="25" className="cf-axis-title">Output contract</text>
      {attempts.map(a=>{
        const mi=models.findIndex(m=>m.id===a.modelId),siblings=attempts.filter(b=>b.modelId===a.modelId),si=siblings.findIndex(b=>b.id===a.id);
        const ay=a.approach==='baseline'?125:270,stage=attempts.filter(b=>b.approach===a.approach),pi=stage.findIndex(b=>b.id===a.id);
        const oy=a.state==='recorded'?125:270,ends=attempts.filter(b=>b.state===a.state),ei=ends.findIndex(b=>b.id===a.id);
        const sy=65+mi*(models.length>1?250/(models.length-1):0)+si*5,cy=ay+(pi-(stage.length-1)/2)*5,ey=oy+(ei-(ends.length-1)/2)*5;
        return <g key={a.id} className={`cf-stream ${a.state==='failed'?'is-failed':''} ${chosen&&chosen.id!==a.id?'is-dimmed':''} ${selected===a.id?'is-selected':''}`} role="button" tabIndex={0} aria-label={`${data.models.find(m=>m.id===a.modelId)?.name}, ${a.caseId}, ${a.approach}, ${a.state==='recorded'?'schema-valid':'schema failure'}`} onClick={()=>setSelected(a.id)} onKeyDown={e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();setSelected(a.id)}}}>
          <title>{a.caseId} · {a.approach} · {a.state}</title><path className="cf-stream-hit" d={`M200 ${sy} C285 ${sy},300 ${cy},390 ${cy} S560 ${ey},680 ${ey}`}/><path className="cf-stream-line" d={`M200 ${sy} C285 ${sy},300 ${cy},390 ${cy} S560 ${ey},680 ${ey}`}/>
        </g>;
      })}
      {models.map((m,i)=><g key={m.id}><text x="24" y={70+i*(models.length>1?250/(models.length-1):0)}>{m.name}</text><rect x="185" y={57+i*(models.length>1?250/(models.length-1):0)} width="7" height="27" rx="2" className="cf-river-node"/></g>)}
      <g className="cf-river-label"><rect x="390" y="99" width="8" height="52" rx="2"/><text x="410" y="88">Baseline</text><text x="410" y="108" className="cf-axis-note">{count('baseline')} attempts</text><rect x="390" y="244" width="8" height="52" rx="2"/><text x="410" y="233">LangGraph</text><text x="410" y="253" className="cf-axis-note">{count('workflow')} attempts</text></g>
      <g className="cf-river-label"><rect x="680" y="86" width="8" height="78" rx="2"/><text x="704" y="119">Schema-valid</text><text x="704" y="140" className="cf-axis-note">{valid} outputs</text><rect x="680" y="244" width="8" height="52" rx="2" className="cf-failed-node"/><text x="704" y="263">Schema failed</text><text x="704" y="284" className="cf-axis-note">{failed} attempts</text></g>
    </svg></div>
    <div className="cf-river-inspection" aria-live="polite">{chosen?<><div><strong>{data.models.find(m=>m.id===chosen.modelId)?.name} / {chosen.approach==='baseline'?'Baseline':'LangGraph workflow'}</strong><p>{data.cases.find(c=>c.id===chosen.caseId)?.title} · {data.cases.find(c=>c.id===chosen.caseId)?.tickers.join(' / ')} · {chosen.state==='recorded'?'Schema-valid memo':chosen.failure?.error}{chosen.failure?.output_limit_reached?' · output token limit reached':''}</p></div><button className="cf-primary" onClick={()=>onOpen(chosen)}>Open this case</button></>:<p>Each line is one recorded attempt. Solid lines reached a schema-valid output; dashed lines failed the output contract.</p>}</div>
    <details className="cf-disclosure"><summary>Inspect all {attempts.length} attempts</summary><div className="cf-attempt-list">{attempts.map(a=><button key={a.id} onClick={()=>{setSelected(a.id);onOpen(a)}}><span>{data.models.find(m=>m.id===a.modelId)?.name}</span><span>{data.cases.find(c=>c.id===a.caseId)?.tickers.join(' / ')} · {a.approach}</span><strong>{a.state==='recorded'?'Schema-valid':a.failure?.error||'Failed'}</strong></button>)}</div></details>
    <p className="cf-footnote">Line positions are categorical. Counts use the current corpus/workflow cohort. Schema validity and source-ID membership do not establish factual accuracy.</p>
  </section>;
}
