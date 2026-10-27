import {useEffect,useMemo,useState,useRef} from 'react';
import type {Evidence,Run} from '../types';

export default function EvidenceConstellation({run,onSource}:{run:Run;onSource:(e:Evidence)=>void}) {
  const [claim,setClaim]=useState(0),[hovered,setHovered]=useState('');
  const [positions,setPositions]=useState<Record<number,number>>({});
  const dragging=useRef<number|null>(null);
  useEffect(()=>{setClaim(0);setHovered('');setPositions({})},[run.id]);
  const cited=useMemo(()=>run.evidence.filter(e=>run.output.findings.some(f=>f.evidence_ids.includes(e.id))),[run]);
  const sources=cited.length?cited:run.evidence;
  const documents=[...new Set(sources.map(e=>e.document_id))];
  const height=Math.max(370,sources.length*49+65),spacing=(count:number)=>(height-110)/Math.max(1,count-1);
  const findings=run.output.findings,active=findings[claim];
  const findingY=(i:number)=>positions[i]??65+i*spacing(findings.length);
  const selectedEvidence=hovered||active?.evidence_ids[0];
  return <section className="cf-constellation"><div className="cf-section-title"><div><span className="cf-kicker">A finding is a starting point</span><h2>Follow the evidence.</h2></div><span className="cf-count">{findings.length} findings / {sources.length} passages / {documents.length} reports</span></div>
    <p className="cf-caption">Select a finding to illuminate its citations; drag a finding to rearrange the view. Select a passage to read the original bounded excerpt.</p>
    <div className="cf-plot-scroll"><svg viewBox={`0 0 850 ${height}`} role="group" aria-label="Evidence constellation connecting recorded findings to cited passages and original reports. Equivalent finding and passage controls follow the diagram." onPointerMove={e=>{if(dragging.current===null)return;const matrix=e.currentTarget.getScreenCTM();if(!matrix)return;const point=new DOMPoint(e.clientX,e.clientY).matrixTransform(matrix.inverse());const index=dragging.current;setPositions(p=>({...p,[index]:Math.max(55,Math.min(height-40,point.y))}))}} onPointerUp={()=>{dragging.current=null}} onPointerCancel={()=>{dragging.current=null}}>
      <text x="30" y="25" className="cf-axis-title">Recorded findings</text><text x="350" y="25" className="cf-axis-title">Cited passages</text><text x="655" y="25" className="cf-axis-title">Original reports</text>
      {findings.map((f,i)=>f.evidence_ids.map(id=>{const j=sources.findIndex(e=>e.id===id);if(j<0)return null;return <path key={`${i}-${id}`} className={`cf-evidence-edge ${i===claim?'is-lit':''}`} d={`M135 ${findingY(i)} C255 ${findingY(i)},245 ${65+j*spacing(sources.length)},350 ${65+j*spacing(sources.length)}`}/>;}))}
      {sources.map((e,i)=>{const d=documents.indexOf(e.document_id),lit=active?.evidence_ids.includes(e.id)||hovered===e.id;return <path key={e.id} className={`cf-evidence-edge ${lit?'is-lit':''}`} d={`M475 ${65+i*spacing(sources.length)} C560 ${65+i*spacing(sources.length)},550 ${65+d*spacing(documents.length)},655 ${65+d*spacing(documents.length)}`}/>;})}
      {findings.map((f,i)=><g key={i} className={`cf-evidence-node ${claim===i?'is-active':''}`} role="button" tabIndex={0} aria-label={`Select finding ${i+1}: ${f.claim}`} onPointerDown={e=>{dragging.current=i;setClaim(i);e.currentTarget.setPointerCapture(e.pointerId)}} onClick={()=>setClaim(i)} onKeyDown={e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();setClaim(i)}}}><circle cx="110" cy={findingY(i)} r="23"/><text x="110" y={findingY(i)+5} textAnchor="middle">F{i+1}</text></g>)}
      {!findings.length&&<text x="30" y="70" className="cf-axis-note">Extractive reference</text>}
      {sources.map((e,i)=><g key={e.id} className={`cf-passage-node ${active?.evidence_ids.includes(e.id)||hovered===e.id?'is-active':''}`} role="button" tabIndex={0} aria-label={`Open passage ${i+1}: ${e.ticker}, ${e.fiscal_period}`} onMouseEnter={()=>setHovered(e.id)} onMouseLeave={()=>setHovered('')} onFocus={()=>setHovered(e.id)} onBlur={()=>setHovered('')} onClick={()=>onSource(e)} onKeyDown={event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();onSource(e)}}}><rect x="350" y={48+i*spacing(sources.length)} width="125" height="34" rx="4"/><text x="362" y={70+i*spacing(sources.length)}>{e.ticker} {e.fiscal_period.slice(0,4)} / P{i+1}</text></g>)}
      {documents.map((id,i)=>{const e=sources.find(s=>s.document_id===id)!;return <g key={id} className={`cf-document-node ${sources.find(s=>s.id===selectedEvidence)?.document_id===id?'is-active':''}`}><rect x="655" y={44+i*spacing(documents.length)} width="170" height="44" rx="3"/><text x="667" y={63+i*spacing(documents.length)}>{e.ticker} / {e.fiscal_period}</text><text x="667" y={79+i*spacing(documents.length)} className="cf-axis-note">{e.page?`PDF page ${e.page}`:'Original filing'}</text></g>;})}
    </svg></div>
    {active&&<div className="cf-selected-finding" aria-live="polite"><span>Finding {claim+1}</span><p>{active.claim}</p><div className="cf-chips">{active.evidence_ids.map(id=>{const e=run.evidence.find(x=>x.id===id);return e?<button key={id} onClick={()=>onSource(e)}>{e.ticker} / {e.fiscal_period} ↗</button>:null;})}</div></div>}
    <details className="cf-disclosure"><summary>All findings and passages</summary>{findings.map((f,i)=><button className="cf-text-row" key={i} onClick={()=>setClaim(i)}>Finding {i+1}: {f.claim}</button>)}{sources.map((e,i)=><button className="cf-text-row" key={e.id} onClick={()=>onSource(e)}>Passage {i+1} · {e.ticker} · {e.fiscal_period} · {e.text.slice(0,100)}…</button>)}</details>
    <p className="cf-footnote">Edges represent the model's cited source IDs. Position, connections, and illumination are not measures of claim support or confidence.</p>
  </section>;
}
