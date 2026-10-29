import type {NerResults} from '../lib/ner-results';
import findings from '../data/ner-findings.json';

const percent=(n:number)=>`${(n*100).toFixed(1)}%`;
const recall=(m:typeof findings.models[number],variant:'lexical'|'entity_assisted')=>m[variant].mean_test_known_positive_recall_at_5;
const rr=(m:typeof findings.models[number],variant:'lexical'|'entity_assisted')=>m[variant].mean_test_first_known_positive_reciprocal_rank_at_5;
function matches(results:NerResults){
  return results.corpus_content_sha256===findings.corpus_content_sha256&&findings.models.every(row=>{
    const model=results.models.find(m=>m.id===row.id);
    return model?.metadata.configuration_sha256===row.configuration_sha256&&model?.evaluation.metrics?.f1===row.tagging.f1;
  });
}

export default function NerFindings({results}:{results:NerResults}){
  if(!matches(results))return null;
  const tagging=[...findings.models].sort((a,b)=>b.tagging.f1-a.tagging.f1);
  const retrieval=[...findings.models].sort((a,b)=>recall(b,'entity_assisted')-recall(a,'entity_assisted'));
  const winner=tagging[0],runnerUp=tagging[1],bestRetrieval=retrieval[0];
  return <section className="mn-section mn-findings" id="ner-findings"><h2>What did the comparison show?</h2>
    <div className="mn-pair"><article><h3>{winner.name} led name tagging.</h3><p>It reached <strong>{percent(winner.tagging.f1)} exact-span F1</strong> on 32 held-out snippets, with {percent(winner.tagging.precision)} precision and {percent(winner.tagging.recall)} recall. {runnerUp.name} reached {percent(runnerUp.tagging.f1)} F1. Both found 108 correct names and missed nine; {winner.name} produced 30 extra or incorrect tags, compared with 51 from {runnerUp.name}.</p></article>
    <article><h3>Retrieval gains were small and model-dependent.</h3><p>{bestRetrieval.name} gave the largest measured gain: mean known-positive recall@5 rose from <strong>{percent(recall(bestRetrieval,'lexical'))} to {percent(recall(bestRetrieval,'entity_assisted'))}</strong>, a {((recall(bestRetrieval,'entity_assisted')-recall(bestRetrieval,'lexical'))*100).toFixed(1)} percentage-point increase across nine test queries. Its recall improved on one query and tied on eight. GLiNER Medium and GLiNER2 did not improve the held-out retrieval measurements.</p></article></div>
    <p><strong>Better tagging did not automatically produce better search.</strong> GLiNER Medium led tagging, but BERT led this retrieval sample. Some ranking changes did not increase judged evidence coverage. Keep lexical search as the baseline; broader human relevance judgments and more test queries are needed before choosing an entity-assisted ranking for general use.</p>
    <div className="mn-table-wrap"><table><caption>Held-out retrieval: nine test queries, top five results</caption><thead><tr><th>Entity model</th><th>Lexical<br/>Recall@5</th><th>Entity-assisted<br/>Recall@5</th><th>Change<br/>percentage points</th><th>Lexical<br/>MRR@5</th><th>Entity-assisted<br/>MRR@5</th></tr></thead><tbody>{findings.models.map(m=><tr key={m.id}><th>{m.name}</th><td>{percent(recall(m,'lexical'))}</td><td>{percent(recall(m,'entity_assisted'))}</td><td>{recall(m,'entity_assisted')>recall(m,'lexical')?'+':''}{((recall(m,'entity_assisted')-recall(m,'lexical'))*100).toFixed(1)}</td><td>{rr(m,'lexical').toFixed(3)}</td><td>{rr(m,'entity_assisted').toFixed(3)}</td></tr>)}</tbody></table></div>
    <p className="mn-note">Recall@5 is the fraction of already judged positive passages found in the first five results, averaged over test queries. MRR@5 averages the reciprocal rank of the first such hit; a miss scores zero. Only positive passages in the held-out reference fold count here. The development query and development passages are excluded. Relevance judgments are incomplete: these are neither exhaustive recall nor precision@5.</p>
    <p className="mn-note">DistilBERT raised mean recall@5 to 20.4%. NuNER kept recall unchanged and moved the first known positive higher in one query. BERT brought an Alan D. Schnitzer reference passage into fifth place and moved an Evan G. Greenberg hit from fourth to third. The sample is agent-authored and separately agent-reviewed, not human gold; these descriptive differences do not establish statistical significance or financial accuracy.</p>
  </section>;
}

export function RecordedQueryMetrics({results,modelId,queryId}:{results:NerResults;modelId:string;queryId:string}){
  if(!matches(results))return null;
  const query=findings.models.find(m=>m.id===modelId)?.queries.find(q=>q.id===queryId);
  if(!query)return null;
  return <details><summary>{query.split==='test'?'Held-out measurements for this test query':'Development query measurements / excluded from the headline comparison'}</summary>
    <p className="mn-note">{query.lexical.known_positive_blocks_in_fold} known positive {query.lexical.known_positive_blocks_in_fold===1?'passage':'passages'} in this query’s {query.split==='test'?'held-out':'development'} fold. Evidence from the other fold is excluded.</p>
    <div className="mn-table-wrap"><table><thead><tr><th>Metric</th><th>Word matching</th><th>Entity-assisted</th></tr></thead><tbody><tr><th>Known-positive recall@5</th><td>{percent(query.lexical.known_positive_recall_at_5)}</td><td>{percent(query.entity_assisted.known_positive_recall_at_5)}</td></tr><tr><th>First known-positive reciprocal rank@5</th><td>{query.lexical.first_known_positive_reciprocal_rank_at_5.toFixed(3)}</td><td>{query.entity_assisted.first_known_positive_reciprocal_rank_at_5.toFixed(3)}</td></tr></tbody></table></div>
  </details>;
}
