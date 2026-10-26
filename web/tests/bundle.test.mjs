import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

const data = JSON.parse(readFileSync(new URL('../public/data/workbench.json',import.meta.url),'utf8'));
test('public snapshot has source provenance and no fabricated pending scores',()=>{
  assert.equal(data.models.length,5);
  const documents=new Set(data.documents.map(d=>d.id));
  for(const d of data.documents){assert.match(d.sha256,/^[a-f0-9]{64}$/);assert.match(d.url,/^https:\/\//)}
  for(const model of data.models){if(model.status==='not_run'){assert.equal(model.runs,0);assert.equal(model.citation_validity,null);assert.equal(model.mean_duration_ms,null)}}
  for(const run of data.runs){
    const ids=new Set(run.evidence.map(e=>e.id));
    for(const e of run.evidence){assert.ok(documents.has(e.document_id));assert.ok(e.text.length<=1250)}
    for(const f of run.output.findings) for(const id of f.evidence_ids) assert.ok(ids.has(id));
    if(run.mode==='reference'){assert.equal(run.model_id,null);assert.equal(run.output.findings.length,0)}
  }
});
test('downloadable Colab notebook is valid and contains no saved outputs or keys',()=>{
  const text=readFileSync(new URL('../public/data/benchmark.ipynb',import.meta.url),'utf8');
  const notebook=JSON.parse(text);
  assert.equal(notebook.nbformat,4);
  assert.ok(notebook.cells.every(c=>c.cell_type!=='code'||(c.execution_count===null&&c.outputs.length===0)));
  assert.doesNotMatch(text,/lsv2_[A-Za-z0-9_-]{16,}/);
});
