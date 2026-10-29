import test from 'node:test';
import assert from 'node:assert/strict';
import {searchEntityIndex} from '../src/lib/ner-results.ts';
const docs=[{id:'a',ticker:'BAC'},{id:'b',ticker:'JPM'}];
const mention=(block_id,document_id,type='ORGANIZATION')=>({block_id,document_id,start:0,end:15,type,score:.9});
const entity=(text,normalized,mentions,types=['ORGANIZATION'])=>({text,normalized,types,mentions,mention_count:mentions.length,document_ids:[],block_ids:[]});
test('browser retrieval uses actual model postings and keeps their source identity',()=>{
 const first={entities:[entity('Federal Reserve','federal reserve',[mention('a1','a'),mention('b2','b')]),entity('Federal Reserve Bank','federal reserve bank',[mention('b1','b')])]};
 const results=searchEntityIndex(first,'Federal Reserve','all','all',docs);
 assert.equal(results.length,3);assert.equal(results[0].exactMatches,1);
 assert.deepEqual(new Set(results.map(r=>r.documentId)),new Set(['a','b']));
 assert.deepEqual(searchEntityIndex(first,'Federal Reserve','all','BAC',docs).map(r=>r.blockId),['a1']);
 assert.equal(searchEntityIndex({entities:[]},'Federal Reserve','all','all',docs).length,0);
});
test('entity filters and empty queries cannot become invented matches',()=>{
 const model={entities:[entity('United States','united states',[mention('a1','a','LOCATION')],['LOCATION']),entity('','',[mention('b1','b')])]};
 assert.equal(searchEntityIndex(model,'united states','ORGANIZATION','all',docs).length,0);
 assert.equal(searchEntityIndex(model,'united states','LOCATION','all',docs).length,1);
 assert.equal(searchEntityIndex(model,'x','all','all',docs).length,0);
 assert.equal(searchEntityIndex(model,'Unknown Entity','all','all',docs).length,0);
});
