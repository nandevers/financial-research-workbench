import {useEffect,useState} from 'react';
import type {Data} from '../types';
import {loadNerResults,type NerResultState} from './ner-results';

export default function useNerResults(data:Data){
  const [attempt,setAttempt]=useState(0),[state,setState]=useState<NerResultState>({status:'loading',results:null,message:'Checking the saved extraction and retrieval results…'});
  const snapshot=data.comparison_cohort?.corpus_snapshot_sha256,documentIds=data.documents.map(d=>d.id).sort().join('|');
  useEffect(()=>{
    const controller=new AbortController();let active=true;
    setState({status:'loading',results:null,message:'Checking the saved extraction and retrieval results…'});
    loadNerResults(`${import.meta.env.BASE_URL}data/ner-results.json`,snapshot,documentIds.split('|'),fetch,controller.signal).then(next=>{if(active)setState(next)});
    return()=>{active=false;controller.abort()};
  },[snapshot,documentIds,attempt]);
  return {state,retry:()=>setAttempt(n=>n+1)};
}
