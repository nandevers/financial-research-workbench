"""Freeze, isolate and execute all NER checkpoints on the complete local corpus.

Usage: python scripts/ner_colab_run.py --corpus artifacts/local/corpus.json
       --reference data/ner-reference-sample.json --resume --langsmith
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import venv
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'engine/src'))
from workbench.ner import MODEL_SPECS, PROFILE_REQUIREMENTS, corpus_fingerprints, digest, export_bundle, load_json, now, save_json, slug


def freeze_manifest(corpus, output, selected, resume):
    path=output/'manifest.json'
    if path.exists():
        if not resume:raise ValueError('Output exists; use --resume for the exact same corpus/config or choose a new --output directory')
        manifest=load_json(path)
        if manifest['corpus_content_sha256']!=corpus_fingerprints(corpus)['corpus_content_sha256']:
            raise ValueError('Corpus content changed; refusing resume')
        if [s['id'] for s in manifest['models']]!=[s['id'] for s in selected]:
            raise ValueError('Selected model list changed; use a new output directory')
        return manifest
    from huggingface_hub import HfApi
    api=HfApi()
    models=[]
    for spec in selected:
        info=api.model_info(spec['id'])
        if info.gated:raise RuntimeError('Selected checkpoint unexpectedly requires gated access: '+spec['id'])
        frozen={**spec,'revision':info.sha}
        if spec.get('backbone'):
            frozen['backbone_revision']=api.model_info(spec['backbone']).sha
        models.append(frozen)
    manifest={'schema_version':1,'created_at':now(),**corpus_fingerprints(corpus),'total_documents':len(corpus['documents']),'total_blocks':len(corpus['blocks']),'models':models,'profile_requirements':PROFILE_REQUIREMENTS,'source':'Exact HF commit revisions resolved before inference. No private inference API.'}
    save_json(path,manifest)
    return manifest


def prepare_profile(profile, output, skip_install=False, no_isolation=False):
    if no_isolation:
        python=Path(sys.executable)
    else:
        directory=output/'environments'/profile
        python=directory/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        if not python.exists():
            # Colab's compatible torch/CUDA runtime is reused. Python packages
            # with conflicting APIs remain inside each named environment.
            # Colab's Debian Python can omit ensurepip. Use the known-working
            # base pip --python strategy instead of bootstrapping pip in venv.
            venv.EnvBuilder(with_pip=False,system_site_packages=True).create(directory)
    if not skip_install:
        command=[sys.executable,'-m','pip','--python',str(python),'install','--disable-pip-version-check',*PROFILE_REQUIREMENTS[profile]]
        subprocess.run(command,check=True)
    return python


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',default='artifacts/local/corpus.json');parser.add_argument('--output',default='artifacts/local/ner')
    parser.add_argument('--reference',default='data/ner-reference-sample.json');parser.add_argument('--models',default='all');parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch-size',type=int,default=16);parser.add_argument('--max-tokens',type=int,default=320);parser.add_argument('--overlap-tokens',type=int,default=48);parser.add_argument('--threshold',type=float,default=.5)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--langsmith',action='store_true');parser.add_argument('--skip-install',action='store_true');parser.add_argument('--no-isolation',action='store_true')
    parser.add_argument('--export',default='web/public/data/ner-results.json');parser.add_argument('--export-only',action='store_true')
    args=parser.parse_args()
    corpus_path=Path(args.corpus).resolve();output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    corpus=load_json(corpus_path)
    chosen=set(args.models.split(',')) if args.models!='all' else {s['id'] for s in MODEL_SPECS}
    unknown=chosen-{s['id'] for s in MODEL_SPECS}
    if unknown:raise ValueError('Unknown NER model IDs: '+','.join(sorted(unknown)))
    selected=[s for s in MODEL_SPECS if s['id'] in chosen]
    manifest=freeze_manifest(corpus,output,selected,args.resume or args.export_only)
    if args.export_only:
        bundle=export_bundle(corpus,manifest,output,Path(args.export));print(json.dumps({'status':bundle['completion_status'],'complete_models':bundle['complete_models'],'export':args.export}));return 0
    reference=Path(args.reference).resolve() if args.reference else None
    if reference and not reference.exists():raise FileNotFoundError('Reference sample is required at '+str(reference)+'; use --reference "" only when accuracy evaluation is deliberately omitted.')
    profile_python={};failures=[]
    for spec in manifest['models']:
        profile=spec['profile']
        try:
            if profile not in profile_python:profile_python[profile]=prepare_profile(profile,output,args.skip_install,args.no_isolation)
            command=[str(profile_python[profile]),str(ROOT/'scripts/ner_worker.py'),'--corpus',str(corpus_path),'--manifest',str(output/'manifest.json'),'--model',spec['id'],'--output',str(output),'--device',args.device,'--batch-size',str(args.batch_size),'--max-tokens',str(args.max_tokens),'--overlap-tokens',str(args.overlap_tokens),'--threshold',str(args.threshold)]
            if reference:command+=['--reference',str(reference)]
            if args.langsmith:command+=['--langsmith']
            print('Running local NER checkpoint:',spec['id'],'revision',spec['revision'],flush=True)
            result=subprocess.run(command,cwd=ROOT)
            if result.returncode:failures.append({'model_id':spec['id'],'exit_code':result.returncode})
        except Exception as error:
            failure={'model_id':spec['id'],'status':'failed','error':type(error).__name__+': '+str(error)[:1000],'total_documents':len(corpus['documents']),'total_blocks':len(corpus['blocks']),'processed_documents':0,'processed_blocks':0,'updated_at':now()}
            save_json(output/slug(spec['id'])/'status.json',failure);failures.append(failure)
        # Export only verified completed/partial records after each checkpoint so
        # an interrupted free Colab session still has an inspectable artifact.
        export_bundle(corpus,manifest,output,Path(args.export))
    bundle=export_bundle(corpus,manifest,output,Path(args.export))
    summary={'schema_version':1,'status':bundle['completion_status'],'complete_models':bundle['complete_models'],'selected_models':len(manifest['models']),'documents':len(corpus['documents']),'blocks':len(corpus['blocks']),'failures':failures,'export_path':str(Path(args.export).resolve()),'updated_at':now()}
    save_json(output/'summary.json',summary);print(json.dumps(summary),flush=True)
    return 0 if bundle['completion_status']=='complete' else 2


if __name__=='__main__':
    sys.exit(main())
