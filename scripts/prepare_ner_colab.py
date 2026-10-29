"""Prepare the full-corpus NER/retrieval Colab job. No inference or API calls."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'artifacts/local'

def prepare():
    required=[LOCAL/'corpus.json',ROOT/'data/ner-reference-sample.json',ROOT/'engine/src/workbench/ner.py',ROOT/'scripts/ner_worker.py',ROOT/'scripts/ner_colab_run.py']
    absent=[str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if absent:
        raise RuntimeError('Finish these job sources before transfer: '+', '.join(absent))
    paths=[ROOT/'pyproject.toml',LOCAL/'corpus.json',ROOT/'web/src/data/ner-models.json',ROOT/'data/ner-reference-sample.json']
    paths+=list((ROOT/'engine/src/workbench').glob('*.py'))
    paths += [ROOT/'scripts/ner_worker.py',ROOT/'scripts/ner_colab_run.py']
    bundle=LOCAL/'workbench-ner-colab.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,p.relative_to(ROOT).as_posix())
    bundle_sha=hashlib.sha256(bundle.read_bytes()).hexdigest()
    def cell(kind,source):
        result={'cell_type':kind,'id':hashlib.sha1(source.encode()).hexdigest()[:12],'metadata':{},'source':source.splitlines(keepends=True)}
        if kind=='code':result.update(execution_count=None,outputs=[])
        return result
    cells=[
      cell('markdown', '# Financial reports: full-corpus named entities and actual retrieval\n\nProcess all 30 original reports / 26,058 source blocks with five downloadable open entity models. Use a free Colab T4 GPU. Predictions build actual per-model entity indexes and entity-assisted passage searches, compared with the original word-matching search. No private inference API is used.\n\nAccuracy is scored only on the explicitly identified, independently reviewed reference sample; full-corpus processing coverage is reported separately. Agent-authored reference annotations are not a human financial gold standard. LangSmith records the small evaluation/retrieval series, not every full-corpus block, to limit hosted trace volume. Full report text and raw predictions remain in this private runtime and the local result download.\n'),
      cell('code', "# Runtime > Change runtime type > T4 GPU (free). No paid runtime is required.\nimport os, sys, json, hashlib, subprocess, zipfile\nfrom pathlib import Path\nsubprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],check=True)\nROOT=Path('/content/financial-ner-workbench')\nROOT.mkdir(parents=True,exist_ok=True)\n"),
      cell('code', f"# Upload the matching archive via the Files sidebar, or use the upload prompt below. Add a saved checkpoint when resuming.\nfrom google.colab import files\nEXPECTED_ARCHIVE_SHA256='{bundle_sha}'\npreuploaded=Path('/content/workbench-ner-colab.zip')\nuploaded={str(p):None for p in [preuploaded,*Path('/content').glob('workbench-ner-checkpoint-*.zip')]} if preuploaded.exists() else files.upload()\narchive=next((Path(name) for name in uploaded if name.endswith('.zip') and hashlib.sha256(Path(name).read_bytes()).hexdigest()==EXPECTED_ARCHIVE_SHA256),None)\nif archive is None: raise RuntimeError('Upload the matching workbench-ner-colab.zip; its fingerprint must match this notebook.')\ndef extract_checked(path):\n    with zipfile.ZipFile(path) as z:\n        for item in z.infolist():\n            resolved=(ROOT/item.filename).resolve()\n            if ROOT.resolve() not in resolved.parents: raise RuntimeError('Unexpected archive path')\n        z.extractall(ROOT)\nextract_checked(archive)\nfor name in uploaded:\n    checkpoint=Path(name)\n    if checkpoint!=archive and checkpoint.name.startswith('workbench-ner-checkpoint-') and checkpoint.suffix=='.zip': extract_checked(checkpoint)\nos.chdir(ROOT)\ncorpus=json.loads((ROOT/'artifacts/local/corpus.json').read_text())\nassert len(corpus['documents'])==30 and len(corpus['blocks'])==26058\nprint('Frozen input:',len(corpus['documents']),'reports /',len(corpus['blocks']),'source blocks')\ndel uploaded,corpus\n"),
      cell('code', "# Model-specific environments are isolated by the runner.\nsubprocess.run([sys.executable,'-m','pip','install','-q','huggingface-hub>=0.34','langsmith>=0.14,<1','scikit-learn>=1.5'],check=True)\n"),
      cell('code', "# LangSmith is observability only. Model inference uses downloaded weights.\n# Keep the key in Colab Secrets; never paste it into source or output.\nUSE_LANGSMITH=True\nif USE_LANGSMITH:\n    from google.colab import userdata\n    os.environ['LANGSMITH_API_KEY']=userdata.get('LANGSMITH_API_KEY')\n    os.environ['LANGSMITH_TRACING']='false' # full-corpus workers do not emit one hosted trace per block\n    os.environ['LANGSMITH_PROJECT']='financial-research-workbench-ner'\n"),
      cell('code', "# Full corpus: no document/block limit. Model files/libraries are excluded from downloads.\nOUT=ROOT/'artifacts/local/ner'\nARTIFACT_NAMES={'manifest.json','summary.json','metadata.json','status.json','predictions.jsonl','entity-index.json','retrieval.json','reference-predictions.jsonl','query-predictions.jsonl','evaluation.json','langsmith.json'}\ndef checkpoint_zip(name):\n    path=Path('/content')/name\n    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:\n        for p in OUT.rglob('*'):\n            if p.is_file() and p.name in ARTIFACT_NAMES and 'environments' not in p.parts and 'pinned-load' not in p.parts: z.write(p,p.relative_to(ROOT).as_posix())\n        public=ROOT/'web/public/data/ner-results.json'\n        if public.exists(): z.write(public,public.relative_to(ROOT).as_posix())\n    return path\ncommand=[sys.executable,'-u','scripts/ner_colab_run.py','--corpus','artifacts/local/corpus.json','--output','artifacts/local/ner','--models','all','--device','cuda','--batch-size','16','--reference','data/ner-reference-sample.json','--resume']\nif USE_LANGSMITH: command+=['--langsmith']\nprocess=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)\nfinished=set()\nfor line in process.stdout:\n    print(line,end='',flush=True)\n    try: event=json.loads(line)\n    except (ValueError,TypeError): continue\n    if event.get('status')=='complete' and event.get('model_id') and event['model_id'] not in finished:\n        finished.add(event['model_id'])\n        files.download(str(checkpoint_zip('workbench-ner-checkpoint-'+str(len(finished)).zfill(2)+'.zip')))\nexit_code=process.wait()\nif exit_code: raise RuntimeError('One or more models did not complete. Inspect saved status and fix before publication.')\n"),
      cell('code', "# Download the final actual artifacts. No placeholder scores are filled in.\nfor p in sorted(OUT.glob('*/status.json')): print(p.parent.name,json.loads(p.read_text()))\nfiles.download(str(checkpoint_zip('workbench-ner-results.zip')))\n"),
    ]
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','name':'python3'},'accelerator':'GPU','colab':{'name':'Financial reports - full NER and retrieval.ipynb'}},'nbformat':4,'nbformat_minor':5}
    p=ROOT/'notebooks/ner_retrieval.ipynb';p.write_text(json.dumps(notebook,indent=2)+'\n',encoding='utf-8')
    shutil.copyfile(p,ROOT/'web/public/data/ner_retrieval.ipynb')
    print(json.dumps({'notebook':str(p),'bundle':str(bundle),'bundle_bytes':bundle.stat().st_size,'bundle_sha256':bundle_sha},indent=2))

if __name__=='__main__':prepare()
