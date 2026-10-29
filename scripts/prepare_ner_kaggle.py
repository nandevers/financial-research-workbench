"""Prepare a private, fifth-only Kaggle notebook. Never launches inference."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'artifacts/local'
SOURCE_SHA = 'd258d8963078e08f544f70aeecc2d8a0d787e366e90fcf157325fd2ea7c4c4e5'
CHECKPOINT_SHA = '1cfdddf37dfcc382c702c0e104547d6ca4e4570f51876ae8c67ca535b7662946'
HELPER_SHA = 'cd2ee45a7f5acde915c2495e2e2709b2cac427949cf58aa60cd46eb59cede130'


def cell(kind, text):
    item = {'cell_type': kind, 'metadata': {}, 'source': text.splitlines(keepends=True)}
    if kind == 'code':
        item.update(outputs=[], execution_count=None)
    return item


def prepare():
    output = LOCAL / 'ner-kaggle-job'
    inputs = output / 'private-inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    source = LOCAL / 'workbench-ner-colab.zip'
    checkpoint = LOCAL / 'ner-downloads/workbench-ner-checkpoint-04.zip'
    helper = ROOT / 'scripts/ner_recover_gliner2.py'
    for path, expected in ((source, SOURCE_SHA), (checkpoint, CHECKPOINT_SHA), (helper, HELPER_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen input identity differs: ' + path.name)
    hashes = {}
    with zipfile.ZipFile(checkpoint) as z:
        if z.testzip() is not None:
            raise ValueError('Checkpoint CRC failed')
        for name in z.namelist():
            if len(Path(name).parts) == 5 and name.startswith('artifacts/local/ner/'):
                hashes[name] = hashlib.sha256(z.read(name)).hexdigest()
    if len(hashes) != 36:
        raise ValueError('Expected all 36 first-four model files')
    with zipfile.ZipFile(source) as z:
        if z.testzip() is not None:
            raise ValueError('Source CRC failed')
    # Opaque suffixes prevent Kaggle's upload ingestion from unpacking our frozen
    # ZIPs. Their exact original bytes, SHA256 and CRC remain independently checked.
    shutil.copyfile(source, inputs / 'source-archive.bin')
    shutil.copyfile(checkpoint, inputs / 'checkpoint04-archive.bin')
    shutil.copyfile(helper, inputs / helper.name)
    (inputs / 'first-four-sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    intro = '''# Private GLiNER2 continuation — no other model inference

Keep this notebook and its input dataset **private**. Use only Kaggle's free GPU;
do not enable Google Cloud or paid resources. Inspect account quota and session
state first. Never use Run All. This notebook runs only fastino/gliner2-base-v1
at the original revision and settings over 30 reports / 26,058 blocks.

The first four model outputs are restored and protected by 36 SHA256 checks.
The raw fifth archive must be downloaded and verified locally before any recovery
or replay. LangSmith receives no corpus inference traces or secrets here.
'''
    inspect = '''import os, sys, json, hashlib, zipfile, subprocess, shutil
from pathlib import Path, PurePosixPath
from IPython.display import display, FileLink
ROOT=Path('/kaggle/working/financial-ner-workbench')
OUT=ROOT/'artifacts/local/ner'
MODEL='fastino/gliner2-base-v1'
SLUG='fastino--gliner2-base-v1'
print('ROOT_EXISTS',ROOT.exists())
print('GPU',subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout)
print('NER_PROCESSES',subprocess.run(['pgrep','-af','ner_worker.py|ner_recover_gliner2.py|ner_colab_run.py'],capture_output=True,text=True).stdout)
print('INPUT_FILES',[(str(p),p.stat().st_size) for p in Path('/kaggle/input').rglob('*') if p.is_file() and p.stat().st_size<50000000])
if OUT.exists():
    for p in sorted(OUT.glob('*/status.json')):print(p.parent.name,json.loads(p.read_text()))
print('SAVED_ARCHIVES',[str(p) for p in Path('/kaggle/working').glob('workbench-ner-*.zip')])
'''
    restore = f'''# Execute only after the read-only inspection shows no existing NER process.
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def unique_input(name):
    matches=list(Path('/kaggle/input').rglob(name))
    if len(matches)!=1:raise RuntimeError('Expected exactly one private input: '+name)
    return matches[0]
def extract_checked(path,expected):
    assert sha(path)==expected,'Input SHA256 differs'
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None,'Input CRC failure'
        for n in z.namelist():
            p=PurePosixPath(n)
            assert not p.is_absolute() and '..' not in p.parts and '\\\\' not in n,'Unsafe archive path'
        z.extractall(ROOT)
if not (OUT/'manifest.json').exists():
    ROOT.mkdir(parents=True,exist_ok=True)
    extract_checked(unique_input('source-archive.bin'),'{SOURCE_SHA}')
    extract_checked(unique_input('checkpoint04-archive.bin'),'{CHECKPOINT_SHA}')
EXPECTED_FIRST_FOUR=json.loads(unique_input('first-four-sha256.json').read_text())
def verify_first_four():
    assert len(EXPECTED_FIRST_FOUR)==36
    assert all(sha(ROOT/name)==expected for name,expected in EXPECTED_FIRST_FOUR.items()),'Completed outputs changed'
verify_first_four()
helper=unique_input('ner_recover_gliner2.py');assert sha(helper)=='{HELPER_SHA}'
shutil.copyfile(helper,ROOT/'scripts/ner_recover_gliner2.py')
os.chdir(ROOT)
corpus=json.loads((ROOT/'artifacts/local/corpus.json').read_text())
assert (len(corpus['documents']),len(corpus['blocks']))==(30,26058)
manifest=json.loads((OUT/'manifest.json').read_text())
assert next(s for s in manifest['models'] if s['id']==MODEL)['revision']=='f9634218e53580c56edf0de97ca1a7d3f1c2354e'
del corpus
print('RESTORED: frozen inputs,30reports/26058blocks,36unchanged first-four files')
'''
    environment = '''# Reuse Kaggle's CUDA-compatible Torch; isolate only the GLiNER2 profile.
# Actual Python, Torch, CUDA and GPU provenance is recorded by the frozen worker.
sys.path.insert(0,str(ROOT/'scripts'))
from ner_colab_run import prepare_profile
PYTHON=prepare_profile('gliner2',OUT)
subprocess.run([str(PYTHON),'-c',"import torch; assert torch.cuda.is_available(); print('TORCH',torch.__version__,'GPU',torch.cuda.get_device_name(0))"],check=True)
'''
    run = '''# ONLY GLiNER2. Do not use the five-model orchestrator.
def package(stage):
    verify_first_four()
    target=Path('/kaggle/working')/('workbench-ner-kaggle-'+stage+'.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            relative=p.relative_to(ROOT)
            if p.is_file() and not {'environments','pinned-load'}.intersection(relative.parts):z.write(p,relative.as_posix())
        for name in ('scripts/ner_worker.py','engine/src/workbench/ner.py','scripts/ner_recover_gliner2.py','data/ner-reference-sample.json'):z.write(ROOT/name,name)
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    receipt={'file':str(target),'bytes':target.stat().st_size,'sha256':sha(target),'first_four_unchanged':True}
    target.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\\n')
    print(json.dumps(receipt));display(FileLink(str(target)))
    return receipt
lock=Path('/kaggle/working/gliner2-raw-started.json')
assert not lock.exists(),'Already launched: inspect actual process/status/backup before any resume'
command=[str(PYTHON),'-u','scripts/ner_worker.py','--corpus','artifacts/local/corpus.json','--manifest','artifacts/local/ner/manifest.json','--model',MODEL,'--output','artifacts/local/ner','--device','cuda','--batch-size','16','--max-tokens','320','--overlap-tokens','48','--threshold','0.5','--reference','data/ner-reference-sample.json']
lock.write_text(json.dumps({'command':command}))
try:raw_exit_code=subprocess.run(command,cwd=ROOT).returncode
finally:RAW_RECEIPT=package('raw')
print('WORKER_EXIT',raw_exit_code)
print('STOP: download and locally verify raw archive SHA256/CRC/content before recovery.')
'''
    recovery = '''# Run only after the raw archive was saved and verified on the user's computer.
VERIFIED_LOCAL_RAW_SHA256=''
assert VERIFIED_LOCAL_RAW_SHA256 and VERIFIED_LOCAL_RAW_SHA256==RAW_RECEIPT['sha256'],'Local raw backup not yet verified'
verify_first_four()
status=json.loads((OUT/SLUG/'status.json').read_text())
subprocess.run([str(PYTHON),'scripts/ner_recover_gliner2.py'],cwd=ROOT,check=True)
if status.get('failed_blocks',0):
    subprocess.run([str(PYTHON),'-u','scripts/ner_recover_gliner2.py','--execute'],cwd=ROOT,check=True)
RECOVERED_RECEIPT=package('recovered')
print('STOP: immediately download and locally verify the recovered archive before replay.')
'''
    notebook = {'cells': [cell('markdown',intro),cell('code',inspect),cell('code',restore),cell('code',environment),cell('code',run),cell('markdown','Download and verify the raw archive locally before continuing. Never Run All.\n'),cell('code',recovery)], 'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    destination = output / 'workbench-gliner2-only.ipynb'
    destination.write_text(json.dumps(notebook,indent=2)+'\n',encoding='utf-8')
    bundle = output / 'workbench-ner-private-inputs.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for p in inputs.iterdir():z.write(p,p.name)
    print(json.dumps({'notebook':str(destination),'private_inputs':str(bundle),'bytes':bundle.stat().st_size,'sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),'first_four_files':len(hashes),'remote_execution_started':False},indent=2))


if __name__ == '__main__':
    prepare()
