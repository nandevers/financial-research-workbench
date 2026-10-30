"""Create a secret-free Colab upload bundle and a matching notebook."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'artifacts/local'


def prepare():
    paths = [ROOT / 'pyproject.toml', ROOT / 'models.json', LOCAL / 'corpus.json']
    paths += list((ROOT / 'engine/src').rglob('*.py'))
    paths += list((ROOT / 'scripts').glob('*.py'))
    paths += list((ROOT / 'evals').glob('*.json'))
    bundle = LOCAL / 'workbench-colab.zip'
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, 'workbench/' + path.relative_to(ROOT).as_posix())
    def cell(kind, source):
        result = dict(cell_type=kind, metadata={}, source=source.splitlines(keepends=True))
        if kind == 'code':
            result.update(execution_count=None, outputs=[])
        return result
    cells = [
        cell('markdown', '# Financial Research Workbench — five downloaded open models\n\nUse a free T4 GPU. This notebook downloads project code from GitHub and original reports from the issuers; no browser file upload is required. Full reports stay in your runtime. No private inference API is used. LangSmith uses its free Developer plan and a separate observability key in Colab Secrets. Start with the two-question smoke set before the full showcase. Failures remain recorded failures.\n'),
        cell('code', "!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader\nfrom pathlib import Path\nimport os, sys, subprocess, venv, json, zipfile\nREPOSITORY = 'https://github.com/nandolab/financial-research-workbench.git'\nROOT = Path('/content/financial-research-workbench')\nif not ROOT.exists():\n    subprocess.run(['git','clone','--depth','1',REPOSITORY,str(ROOT)],check=True)\nos.chdir(ROOT)\nprint('Project commit:',subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())\n"),
        cell('code', "# Isolate model libraries; the notebook kernel never imports Torch.\nenv = ROOT / 'artifacts/local/env-5.13.1'\nvenv.EnvBuilder(with_pip=False,system_site_packages=True).create(env)\nPYTHON = str(env/'bin/python')\nsubprocess.run([sys.executable,'-m','pip','--python',PYTHON,'install','-q','-e','.[gpu]','transformers==5.13.1'],check=True)\n# Fetch authentic issuer reports directly. Raw files are never committed.\nsubprocess.run([PYTHON,'-m','workbench.cli','corpus'],check=True)\n"),
        cell('code', "# Store LANGSMITH_API_KEY in Colab Secrets and enable notebook access.\n# Never paste a key into notebook source, output, or git.\nUSE_LANGSMITH = True\nif USE_LANGSMITH:\n    from google.colab import userdata\n    os.environ['LANGSMITH_API_KEY'] = userdata.get('LANGSMITH_API_KEY')\n    os.environ['LANGSMITH_TRACING'] = 'true'\n    os.environ['LANGSMITH_PROJECT'] = 'financial-research-workbench'\nelse:\n    os.environ['LANGSMITH_TRACING'] = 'false'\n"),
        cell('code', "CASE_SET = 'smoke'  # Banking and insurance, both cross-period comparisons.\nLIMIT = 2  # For the full showcase use CASE_SET='showcase' and LIMIT=12.\nMODEL = None  # Set one exact ID from models.json for an independent job.\ncommand = [sys.executable,'scripts/colab_run.py','--limit',str(LIMIT),'--case-set',CASE_SET,'--release-cache']\nif MODEL: command += ['--model',MODEL]\nif USE_LANGSMITH: command += ['--langsmith']\nsubprocess.run(command,check=True)\n"),
        cell('code', "from google.colab import files\nfor p in sorted(Path('artifacts/local/models').glob('*-status.json')):\n    status=json.loads(p.read_text())\n    print(status['model_id'],'completed:',status['completed'],'failures:',len(status['failures']))\nwith zipfile.ZipFile('/content/workbench-results.zip','w',zipfile.ZIP_DEFLATED) as z:\n    for folder in ('runs','models'):\n        for p in Path('artifacts/local',folder).glob('*.json'):\n            z.write(p,p.as_posix())\nfiles.download('/content/workbench-results.zip')\n"),
    ]
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', name='python3'), accelerator='GPU', colab=dict(name='Financial Research Workbench.ipynb')), nbformat=4, nbformat_minor=5)
    destination = ROOT / 'notebooks/benchmark.ipynb'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(notebook, indent=2))
    public = ROOT / 'web/public/data/benchmark.ipynb'
    public.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(destination, public)
    print(json.dumps({'bundle': str(bundle), 'bytes': bundle.stat().st_size, 'sha256': hashlib.sha256(bundle.read_bytes()).hexdigest(), 'notebook': str(destination)}, indent=2))


if __name__ == '__main__':
    prepare()
