"""Run one open model per GPU process. Supports independent/restartable Colab jobs."""
import argparse
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'engine/src'))
from workbench.gpu import run_benchmark

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--limit', type=int)
parser.add_argument('--langsmith', action='store_true')
parser.add_argument('--revision')
parser.add_argument('--case-set', choices=['showcase', 'smoke'], default='showcase')
args = parser.parse_args()
try:
    run_benchmark(args.model, args.limit, args.langsmith, args.revision, args.case_set)
except Exception as error:
    from workbench.corpus import LOCAL, write_json
    write_json(LOCAL / 'models' / (args.model.replace('/', '--') + '-status.json'), {'model_id': args.model, 'completed': 0, 'failures': [{'phase': 'initialization_or_experiment', 'error': type(error).__name__, 'message': str(error)[:400]}]})
    raise
