"""Separate subprocesses isolate model-specific Transformers versions and GPU memory."""
import argparse
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=1, help='Tasks per model; each runs baseline and workflow. Use 12 for all showcase tasks.')
    parser.add_argument('--model', help='One exact model ID; omission runs all five.')
    parser.add_argument('--case-set', choices=['showcase', 'smoke'], default='showcase')
    parser.add_argument('--langsmith', action='store_true')
    parser.add_argument('--release-cache', action='store_true', help='Remove this job\'s dedicated downloaded-weight cache after each model to bound Colab disk use.')
    args = parser.parse_args()
    specs = json.loads((ROOT / 'models.json').read_text())['models']
    if args.model:
        specs = [s for s in specs if s['id'] == args.model]
        if not specs:
            parser.error('Model must be one of the five configured IDs')
    if args.langsmith and not os.environ.get('LANGSMITH_API_KEY'):
        parser.error('Set LANGSMITH_API_KEY in the environment first. No inference API key is used.')
    outcomes = []
    for spec in specs:
        profile = spec['transformers_profile']
        env_path = ROOT / 'artifacts' / 'local' / ('env-' + profile)
        interpreter = env_path / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        if not interpreter.exists():
            # Colab's Debian interpreter may not ship ensurepip. Base pip can
            # manage a target interpreter even when that venv has no pip module.
            venv.EnvBuilder(with_pip=False, system_site_packages=True).create(env_path)
        subprocess.run([sys.executable, '-m', 'pip', '--python', str(interpreter), 'install', '-q', '-e', str(ROOT) + '[gpu]', 'transformers==' + profile], check=True)
        command = [str(interpreter), str(ROOT / 'scripts/gpu_benchmark.py'), '--model', spec['id'], '--limit', str(args.limit), '--case-set', args.case_set]
        if args.langsmith:
            command.append('--langsmith')
        cache_base = (ROOT / 'artifacts/local/model-cache').resolve()
        cache = (cache_base / spec['id'].replace('/', '--')).resolve()
        environment = dict(os.environ, HF_HOME=str(cache))
        completed = subprocess.run(command, cwd=ROOT, env=environment)
        outcomes.append({'model_id': spec['id'], 'process_exit_code': completed.returncode})
        (ROOT / 'artifacts/local/batch-status.json').write_text(json.dumps(outcomes, indent=2))
        # A model process exiting releases GPU memory; an error does not erase prior runs.
        print(outcomes[-1], flush=True)
        if args.release_cache and cache.is_dir():
            if cache == cache_base or not cache.is_relative_to(cache_base):
                raise ValueError('Refusing to remove a cache outside this job\'s directory')
            shutil.rmtree(cache)
    if any(item['process_exit_code'] for item in outcomes):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
