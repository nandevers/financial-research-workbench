"""Validate a Colab results archive before copying records into the local engine."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'engine/src'))
from workbench.contracts import ResearchOutput
from workbench.corpus import load_json, write_json


def import_results(path):
    local = ROOT / 'artifacts/local'
    models = {m['id'] for m in load_json(ROOT / 'models.json')['models']}
    cases = {c['id'] for c in load_json(ROOT / 'evals/cases.json')}
    snapshots = [load_json(local / 'corpus.json')]
    snapshots += [load_json(p) for p in local.glob('corpus-snapshot-*.json')]
    block_maps = [{b['id']: b for b in snapshot['blocks']} for snapshot in snapshots]
    validated = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            parts = Path(info.filename).parts
            if len(parts) != 4 or parts[:2] != ('artifacts', 'local') or parts[2] not in ('runs', 'models') or not parts[3].endswith('.json'):
                raise ValueError('Unexpected results archive member: ' + info.filename)
            if info.file_size > 2_000_000:
                raise ValueError('Oversized result record')
            data = json.loads(archive.read(info))
            if data.get('model_id') not in models:
                raise ValueError('Unknown model in results')
            if parts[2] == 'runs':
                if data.get('case_id') not in cases or data.get('mode') != 'model':
                    raise ValueError('Unknown case or non-model result')
                ResearchOutput.model_validate(data['output'])
                if not any(all(e['id'] in blocks and e['text'] == blocks[e['id']]['text'] for e in data['evidence']) for blocks in block_maps):
                    raise ValueError('Result does not match a preserved local source snapshot')
                # Invalid references are a measured failure, not grounds to hide an
                # entire experiment. The exporter withholds that memo from the site.
            validated.append((local / parts[2] / parts[3], data))
    # Complete validation before making any writes.
    for destination, data in validated:
        if destination.exists() and destination.parent.name == 'models':
            previous = destination.read_bytes()
            digest = hashlib.sha256(previous).hexdigest()[:16]
            write_json(local / 'status-history' / f'{digest}-{destination.name}', json.loads(previous))
        write_json(destination, data)
    print(f'Imported {len(validated)} validated records. Run workbench export, then npm run build.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    import_results(parser.parse_args().archive)
