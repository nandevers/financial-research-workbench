"""Verify and preserve a private raw/recovered checkpoint before postprocessing.

This is an immediate backup gate, not a replacement for strict final import.
Raw first passes may contain honestly recorded failed blocks.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'engine/src'))
from workbench.ner import corpus_fingerprints, digest


def verify(archive, expected_sha, expected_bytes, recovered=False):
    archive = Path(archive).resolve()
    actual_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual_sha != expected_sha or archive.stat().st_size != expected_bytes:
        raise ValueError('Downloaded bytes/SHA256 differ from runtime receipt')
    source = ROOT / 'artifacts/local/workbench-ner-colab.zip'
    previous = ROOT / 'artifacts/local/ner-downloads/workbench-ner-checkpoint-04.zip'
    if hashlib.sha256(source.read_bytes()).hexdigest() != 'd258d8963078e08f544f70aeecc2d8a0d787e366e90fcf157325fd2ea7c4c4e5':
        raise ValueError('Frozen source identity changed')
    if hashlib.sha256(previous.read_bytes()).hexdigest() != '1cfdddf37dfcc382c702c0e104547d6ca4e4570f51876ae8c67ca535b7662946':
        raise ValueError('Completed checkpoint04 identity changed')
    prefix = 'artifacts/local/ner/fastino--gliner2-base-v1/'
    with zipfile.ZipFile(source) as src, zipfile.ZipFile(previous) as old, zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('Downloaded ZIP CRC failure')
        for n in z.namelist():
            p = Path(n)
            if p.is_absolute() or '..' in p.parts or '\\' in n:
                raise ValueError('Unsafe archive member')
        preserved = [n for n in old.namelist() if n.startswith('artifacts/local/ner/') and len(Path(n).parts) == 5]
        if len(preserved) != 36 or any(z.read(n) != old.read(n) for n in preserved):
            raise ValueError('The four completed models changed')
        if z.read('artifacts/local/ner/manifest.json') != old.read('artifacts/local/ner/manifest.json'):
            raise ValueError('Original manifest changed')
        for n in ('scripts/ner_worker.py', 'engine/src/workbench/ner.py', 'data/ner-reference-sample.json'):
            if z.read(n) != src.read(n):
                raise ValueError('Frozen source/reference changed: ' + n)
        corpus = json.loads(src.read('artifacts/local/corpus.json'))
        metadata = json.loads(z.read(prefix + 'metadata.json'))
        if metadata['model_id'] != 'fastino/gliner2-base-v1' or metadata['revision'] != 'f9634218e53580c56edf0de97ca1a7d3f1c2354e':
            raise ValueError('Fifth checkpoint identity differs')
        if any(metadata.get(k) != v for k, v in corpus_fingerprints(corpus).items()):
            raise ValueError('Fifth corpus identity differs')
        rows = {}
        raw = z.read(prefix + 'predictions.jsonl')
        if not raw.endswith(b'\n'):
            raise ValueError('Prediction stream has an unfinished record')
        for line in raw.splitlines():
            row = json.loads(line)
            rows[row['id']] = row
        blocks = {b['id']: b for b in corpus['blocks']}
        if len(blocks) != 26058 or set(rows) != set(blocks):
            raise ValueError('Fifth pass does not cover every original block')
        for key, row in rows.items():
            block = blocks[key]
            if row['text_sha256'] != digest(block['text']) or row['document_id'] != block['document_id']:
                raise ValueError('Prediction source identity mismatch')
            if row['status'] not in ('complete', 'failed'):
                raise ValueError('Unexpected block completion state')
        counts = dict(Counter(row['status'] for row in rows.values()))
        if recovered and counts.get('failed', 0):
            raise ValueError('Recovered backup still contains failed blocks')
        status = json.loads(z.read(prefix + 'status.json'))
        if status.get('recovery'):
            required = ('recovery.json', 'recovery-native.jsonl', 'recovery-originals/predictions.jsonl', 'recovery-originals/source/ner_worker.py', 'recovery-originals/source/ner.py', 'recovery-sources/ner_recover_gliner2.py')
            for n in required:
                z.getinfo(prefix + n)
    target = ROOT / 'artifacts/local/ner-downloads' / archive.name
    if archive != target:
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != actual_sha:
            raise ValueError('Different backup already exists at destination')
        shutil.copy2(archive, target)
    if hashlib.sha256(target.read_bytes()).hexdigest() != actual_sha:
        raise ValueError('Preserved local copy differs')
    receipt = {'local_path': str(target), 'bytes': expected_bytes, 'sha256': actual_sha, 'crc_passed': True, 'first_four_unchanged_files': 36, 'fifth_blocks': len(rows), 'fifth_states': counts, 'recovery_identity': status.get('recovery', {}).get('recovery_sha256'), 'verified_at': datetime.now(timezone.utc).isoformat(), 'final_strict_validation_still_required': True}
    target.with_suffix('.verified.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--expected-bytes', type=int, required=True)
    parser.add_argument('--recovered', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.archive, args.expected_sha, args.expected_bytes, args.recovered), indent=2))
