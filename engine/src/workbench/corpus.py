from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[3]
LOCAL = ROOT / 'artifacts' / 'local'
PUBLIC = ROOT / 'web' / 'public'


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def clean(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def passages(text: str, limit=1500):
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z(])', clean(text))
    current = ''
    for sentence in sentences:
        if len(current) + len(sentence) > limit and current:
            yield current
            current = ''
        if len(sentence) > limit * 2:
            for start in range(0, len(sentence), limit):
                yield sentence[start:start + limit]
        else:
            current += (' ' if current else '') + sentence
    if len(current) > 70:
        yield current


def parse_html(raw: bytes):
    soup = BeautifulSoup(raw, 'lxml')
    for tag in soup(['script', 'style', 'noscript', 'nav', 'header', 'footer', 'ix:header']):
        tag.decompose()
    # Only leaf-sized structural blocks; avoid collecting every ancestor's text.
    section = 'Financial report'
    for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'div', 'table']):
        if element.find_parent('table'):
            continue
        # Inline-XBRL filings often use div/span paragraphs, with almost no p
        # elements. Collect leaf block containers, never their whole ancestors.
        if element.name == 'div' and element.find(['p', 'div', 'table', 'h1', 'h2', 'h3', 'h4']):
            continue
        text = clean(element.get_text(' ', strip=True))
        if element.name.startswith('h'):
            section = text[:180] or section
            continue
        if len(text) < 90:
            continue
        if element.name == 'table':
            rows = [[clean(cell.get_text(' ', strip=True)) for cell in row.find_all(['th', 'td'], recursive=False)] for row in element.find_all('tr')]
            rows = [row for row in rows if row]
            if not rows or len(text) > 16000:
                # Layout tables often contain the whole filing. Their nested p elements
                # were skipped above: extract text in bounded blocks instead.
                for part in passages(text):
                    yield {'text': part, 'section': section, 'kind': 'text', 'anchor': element.get('id')}
            else:
                yield {'text': text, 'section': section, 'kind': 'table', 'rows': rows, 'anchor': element.get('id')}
        else:
            for part in passages(text):
                yield {'text': part, 'section': section, 'kind': 'text', 'anchor': element.get('id')}


def parse_pdf(path: Path):
    reader = PdfReader(path)
    for page_number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ''
        # Page provenance is exact. PDF text flow is approximate and visibly labeled.
        for part in passages(text):
            yield {'text': part, 'section': f'Annual report · page {page_number}', 'kind': 'pdf_text', 'page': page_number}


def fetch_and_parse(limit: int | None = None):
    manifest = load_json(ROOT / 'corpus/manifests/filings.json')
    raw_dir = LOCAL / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    documents, blocks, errors = [], [], []
    previous = load_json(LOCAL / 'corpus.json') if (LOCAL / 'corpus.json').exists() else {'documents': [], 'blocks': []}
    previous_docs = {doc['id']: doc for doc in previous['documents']}
    for source in manifest['documents'][:limit]:
        path = raw_dir / (source['id'] + ('.pdf' if source['format'] == 'pdf' else '.html'))
        try:
            if not path.exists():
                response = requests.get(source['url'], headers={'User-Agent': 'Fernando Barbosa Financial Research Workbench (github.com/nandevers)'}, timeout=100)
                response.raise_for_status()
                if source['format'] == 'pdf' and not response.content.startswith(b'%PDF'):
                    raise ValueError('Response is not a PDF')
                path.write_bytes(response.content)
                time.sleep(.4)
            raw = path.read_bytes()
            raw_hash = hashlib.sha256(raw).hexdigest()
            old = previous_docs.get(source['id'])
            parser_version = '1.1.0' if source['format'] == 'html' else '1.0.0'
            if old and old.get('sha256') == raw_hash and old.get('parser_version') == parser_version:
                accepted = [block for block in previous['blocks'] if block['document_id'] == source['id']]
                documents.append(old)
                blocks.extend(accepted)
                print(f"Reused parsed {source['id']}: {len(accepted)} blocks", flush=True)
                continue
            parsed = list(parse_pdf(path) if source['format'] == 'pdf' else parse_html(raw))
            seen, accepted = set(), []
            for block in parsed:
                fingerprint = hashlib.sha256(block['text'].encode()).hexdigest()
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                block.update(id=f"{source['id']}-{fingerprint[:12]}", document_id=source['id'], ticker=source['ticker'], fiscal_period=source['fiscal_period'], filing_date=source.get('filing_date'), source_url=source['url'])
                accepted.append(block)
            if len(accepted) < 5:
                raise ValueError('Too little report text extracted; inspect the parser/source')
            doc = {**source, 'sha256': raw_hash, 'bytes': len(raw), 'blocks': len(accepted), 'parser_version': parser_version, 'retrieved_at': old['retrieved_at'] if old and old.get('sha256') == raw_hash else datetime.now(timezone.utc).isoformat(), 'parsed_at': datetime.now(timezone.utc).isoformat()}
            documents.append(doc)
            blocks.extend(accepted)
            print(f"Parsed {source['id']}: {len(accepted)} blocks", flush=True)
        except Exception as error:
            errors.append({'document_id': source['id'], 'error': type(error).__name__, 'message': str(error)[:200]})
            print(f"Could not parse {source['id']}: {type(error).__name__}", flush=True)
        # Save incremental progress so a slow PDF or interrupted download is resumable.
        write_json(LOCAL / 'corpus.json', {'documents': documents, 'blocks': blocks, 'errors': errors, 'created_at': datetime.now(timezone.utc).isoformat()})
    return documents, blocks
