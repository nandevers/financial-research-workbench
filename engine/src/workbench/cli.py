import argparse
import json
from datetime import datetime, timezone

from .corpus import ROOT, LOCAL, PUBLIC, load_json, write_json, fetch_and_parse


def select_comparison_cohort(runs):
    measured = [run for run in runs if run.get('mode') == 'model']
    versioned = [run for run in measured if run.get('hardware', {}).get('corpus_snapshot_sha256')]
    if not versioned:
        return measured, None
    newest = max(versioned, key=lambda run: run['recorded_at'])
    def key(run):
        hardware = run.get('hardware', {})
        return hardware.get('corpus_snapshot_sha256'), hardware.get('workflow_sha256')
    hardware = newest['hardware']
    return [run for run in measured if key(run) == key(newest)], {k: hardware.get(k) for k in ('corpus_snapshot_sha256', 'corpus_block_count', 'workflow_sha256')}


def make_cases():
    specs = [
        ('liquidity', 'Liquidity under pressure', 'What do the filings disclose about liquidity risk, funding sources, deposits and liquidity buffers?', ['liquidity risk funding deposits liquidity coverage ratio', 'liquidity sources cash stress capital resources']),
        ('credit', 'Credit quality in context', 'How do credit-loss allowances, charge-offs and management risk discussion compare across the selected periods?', ['allowance credit losses net charge-offs credit quality', 'provision loan credit risk delinquencies']),
        ('capital', 'Capital and resilience', 'What do the selected reports say about capital resources, regulatory constraints and capital management?', ['capital resources regulatory capital common equity tier', 'capital management stress tests financial condition']),
        ('catastrophe', 'Catastrophe exposure', 'How do the insurers describe catastrophe exposure, risk concentration and uncertainty?', ['catastrophe losses natural disasters risk concentration', 'catastrophe modeling weather uncertainty climate']),
        ('reinsurance', 'The reinsurance boundary', 'How do the insurers explain reinsurance arrangements and the risks those arrangements leave?', ['reinsurance recoverables reinsurers counterparty risk', 'reinsurance retention limits protection catastrophe']),
        ('cash', 'Cash flow and the narrative', 'What explanations and financial-statement notes support changes in operating cash flows?', ['cash flows operating activities changes cash', 'cash provided operations financial statements notes']),
    ]
    cases = []
    for theme, title, query, followups in specs:
        tickers = ['CB', 'TRV'] if theme in ('catastrophe', 'reinsurance') else ['BAC', 'JPM']
        for years in ([2025], [2024], [2023], [2024, 2025], [2023, 2024, 2025]):
            for scope in ('pair', 'single'):
                cases.append({'id': f"{theme}-{'-'.join(map(str, years))}-{scope}", 'title': title, 'theme': theme, 'query': query, 'tickers': tickers if scope == 'pair' else tickers[:1], 'years': years, 'followups': followups, 'as_of': None, 'split': 'test' if len(cases) % 3 == 2 else 'development', 'evaluation_kind': 'metadata_and_citation_checks_not_gold_entailment'})
    write_json(ROOT / 'evals/cases.json', cases)
    # The public selection includes narrative, cross-period and cross-institution cases.
    selected = [case for case in cases if (case['years'] == [2024, 2025] and case['id'].endswith('pair')) or (case['years'] == [2025] and case['id'].endswith('single'))]
    write_json(ROOT / 'evals/showcase.json', selected)
    return selected


def export_site():
    from .calculations import prepare_reference_calculations
    corpus = load_json(LOCAL / 'corpus.json')
    cases = load_json(ROOT / 'evals/showcase.json')
    runs = [load_json(path) for path in (LOCAL / 'runs').glob('*.json')]
    # Most recent reference/model runs retained per case/model/baseline.
    latest = {}
    for run in sorted(runs, key=lambda item: item['recorded_at']):
        latest[(run['case_id'], run.get('model_id'), run.get('baseline', False))] = run
    runs = list(latest.values())
    measured_runs = list(runs)
    withheld = []
    for run in list(runs):
        allowed = {item['id'] for item in run['evidence']}
        if any(item not in allowed for finding in run['output']['findings'] for item in finding['evidence_ids']):
            withheld.append({'run_id': run['id'], 'model_id': run.get('model_id'), 'case_id': run['case_id'], 'reason': 'Unresolved source references; memo withheld pending review'})
            runs.remove(run)
    documents = [{k: v for k, v in doc.items() if k not in ('publication_evidence',)} for doc in corpus['documents']]
    for run in runs:
        # Only bounded evidence excerpts are exported, not full reports.
        for evidence in run['evidence']:
            # Preserve exactly the bounded passage supplied to the generator.
            evidence['text'] = evidence['text'][:1250]
            evidence.pop('rows', None)
            doc = next(doc for doc in documents if doc['id'] == evidence['document_id'])
            evidence['issuer'] = doc['issuer']
            evidence['source_url'] = doc['url'] + (f"#page={evidence['page']}" if evidence.get('page') else '')
        allowed = {item['id'] for item in run['evidence']}
        for finding in run['output']['findings']:
            if any(item not in allowed for item in finding['evidence_ids']):
                raise ValueError('Refusing export with unresolved citation')
    model_config = load_json(ROOT / 'models.json')
    comparison_runs, cohort = select_comparison_cohort(measured_runs)
    by_model = []
    for model in model_config['models']:
        measured = [run for run in comparison_runs if run.get('model_id') == model['id'] and not run.get('baseline')]
        baselines = [run for run in comparison_runs if run.get('model_id') == model['id'] and run.get('baseline')]
        status_file = LOCAL / 'models' / (model['id'].replace('/', '--') + '-status.json')
        status = load_json(status_file) if status_file.exists() else {}
        status_metadata = status.get('metadata', {})
        matches_cohort = cohort is None or (status_metadata.get('corpus_snapshot_sha256') == cohort['corpus_snapshot_sha256'] and status_metadata.get('workflow_sha256') == cohort['workflow_sha256'])
        failures = status.get('failures', []) if matches_cohort else []
        validity = [run['verification']['citation_validity'] for run in measured if run['verification']['citation_validity'] is not None]
        generation = [run['generation'] for run in measured if run.get('generation')]
        generation += [failure['generation'] for failure in failures if failure.get('generation') and not failure.get('baseline')]
        failure_summaries = [{k: failure[k] for k in ('case_id', 'baseline', 'error', 'output_limit_reached') if k in failure} for failure in failures]
        by_model.append({**model, 'status': 'recorded' if measured else 'failed' if failures else 'not_run', 'runs': len(measured), 'baseline_runs': len(baselines), 'failed_attempts': len(failures), 'failures': failure_summaries, 'tokens_per_second': sum(g['tokens_per_second'] for g in generation) / len(generation) if generation else None, 'peak_gpu_memory_gb': max((g['peak_gpu_memory_gb'] for g in generation), default=None), 'mean_duration_ms': sum(run['duration_ms'] for run in measured) / len(measured) if measured else None, 'citation_validity': sum(validity) / len(validity) if validity else None, 'claim_entailment': None})
    snapshot = {'schema_version': '1.0', 'title': 'Financial Research Workbench', 'author': 'Fernando Barbosa', 'prepared_at': datetime.now(timezone.utc).isoformat(), 'documents': documents, 'cases': cases, 'runs': runs, 'models': by_model, 'corpus_errors': corpus.get('errors', []), 'limitations': ['Source passages are verbatim bounded excerpts; PDF reading order can differ from the original report.', 'A valid citation reference is not a verified claim-entailment score.', 'Model comparisons are shown only after actual GPU results have been imported.', 'Some issuer reports have no independently verified publication date; strict as-of queries exclude them.'], 'benchmark': {'cases': len(load_json(ROOT / 'evals/cases.json')), 'measured_model_runs': sum(run.get('mode') == 'model' for run in runs), 'gold_entailment_labels': 0}}
    snapshot['withheld_memos'] = withheld
    snapshot['comparison_cohort'] = cohort
    reviews_path = ROOT / 'evals/reviews.json'
    snapshot['reviews'] = load_json(reviews_path) if reviews_path.exists() else []
    snapshot['calculations'] = prepare_reference_calculations(corpus)
    snapshot['benchmark']['recorded_model_runs_total'] = len([run for run in measured_runs if run.get('mode') == 'model'])
    snapshot['benchmark']['measured_model_runs'] = len(comparison_runs)
    snapshot['benchmark']['measured_questions'] = len({run['case_id'] for run in comparison_runs})
    write_json(PUBLIC / 'data/workbench.json', snapshot)
    print(f"Exported {len(documents)} documents, {len(runs)} recorded runs; {snapshot['benchmark']['measured_model_runs']} LLM runs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['corpus', 'cases', 'reference', 'export'])
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.command == 'corpus':
        fetch_and_parse(args.limit)
    elif args.command == 'cases':
        make_cases()
    elif args.command == 'reference':
        from .retrieval import CorpusIndex
        from .research import run_case
        corpus = load_json(LOCAL / 'corpus.json')
        index = CorpusIndex(corpus['blocks'], corpus['documents'])
        for case in make_cases():
            result = run_case(index, case)
            print(f"Recorded source retrieval {case['id']}: {len(result['evidence'])} passages", flush=True)
    elif args.command == 'export':
        export_site()


if __name__ == '__main__':
    main()
