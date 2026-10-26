import json
from decimal import Decimal

import pytest

from workbench.contracts import Calculation
from workbench.research import build_graph, parse_response
from workbench.retrieval import CorpusIndex


@pytest.fixture
def index():
    blocks = [
        dict(id='old', document_id='d1', ticker='BAC', fiscal_period='2024-12-31', filing_date='2025-02-20', text='Liquidity deposits cash funding capital buffers'),
        dict(id='future', document_id='d2', ticker='BAC', fiscal_period='2025-12-31', filing_date='2026-02-20', text='Liquidity deposits cash funding capital buffers'),
        dict(id='unknown', document_id='d3', ticker='JPM', fiscal_period='2024-12-31', filing_date=None, text='Liquidity deposits cash funding capital buffers'),
    ]
    return CorpusIndex(blocks, [{'id': b['document_id']} for b in blocks])


def test_as_of_excludes_future_and_unknown_publication(index):
    assert [b['id'] for b in index.search('liquidity', as_of='2025-12-31')] == ['old']


def test_fiscal_year_does_not_substitute_for_publication_date(index):
    assert index.search('liquidity', years=[2025], as_of='2025-12-31') == []
    assert index.tool().invoke(dict(title='Test investigation', query='liquidity', tickers=['BAC'], years=[2025]))[0]['id'] == 'future'


def test_invalid_model_citations_require_review(index):
    response = dict(headline='Review', summary='Research', findings=[dict(claim='A claim', evidence_ids=['invented-id'])], limitations=[])
    case = dict(title='Test investigation', query='liquidity', tickers=['BAC'], years=[2024], followups=['cash funding'])
    result = build_graph(index, lambda _: json.dumps(response)).compile().invoke(dict(case=case, events=[]))
    assert result['status'] == 'needs_review'
    assert result['verification']['citation_validity'] == 0
    assert result['verification']['claim_entailment'] is None


def test_reference_run_does_not_claim_ai_or_quality(index):
    case = dict(title='Test investigation', query='liquidity', tickers=['BAC'], years=[2024], followups=['cash funding'])
    result = build_graph(index).compile().invoke(dict(case=case, events=[]))
    assert result['mode'] == 'reference'
    assert result['output']['findings'] == []
    assert result['verification']['citation_validity'] is None


def test_missing_period_triggers_bounded_followup():
    blocks = [dict(id=f'new{i}', document_id=f'n{i}', ticker='BAC', fiscal_period='2025-12-31', filing_date='2026-02-20', text='liquidity deposits') for i in range(7)]
    blocks.append(dict(id='old', document_id='old', ticker='BAC', fiscal_period='2024-12-31', filing_date='2025-02-20', text='liquidity capital buffer cash treasury funding uncertainty'))
    index = CorpusIndex(blocks, [{'id': b['document_id']} for b in blocks])
    case = dict(title='Test investigation', query='liquidity', tickers=['BAC'], years=[2024, 2025], followups=['liquidity'])
    result = build_graph(index).compile().invoke(dict(case=case, events=[]))
    assert 'old' in {e['id'] for e in result['evidence']}
    assert sum(e['stage'] == 'Search missing periods' for e in result['events']) == 1


def test_json_parser_excludes_private_reasoning():
    response = '<think>Internal draft {wrong}</think>```json\n' + json.dumps(dict(headline='Memo', summary='Only final output', findings=[], limitations=[])) + '\n```'
    assert parse_response(response)['summary'] == 'Only final output'
    with pytest.raises(ValueError):
        parse_response('There is no structured memo here')


def test_decimal_calculation_handles_zero_and_negative_baseline():
    calculation = Calculation(label='Loss change', earlier=Decimal('-10'), later=Decimal('-5'), unit='USD million', earlier_period='2024', later_period='2025', source_ids=['e1', 'e2'])
    assert calculation.result()['percentage_change'] == '50.00'
    assert calculation.result()['absolute_change'] == '5'
    calculation.earlier = Decimal('0')
    with pytest.raises(ValueError, match='zero baseline'):
        calculation.result()
