import pytest

from workbench.calculations import prepare_reference_calculations


def test_curated_table_calculation_preserves_period_basis_and_units():
    source = dict(id='bac-2025-12-31-bed1b8b0995a', text='Three Months Ended (Dollars in billions) December 31 2025 December 31 2024 Total Average Global Liquidity Sources $ 975 $ 953', source_url='https://example.test/fixture')
    result = prepare_reference_calculations({'blocks': [source]})[0]
    assert result['absolute_change'] == '22'
    assert result['percentage_change'] == '2.31'
    assert result['earlier_period'] == 'Q4 2024 average'
    source['text'] = source['text'].replace('billions', 'millions')
    with pytest.raises(ValueError, match='units changed'):
        prepare_reference_calculations({'blocks': [source]})
