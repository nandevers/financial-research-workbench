"""A curated table calculation, separate from model inference and quality scores."""
import re
from datetime import datetime, timezone

from langchain_core.tools import tool

from .contracts import Calculation


@tool
def calculate_change(label: str, earlier: str, later: str, unit: str, earlier_period: str, later_period: str, source_ids: list[str]) -> dict:
    """Calculate an explicit Decimal change from source-identified comparable values."""
    return Calculation(label=label, earlier=earlier, later=later, unit=unit, earlier_period=earlier_period, later_period=later_period, source_ids=source_ids).result()


def prepare_reference_calculations(corpus):
    # This one table mapping is curated, not a claim of general financial extraction.
    source = next((b for b in corpus['blocks'] if b['id'] == 'bac-2025-12-31-bed1b8b0995a'), None)
    if source is None:
        return []
    text = source['text']
    required = ('Three Months Ended', 'Dollars in billions', 'December 31 2025', 'December 31 2024')
    if not all(header in text for header in required):
        raise ValueError('Curated liquidity table period or units changed; review before calculation')
    row = re.search(r'Total Average Global Liquidity Sources\s+\$\s*([\d,]+)\s+\$\s*([\d,]+)', text)
    if row is None:
        raise ValueError('Curated liquidity row changed; review before calculation')
    later, earlier = (value.replace(',', '') for value in row.groups())
    result = calculate_change.invoke(dict(label='Average Global Liquidity Sources', earlier=earlier, later=later, unit='USD billions', earlier_period='Three months ended December 31, 2024', later_period='Three months ended December 31, 2025', source_ids=[source['id']]))
    return [dict(id='bac-gls-2024q4-2025q4', ticker='BAC', label='Average Global Liquidity Sources', earlier=earlier, later=later, earlier_period='Q4 2024 average', later_period='Q4 2025 average', **result, source_url=source['source_url'], source_text=text, recorded_at=datetime.now(timezone.utc).isoformat(), executor='langchain_decimal_tool_after_inference', note='One curated table mapping. Independent arithmetic reference; not supplied to the recorded model generations.')]
