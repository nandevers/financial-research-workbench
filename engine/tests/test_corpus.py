from workbench.corpus import parse_html


def test_inline_xbrl_div_paragraphs_survive_without_duplicate_ancestors():
    paragraph = 'Our liquidity coverage ratio is affected by stress scenarios, deposit outflows and the composition of high quality liquid assets.'
    html = f'<html><body><div class="report"><div><span>{paragraph}</span></div><div><table><tr><td>Regulatory capital data and notes: {"material financial details " * 5}</td></tr></table></div></div></body></html>'.encode()
    blocks = list(parse_html(html))
    assert sum(paragraph in block['text'] for block in blocks) == 1
    assert any(block['kind'] == 'table' for block in blocks)
    assert not any(paragraph in block['text'] and 'Regulatory capital' in block['text'] for block in blocks)
