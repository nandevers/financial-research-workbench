from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_core.tools import tool


class CorpusIndex:
    """A transparent sparse baseline. No neural embedding claim is made."""
    def __init__(self, blocks, documents):
        self.blocks = blocks
        self.documents = {doc['id']: doc for doc in documents}
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=45000, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(block['text'] for block in blocks)

    def search(self, query, tickers=None, years=None, as_of=None, k=6):
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        eligible = []
        for i, block in enumerate(self.blocks):
            if tickers and block['ticker'] not in tickers:
                continue
            if years and int(block['fiscal_period'][:4]) not in years:
                continue
            # Unknown publication dates cannot establish availability at an as-of date.
            if as_of and (not block.get('filing_date') or block['filing_date'] > as_of):
                continue
            if scores[i] > 0:
                eligible.append((float(scores[i]), i))
        eligible.sort(reverse=True)
        # Avoid one filing consuming all evidence on comparative questions.
        results, per_doc = [], {}
        for score, i in eligible:
            block = self.blocks[i]
            if per_doc.get(block['document_id'], 0) >= 2:
                continue
            per_doc[block['document_id']] = per_doc.get(block['document_id'], 0) + 1
            results.append({**block, 'retrieval_score': round(score, 6)})
            if len(results) == k:
                break
        return results

    def tool(self):
        @tool
        def search_filings(query: str, tickers: list[str], years: list[int], as_of: str = '', k: int = 6) -> list[dict]:
            """Retrieve original filing passages with source, period and publication metadata."""
            return self.search(query, tickers, years, as_of or None, min(k, 10))
        return search_filings
