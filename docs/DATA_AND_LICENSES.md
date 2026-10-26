# Data and licensing

Project code is MIT licensed. Model weights retain their publisher's Apache-2.0 or MIT license, recorded per model in `models.json`; weight files are not distributed with this project. Attribution and original licenses must accompany redistributed weights or derivatives as required by those licenses.

The 30-document source manifest is `corpus/manifests/filings.json`. Sources are original issuer investor-relations pages and report CDNs. Freely accessible financial reports are not automatically public-domain or unrestricted open-licensed datasets. Full documents remain in ignored local storage. The public snapshot exports bounded excerpts for evidence inspection and links to originals; review excerpts and the issuer's terms before publication. This document does not assert blanket redistribution rights.

Original HTML tables retain rows locally. PDFs retain exact page provenance, with approximate text extraction order. Whitespace is normalized. Excerpts are bounded to exactly the same 1,250-character source spans supplied to model generation. They are not full-report reproductions. HTML element anchors are retained when present; not every source provides stable passage anchors.

No private correspondence, employment material, account credentials, or unrelated workspace files belong in the corpus, Colab bundle or public export. LangSmith receives only public task/source material and model output when tracing is explicitly enabled. LangSmith project sharing remains private unless separately authorized.

FinQA was considered during planning; it has **not** been imported or used to claim numerical benchmark results in this release.
