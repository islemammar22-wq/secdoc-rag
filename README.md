# SecDoc RAG

Question answering over public cybersecurity standards, with page-level
citations and a measured retrieval evaluation.

## Setup

```bash
pip install -r requirements.txt
mkdir data
```

Put public PDFs in `data/`. Suggested (free, no login):
- NIST Cybersecurity Framework 2.0 (NIST CSWP 29)
- CIS Critical Security Controls v8.1.2

The PDFs are not included in this repository. Download them from NIST and CIS
(ISO 27001 and PCI DSS are paywalled, so they are not used).

## Run

```bash
python ingest.py                 # builds index/
uvicorn app:app --reload         # API on http://127.0.0.1:8000/docs
python eval.py                   # hybrid retrieval, hit@1/3/5 on eval_questions.jsonl
python eval.py --dense           # embeddings-only baseline
python eval.py heldout_questions.jsonl   # the 5 held-out questions
```

Query example:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the Govern function cover?", "k": 5}'
```

Without an LLM, `/ask` returns the top passages with source and page.
For generated answers with citations, run Ollama locally and set
`OLLAMA_MODEL=llama3.1:8b` before starting the API.

## Design

- Page-aware chunking: 180 words, 40 overlap, each chunk keeps source, position and
  page label (the number a PDF viewer shows). `python find_page.py "phrase"` finds
  the page of any phrase when writing evaluation questions.
- Embeddings: all-MiniLM-L6-v2, cosine similarity over normalized vectors.
- Hybrid retrieval: BM25 keyword search (bm25.py) merged with the embedding ranking
  by reciprocal rank fusion, because embeddings alone miss exact strings like
  "Safeguard 1.2". `python eval.py --dense` is the embeddings-only baseline;
  `python eval.py` is hybrid. `LEX_WEIGHT` in rag.py tunes the keyword share.
- Evaluation: questions with known answer pages. A question counts as a hit@k if
  a chunk from a correct page is in the top k results. This measures retrieval
  only, not the quality of generated answers.

## Evaluation questions

- `eval_questions.jsonl` (23 questions): 5 written by me from the documents, 18
  drafted with AI assistance and checked against the source pages. The first
  version of the retrieval was designed while looking at this set.
- `heldout_questions.jsonl` (5 questions): written by me from pages not used above,
  with wording cleaned up with AI help. Not used for any tuning.

## Results

| Set | Method | hit@1 | hit@3 | hit@5 |
|---|---|---|---|---|
| 23 questions | Embeddings only | 65% | 83% | 91% |
| 23 questions | Hybrid (BM25 + embeddings) | 74% | 91% | 96% |
| 5 held-out | Embeddings only | 40% | 40% | 60% |
| 5 held-out | Hybrid (BM25 + embeddings) | 60% | 80% | 80% |

Hybrid search improved every measure on both sets. Both sets are small (the
held-out one has 5 questions, so one question is 20 points), so treat these as
indications, not precise measurements.

What I tried:
- Increasing the keyword weight (`LEX_WEIGHT` 1.0 to 2.0) made results worse
  (hit@5 on the 23 questions fell from 96% to 91%), so I kept 1.0.

## Known limitations

- **Exact identifiers.** "Describe CIS Safeguard 1.2" fails in both modes. The
  keyword search alone ranks it well in a test on raw text, so the cause is
  probably how the two rankings are merged (rank-based fusion lets pages that rank
  moderately in both lists beat an exact match). Not confirmed. Next step: merge
  by normalized scores instead of ranks.
- **Table pages.** The CSF Core tables (for example the Asset Management category
  definition) are not retrieved well, in either mode.
- **Hybrid is not better on every question.** "What does the Detect function do?"
  is a top-3 hit with embeddings only but ranks 4th with hybrid.
- Retrieval only: no generated-answer evaluation yet.
