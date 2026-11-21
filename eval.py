"""Retrieval evaluation: does the right page show up in the top-k results?

eval_questions.jsonl, one JSON object per line:
  {"q": "...", "source": "NIST.CSWP.29.pdf", "pages": ["11", "12"]}

"pages" are page LABELS (what python find_page.py shows). If a PDF defines no
labels of its own, the label is just the position in the file.
Reports hit@1, hit@3, hit@5 and lists the misses.

Usage: python eval.py
"""
import json
import sys
from pathlib import Path

import rag

KS = (1, 3, 5)
# python eval.py            -> hybrid (embeddings + keywords)
# python eval.py --dense    -> embeddings only (the baseline)
MODE = "dense" if "--dense" in sys.argv else "hybrid"
# python eval.py heldout_questions.jsonl  -> use another questions file
_files = [a for a in sys.argv[1:] if not a.startswith("--")]
QUESTIONS_FILE = _files[0] if _files else "eval_questions.jsonl"


def main():
    rows = [
        json.loads(line)
        for line in Path(QUESTIONS_FILE).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    hits = {k: 0 for k in KS}
    misses = []
    for row in rows:
        results = rag.search(row["q"], max(KS), mode=MODE)
        ranks = [
            i + 1
            for i, r in enumerate(results)
            if r["source"] == row["source"]
            and r.get("label", str(r["page"])) in {str(p) for p in row["pages"]}
        ]
        first = ranks[0] if ranks else None
        for k in KS:
            if first is not None and first <= k:
                hits[k] += 1
        if first is None or first > 3:
            misses.append((row, first, results))

    n = len(rows)
    print(f"{n} questions from {QUESTIONS_FILE}, mode={MODE}")
    for k in KS:
        print(f"hit@{k}: {hits[k]}/{n} = {hits[k] / n:.0%}")
    if misses:
        print("\nNot found in top 3:")
        for row, first, results in misses:
            print(f"\n- {row['q']}  (first correct rank: {first})")
            print(f"  expected: {row['source']} labels {row['pages']}")
            for i, r in enumerate(results, start=1):
                label = r.get("label", r["page"])
                print(
                    f"  {i}. {r['source'][:10]} label={label} "
                    f"score={r['score']:.2f}  {r['text'][:100]}..."
                )


if __name__ == "__main__":
    main()
