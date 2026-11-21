"""Find which page(s) contain a phrase, to fill in eval_questions.jsonl.

Usage: python find_page.py "Address Unauthorized Assets"

Prints, for every indexed page that contains the phrase: source file,
position in the file ("pos") and the page label to use in "pages".
Run `python ingest.py` first.
"""
import json
import sys
from pathlib import Path

if len(sys.argv) < 2:
    sys.exit('Usage: python find_page.py "some phrase"')

phrase = " ".join(sys.argv[1:]).lower()
chunks = json.loads(Path("index/chunks.json").read_text(encoding="utf-8"))

seen = set()
for c in chunks:
    key = (c["source"], c["page"])
    if key in seen:
        continue
    if phrase in c["text"].lower():
        seen.add(key)
        print(f'{c["source"]}  pos={c["page"]}  label={c.get("label", c["page"])}')

if not seen:
    print("Not found. Try a shorter phrase (text can be split across chunks).")
