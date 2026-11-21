"""Read PDFs from data/, split into overlapping chunks, embed, save index/.

Usage: python ingest.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
INDEX_DIR = Path("index")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_WORDS = 180
OVERLAP_WORDS = 40


def chunk_words(words, size, overlap):
    step = size - overlap
    for start in range(0, max(len(words), 1), step):
        piece = words[start:start + size]
        if len(piece) >= 30:  # skip tiny fragments (page numbers, headers)
            yield " ".join(piece)
        if start + size >= len(words):
            break


def main():
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit("No PDFs found in data/. See README for what to download.")

    chunks = []
    for pdf in pdfs:
        reader = PdfReader(str(pdf))
        # "page" = position in the file (1, 2, 3...).
        # "label" = the page number a PDF viewer shows, when the PDF defines
        # its own numbering (e.g. i, ii, 1, 2, A1). Without labels it equals "page".
        labels = list(reader.page_labels)
        for page_no, page in enumerate(reader.pages, start=1):
            label = labels[page_no - 1] if page_no - 1 < len(labels) else str(page_no)
            text = (page.extract_text() or "").replace("\n", " ")
            words = text.split()
            for piece in chunk_words(words, CHUNK_WORDS, OVERLAP_WORDS):
                chunks.append(
                    {"source": pdf.name, "page": page_no, "label": label, "text": piece}
                )
        print(f"{pdf.name}: {len(reader.pages)} pages, first labels: {labels[:10]}")

    print(f"{len(chunks)} chunks, embedding with {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(
        [c["text"] for c in chunks],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    INDEX_DIR.mkdir(exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", emb.astype("float32"))
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved index to {INDEX_DIR}/")


if __name__ == "__main__":
    main()
