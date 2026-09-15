"""
ingest.py
Reads all PDFs under hospital_knowledge_base/ (including subfolders),
splits them into overlapping text chunks, embeds them, and saves:
  - a FAISS index (faiss_index/) for similarity search
  - a metadata.jsonl file (faiss_index/metadata.jsonl) with the full
    text, department (subfolder), and source file for every chunk,
    for auditing, debugging, and citing sources later.

Usage:
    python ingest.py
"""

import os
import json
import glob
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# ---------- Config ----------
SOURCE_DIR = "hospital_knowledge_base"
INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_pdfs(source_dir):
    """Recursively find all PDFs and extract text page by page,
    keeping track of which subfolder (department) and file each chunk came from."""
    pdf_paths = glob.glob(os.path.join(source_dir, "**", "*.pdf"), recursive=True)
    print(f"Found {len(pdf_paths)} PDF files under '{source_dir}'")

    documents = []
    for path in pdf_paths:
        try:
            reader = PdfReader(path)
        except Exception as e:
            print(f"  ! Skipping {path} (could not open: {e})")
            continue

        # top-level subfolder name relative to SOURCE_DIR, used as the "department" tag
        rel_path = os.path.relpath(path, source_dir)
        rel_parts = os.path.normpath(rel_path).split(os.sep)
        department = rel_parts[0] if len(rel_parts) > 1 else "root"
        filename = os.path.basename(path)

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source_file": filename,
                        "department": department,
                        "page": page_num,
                        "full_path": rel_path,
                    },
                )
            )

    print(f"Extracted text from {len(documents)} non-empty pages")
    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def save_metadata(chunks, chunk_ids, index_dir):
    """Write one JSON line per chunk: id, full text, department, source file, page.
    This mirrors what's inside the FAISS docstore, but in a plain, greppable
    file you can open without loading the index."""
    metadata_path = os.path.join(index_dir, "metadata.jsonl")
    with open(metadata_path, "w", encoding="utf-8") as f:
        for chunk_id, chunk in zip(chunk_ids, chunks):
            record = {
                "id": chunk_id,
                "text": chunk.page_content,
                "department": chunk.metadata.get("department"),
                "source_file": chunk.metadata.get("source_file"),
                "page": chunk.metadata.get("page"),
                "full_path": chunk.metadata.get("full_path"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(chunks)} chunk records to '{metadata_path}'")


def build_and_save_index(chunks):
    print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Explicit ids so the metadata.jsonl entries line up with the FAISS docstore
    chunk_ids = [f"chunk_{i:06d}" for i in range(len(chunks))]

    print("Embedding chunks and building FAISS index (this may take a while)...")
    vectorstore = FAISS.from_documents(chunks, embeddings, ids=chunk_ids)

    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)
    print(f"Saved FAISS index to '{INDEX_DIR}/'")

    save_metadata(chunks, chunk_ids, INDEX_DIR)


def main():
    if not os.path.isdir(SOURCE_DIR):
        raise FileNotFoundError(
            f"'{SOURCE_DIR}' not found. Make sure the Drive folder was downloaded first."
        )

    documents = load_pdfs(SOURCE_DIR)
    if not documents:
        raise ValueError("No text extracted from any PDFs. Check that the PDFs aren't scanned images.")

    chunks = chunk_documents(documents)
    build_and_save_index(chunks)
    print("\nDone. faiss_index/ now contains:")
    print("  - index.faiss, index.pkl   (the vector index + docstore, for search)")
    print("  - metadata.jsonl           (plain-text chunk + department + source file, for reference)")
    print("\nLoad the index with:")
    print("  FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)")


if __name__ == "__main__":
    main()
