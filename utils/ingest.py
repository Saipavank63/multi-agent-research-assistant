import uuid
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag.vector_store import upsert_documents

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)


def ingest_file(filepath: str):
    path = Path(filepath)
    text = path.read_text(encoding="utf-8")
    chunks = splitter.split_text(text)

    docs = [
        {"id": str(uuid.uuid4()), "text": chunk, "source": path.name}
        for chunk in chunks
    ]

    upsert_documents(docs)
    print(f"Ingested {len(docs)} chunks from {path.name}")


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        ingest_file(f)
