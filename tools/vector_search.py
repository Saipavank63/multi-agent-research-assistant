import os
import uuid
from pathlib import Path

from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "research-docs")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TOP_K = 6

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

dense_embedder = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
bm25 = BM25Encoder()


def _query_vectors(query: str):
    dense = dense_embedder.embed_query(query)
    sparse = bm25.encode_queries(query)
    return dense, sparse


def hybrid_search(query: str, top_k: int = TOP_K) -> list[dict]:
    dense_vec, sparse_vec = _query_vectors(query)

    results = index.query(
        vector=dense_vec,
        sparse_vector=sparse_vec,
        top_k=top_k,
        include_metadata=True
    )

    return [
        {
            "id": m["id"],
            "score": m["score"],
            "text": m["metadata"].get("text", ""),
            "source": m["metadata"].get("source", ""),
        }
        for m in results["matches"]
    ]


def rerank(docs: list[dict]) -> list[dict]:
    return sorted(docs, key=lambda x: x["score"], reverse=True)


def ingest_file(filepath: str):
    path = Path(filepath)
    text = path.read_text(encoding="utf-8")
    chunks = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64).split_text(text)

    texts = chunks
    dense_vecs = dense_embedder.embed_documents(texts)
    sparse_vecs = bm25.encode_documents(texts)

    vectors = [
        {
            "id": str(uuid.uuid4()),
            "values": dense_vecs[i],
            "sparse_values": sparse_vecs[i],
            "metadata": {"text": chunks[i], "source": path.name}
        }
        for i in range(len(chunks))
    ]

    index.upsert(vectors=vectors)
    print(f"Ingested {len(vectors)} chunks from {path.name}")


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        ingest_file(f)
