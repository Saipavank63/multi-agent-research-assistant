from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, TOP_K
from rag.embeddings import get_query_vectors

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


def hybrid_search(query: str, top_k: int = TOP_K) -> list[dict]:
    dense_vec, sparse_vec = get_query_vectors(query)

    results = index.query(
        vector=dense_vec,
        sparse_vector=sparse_vec,
        top_k=top_k,
        include_metadata=True
    )

    docs = []
    for match in results["matches"]:
        docs.append({
            "id": match["id"],
            "score": match["score"],
            "text": match["metadata"].get("text", ""),
            "source": match["metadata"].get("source", ""),
        })

    return docs


def rerank(query: str, docs: list[dict]) -> list[dict]:
    # simple score-based rerank; swap with Cohere rerank if needed
    return sorted(docs, key=lambda x: x["score"], reverse=True)


def upsert_documents(chunks: list[dict]):
    # each chunk: {"id": str, "text": str, "source": str}
    from rag.embeddings import get_dense, get_sparse

    texts = [c["text"] for c in chunks]
    dense_vecs = get_dense(texts)
    sparse_vecs = get_sparse(texts)

    vectors = []
    for i, chunk in enumerate(chunks):
        vectors.append({
            "id": chunk["id"],
            "values": dense_vecs[i],
            "sparse_values": sparse_vecs[i],
            "metadata": {"text": chunk["text"], "source": chunk["source"]}
        })

    index.upsert(vectors=vectors)
