from langchain_openai import OpenAIEmbeddings
from pinecone_text.sparse import BM25Encoder
from config import OPENAI_API_KEY, EMBEDDING_MODEL

dense_embedder = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=OPENAI_API_KEY
)

# BM25 for sparse (keyword) vectors — fit on your corpus once and save/load
bm25 = BM25Encoder()


def get_dense(texts: list[str]) -> list[list[float]]:
    return dense_embedder.embed_documents(texts)


def get_sparse(texts: list[str]):
    return bm25.encode_documents(texts)


def get_query_vectors(query: str):
    dense = dense_embedder.embed_query(query)
    sparse = bm25.encode_queries(query)
    return dense, sparse
