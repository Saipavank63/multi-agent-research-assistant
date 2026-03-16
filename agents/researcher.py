from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY")
)

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

QUERY_EXPANSION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Generate 3 alternative search queries for the given question. "
               "Return only the queries, one per line, no numbering."),
    ("human", "Original query: {query}")
])


def expand_query(query: str) -> List[str]:
    response = llm.invoke(QUERY_EXPANSION_PROMPT.format_messages(query=query))
    alternatives = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
    return [query] + alternatives[:3]


def get_vector_store() -> PineconeVectorStore:
    return PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME", "research-assistant"),
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )


def run_researcher(query: str, critique_feedback: str = "") -> List[str]:
    refined_query = f"{query}\n\nFocus on: {critique_feedback}" if critique_feedback else query
    queries = expand_query(refined_query)
    print(f"[Researcher] Expanded to {len(queries)} query variants")

    vector_store = get_vector_store()
    all_docs = []
    seen_ids = set()

    for q in queries:
        try:
            docs = vector_store.similarity_search(q, k=5)
            for doc in docs:
                doc_id = hash(doc.page_content[:100])
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_docs.append(doc)
        except Exception as e:
            print(f"[Researcher] Search error: {e}")

    print(f"[Researcher] Retrieved {len(all_docs)} unique documents")
    return [doc.page_content for doc in all_docs[:10]]
