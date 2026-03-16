# Multi-Agent Research Assistant

A self-correcting research pipeline built with LangGraph, GPT-4o, and Pinecone. Given a natural language query, the system retrieves relevant documents, validates the findings through an LLM-based critic, and synthesizes a final answer — streamed token-by-token to the frontend via Server-Sent Events.

---

## How it works

```
User Query
    │
    ▼
┌─────────────┐
│  Researcher │  ← hybrid search (dense + sparse) on Pinecone
│   Agent     │    summarizes relevant document chunks
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Critic    │  ← evaluates research notes
│   Agent     │    returns PASS or REVISE: <reason>
└──────┬──────┘
       │
  ┌────┴────┐
  │         │
REVISE     PASS  (max 2 revision loops)
  │         │
  └──→ Researcher
            │
            ▼
┌─────────────┐
│ Synthesizer │  ← combines notes into a structured markdown answer
│   Agent     │    streamed via SSE to the client
└─────────────┘
```

The orchestrator (`agents/orchestrator.py`) manages state and routing using LangGraph's `StateGraph`. Each agent reads from and writes to a shared `ResearchState` dict — no agent calls another directly.

---

## Project structure

```
multi-agent-research-assistant/
├── README.md
├── requirements.txt
├── .env.example
├── agents/
│   ├── orchestrator.py   ← LangGraph state machine + routing logic
│   ├── researcher.py     ← retrieves + summarizes docs
│   ├── critic.py         ← validates research, decides PASS/REVISE
│   └── synthesizer.py    ← produces the final answer
├── tools/
│   └── vector_search.py  ← Pinecone hybrid search + document ingestion
└── api/
    └── main.py           ← FastAPI streaming endpoint + Lambda handler
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| LLM | GPT-4o (OpenAI) |
| Embeddings | text-embedding-3-small (OpenAI) |
| Vector store | Pinecone (hybrid index) |
| Sparse embeddings | BM25 via `pinecone-text` |
| API | FastAPI |
| Streaming | Server-Sent Events (SSE) |
| Deployment | AWS Lambda + Mangum |

---

## Setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/Saipavank63/multi-agent-research-assistant.git
cd multi-agent-research-assistant

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Configure environment variables**

```bash
cp .env.example .env
```

Open `.env` and fill in:

```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=research-docs
PINECONE_ENVIRONMENT=us-east-1-aws
```

**3. Create a Pinecone index**

In your Pinecone dashboard, create an index with:
- Dimensions: `1536` (matches `text-embedding-3-small`)
- Metric: `dotproduct` (required for hybrid search)

---

## Ingest documents

Before running queries, load your documents into Pinecone:

```bash
python -m tools.vector_search path/to/paper.txt path/to/report.pdf
```

Each file is split into 512-token chunks with 64-token overlap, embedded with both dense (OpenAI) and sparse (BM25) vectors, and upserted to Pinecone.

---

## Run locally

```bash
uvicorn api.main:app --reload
```

Send a query:

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings on transformer attention mechanisms?"}'
```

The response streams as SSE events:
```
data: {"status": "researcher", "done": false}
data: {"status": "critic", "done": false}
data: {"token": "Based ", "done": false}
data: {"token": "on the ", "done": false}
...
data: {"done": true}
```

---

## Deploy to AWS Lambda

**Package dependencies:**

```bash
pip install -r requirements.txt -t package/
cd package && zip -r ../lambda.zip . && cd ..
zip -g lambda.zip -r agents/ tools/ api/ config.py
```

**Upload to Lambda:**

```bash
aws lambda update-function-code \
  --function-name research-assistant \
  --zip-file fileb://lambda.zip
```

Set the handler to `api.main.handler` and add all environment variables from `.env` in the Lambda console.

---

## Hybrid search — why it matters

Standard RAG uses only dense (semantic) embeddings. Dense search is great for paraphrased or conceptual queries but can miss exact keyword matches. BM25 sparse vectors capture term frequency — strong for domain-specific terminology.

Sending both vectors together in a single Pinecone query combines the strengths of both:
- Dense: "what causes neural network overfitting" → finds semantically similar content even if worded differently
- Sparse: "dropout regularization L2" → finds documents with those exact terms

---

## Environment variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `PINECONE_API_KEY` | Pinecone API key |
| `PINECONE_INDEX_NAME` | Name of your Pinecone index |
| `PINECONE_ENVIRONMENT` | Pinecone environment (e.g. `us-east-1-aws`) |
