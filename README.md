# Multi-Agent Research Assistant

A multi-agent system that autonomously retrieves, validates, and synthesizes information across documents using LangGraph, GPT-4o, and Pinecone.

## Architecture

```
Query → Researcher → Critic → (revise loop) → Synthesizer → Streamed Answer
                ↑_______________|
```

- **Researcher**: runs hybrid dense+sparse search on Pinecone, summarizes relevant chunks
- **Critic**: validates the research notes, sends back for revision if needed (max 2 iterations)
- **Synthesizer**: produces a final markdown answer, streamed token-by-token via SSE

## Stack

- LangGraph — agent orchestration + state machine
- GPT-4o — all three agents
- Pinecone — vector store with hybrid search (dense OpenAI + sparse BM25)
- FastAPI + Mangum — REST API + AWS Lambda adapter
- SSE — streaming token output to frontend

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your keys
```

## Ingest documents

```bash
python -m utils.ingest path/to/doc.txt path/to/another.pdf
```

## Run locally

```bash
uvicorn api.main:app --reload
```

POST `http://localhost:8000/research`
```json
{"query": "What are the key findings on transformer attention?"}
```

## Deploy to AWS Lambda

```bash
pip install -r requirements.txt -t package/
cd package && zip -r ../lambda.zip . && cd ..
zip -g lambda.zip -r agents/ rag/ api/ utils/ config.py

aws lambda update-function-code \
  --function-name research-assistant \
  --zip-file fileb://lambda.zip
```

Set env vars (`OPENAI_API_KEY`, `PINECONE_API_KEY`, etc.) in Lambda console.

## Tests

```bash
pytest tests/
```
