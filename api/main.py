import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mangum import Mangum
from agents.orchestrator import research_graph

app = FastAPI(title="Multi-Agent Research Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


@app.post("/research")
async def research(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    async def stream():
        state = {
            "query": req.query,
            "retrieved_docs": [],
            "research_notes": "",
            "critique": "",
            "final_answer": "",
            "iteration": 0,
            "messages": []
        }

        for step in research_graph.stream(state):
            node_name = list(step.keys())[0]
            node_output = step[node_name]

            if node_name == "synthesizer" and "final_answer" in node_output:
                for token in node_output["final_answer"].split(" "):
                    yield f"data: {json.dumps({'token': token + ' ', 'done': False})}\n\n"
            else:
                yield f"data: {json.dumps({'status': node_name, 'done': False})}\n\n"

        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok"}


handler = Mangum(app)
