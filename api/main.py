from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from agents.orchestrator import run_research_pipeline
import asyncio
import json

app = FastAPI(
    title="Multi-Agent Research Assistant",
    description="Production-grade multi-agent research system using LangGraph and RAG",
    version="1.0.0"
)


class ResearchRequest(BaseModel):
    query: str
    stream: bool = False


class ResearchResponse(BaseModel):
    query: str
    answer: str
    status: str = "success"


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "multi-agent-research-assistant"}


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        answer = run_research_pipeline(request.query)
        return ResearchResponse(query=request.query, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research pipeline error: {str(e)}")


@app.post("/research/stream")
async def research_stream(request: ResearchRequest):
    async def generate():
        try:
            answer = run_research_pipeline(request.query)
            for word in answer.split():
                yield f"data: {json.dumps({'token': word + ' '})}\n\n"
                await asyncio.sleep(0.02)
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
