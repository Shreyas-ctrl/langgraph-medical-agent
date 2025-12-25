from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import build_graph

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_graph()

# 🔥 THIS IS REQUIRED
class SymptomRequest(BaseModel):
    symptom: str

@app.post("/analyze")
async def analyze(req: SymptomRequest):
    result = graph.invoke({
        "symptom": req.symptom
    })
    return {
        "category": result["category"],
        "answer": result["answer"]
    }
