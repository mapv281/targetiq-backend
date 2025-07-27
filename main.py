from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import uuid
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, use specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_targets"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class ScoreResult(BaseModel):
    total_shots: int
    x_ring: int
    ten_ring: int
    nine_ring: int
    other_hits: int
    suggestions: list[str]

@app.post("/upload", response_model=ScoreResult)
async def upload_target(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.jpg")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return simulate_scoring(file_path)

def simulate_scoring(path: str) -> ScoreResult:
    import random
    hits = [random.randint(6, 10) for _ in range(20)]
    return ScoreResult(
        total_shots=len(hits),
        x_ring=hits.count(10),
        ten_ring=hits.count(9),
        nine_ring=hits.count(8),
        other_hits=hits.count(6) + hits.count(7),
        suggestions=["Good grouping, work on consistent trigger press."]
    )
