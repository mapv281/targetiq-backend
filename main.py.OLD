# 📦 FastAPI backend - Bullet Hole Detection (Prototype)

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import os
import uuid
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    target_path = os.path.join(UPLOAD_DIR, f"{file_id}.jpg")
    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return detect_bullet_holes(target_path)

def detect_bullet_holes(image_path: str) -> ScoreResult:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles (holes) using Hough Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=100,
        param2=30,
        minRadius=5,
        maxRadius=15
    )

    x_ring = ten_ring = nine_ring = other_hits = 0
    total = 0

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        total = len(circles)

        for (x, y, r) in circles:
            # For prototype, simulate zone based on y position (you can enhance this)
            if y < 100:
                x_ring += 1
            elif y < 200:
                ten_ring += 1
            elif y < 300:
                nine_ring += 1
            else:
                other_hits += 1

    suggestions = []
    if other_hits > 5:
        suggestions.append("Try to improve consistency and focus on sight alignment.")
    if x_ring >= 3:
        suggestions.append("Excellent precision! Work on tightening group further.")

    return ScoreResult(
        total_shots=total,
        x_ring=x_ring,
        ten_ring=ten_ring,
        nine_ring=nine_ring,
        other_hits=other_hits,
        suggestions=suggestions
    )
