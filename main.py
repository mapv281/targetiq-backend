# 📦 FastAPI backend - Bullet Hole Detection using OpenAI Vision (Improved with Error Logging)

from fastapi import FastAPI, UploadFile, File, HTTPException
#from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import uuid
import os
import openai
import base64
import logging

app = FastAPI()

origins = [
    "http://localhost:3000",  # Your local frontend development server
    "https://targetiq-frontend-f2p3axuf1-mauricios-projects-1565b5ab.vercel.app",
    "https://targetiq-frontend.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins= origins,
    allow_credentials=True,
    allow_methods=["POST", "GET", "HEAD", "OPTIONS"],
    allow_headers=["*"]
)

UPLOAD_DIR = "uploaded_targets"
os.makedirs(UPLOAD_DIR, exist_ok=True)

openai.api_key = os.getenv("OPENAI_API_KEY")  # Set this in your Render environment variables

class ScoreResult(BaseModel):
    total_shots: int
    x_ring: int
    ten_ring: int
    nine_ring: int
    other_hits: int
    suggestions: list[str]

@app.post("/upload", response_model=ScoreResult)
async def upload_target(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        target_path = os.path.join(UPLOAD_DIR, f"{file_id}.jpeg")

        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = await detect_bullet_holes_with_openai(target_path)
        return result
    except Exception as e:
        logging.exception("Error occurred while processing the image")
        raise HTTPException(status_code=500, detail="An error occurred while processing the image.")

async def detect_bullet_holes_with_openai(image_path: str) -> ScoreResult:
    try:
        with open(image_path, "rb") as img_file:
            b64_img = base64.b64encode(img_file.read()).decode("utf-8")

        prompt = (
            "You are an expert firearms instructor and target analysis AI. "
            "You are given an image of a paper shooting target. "
            "Identify and count the number of visible bullet holes. "
            "Then estimate how many of them landed in the X-ring, 10-ring, 9-ring, and outside those zones. "
            "Respond in JSON format with: total_shots, x_ring, ten_ring, nine_ring, other_hits, and a short list of suggestions."
        )

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]}
            ],
            max_tokens=500
        )

        content = response["choices"][0]["message"]["content"]
        logging.info(f"OpenAI Response: {content}")  # ADD THIS LINE
        import json
        data = json.loads(content)
        return ScoreResult(**data)

    except Exception as e:
        logging.exception("OpenAI Vision processing failed")
        raise Exception("OpenAI Vision processing failed")
