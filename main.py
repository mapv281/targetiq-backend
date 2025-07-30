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
    #shooter profile
    #shooter_name: str
    #dominant_eye: str
    #training_goals: str
    #shooter_handedness: str
    #shooter_caliber: str
    #shooter_target_type: str
    #shooter_firearm_make: str
    #shooter_firearm_model: str
    #shooter_distance: str
    #shooter_location: str
    #analysis results
    #total_shots: int
    #x_ring: int
    #ten_ring: int
    #nine_ring: int
    #other_hits: int
    #shot_distribution_overview: str
    #coaching_analysis: list[str]
    #areas_of_improvement: list[str]
    #suggestions: list[str]
    #summary: str
    #recommendations: str
    #corrective_drills: str
    html_response: str
    
    

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
            "You are given an image of a paper shooting target from uploaded image with information about the shooter's handedness, dominant eye, distance from target, firearm make, firearm model, firearm caliber, target type, and whether the shooting range is indoor or outdoor. "
            "The shooter's information is as follows: Shooter's name is Mauricio Patino, Handedness is left, Dominant eye is left, Training goals is self-defense,  Distance from target is 7 yards, Firearm make  Glock, Firearm model is 34 Gen4, Firearm ammunition is 9mm Luger, Target type is B-3 Orange, Location is Indoor Range. "
"Provide total shots, shot group pattern, shot vertical pattern, shot distribution overview, coaching analysis, corrective drills, analysis, recommendations, and suggestions for improvement."
             #"Respond in compact JSON format like: {\"html_response\": text}  with value in HTML5 format embedded inside a <div> tag"
        )

        response = openai.ChatCompletion.create(
            model="gpt-4.1",
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
        #import json
        #data = json.loads(content)
        import json
        try:
            data = json.loads(content)
            return ScoreResult(**data)
        except json.JSONDecodeError as json_err:
            logging.error(f"JSON parsing failed: {json_err}")
            logging.error(f"Raw content: {content}")
            raise HTTPException(status_code=500, detail="Failed to parse OpenAI response as JSON.")
            
    except Exception as e:
        logging.error(f"OpenAI Vision processing failed: {str(e)}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            logging.error(f"OpenAI API response: {e.response.text}")
        raise HTTPException(status_code=500, detail=f"OpenAI Vision processing failed: {str(e)}")
