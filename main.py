# 📦 FastAPI backend - Bullet Hole Detection using OpenAI Vision (Improved with Error Logging)

from fastapi import FastAPI, Form, UploadFile, File, HTTPException
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
    shooter_name: str
    shooter_dominant_eye: str
    shooter_training_goals: str
    shooter_handedness: str
    shooter_caliber: str
    shooter_target_type: str
    shooter_firearm_make: str
    shooter_firearm_model: str
    shooter_distance: str
    shooter_range_location: str
    #analysis results
    shot_group_pattern: str
    shot_vertical_pattern: str 
    shot_distribution_overview: str
    coaching_analysis: list[str]
    areas_of_improvement: list[str]
    suggestions: list[str]
    summary: str
    recommendations: str
    corrective_drills: str
    #html_response: str
    
    

@app.post("/upload", response_model=ScoreResult)
#async def upload_target(file: UploadFile = File(...)):
async def upload_target(
        file: UploadFile = File(...),
        first_name: str = Form(...), #"Mauricio",
        last_name: str = Form(...), #"Patino",
        handedness: str = Form(...), #"Left-handed",
        dominant_eye: str = Form(...), #"Left Eye",
        training_goals: str = Form(...), #"Self-Defense",
        distance: str = Form(...), #"7 Yards",
        firearm_make: str = Form(...), #"Glock",
        firearm_model: str = Form(...), #"34 Gen4",
        firearm_caliber: str = Form(...), #"9mm Luger",
        target_type: str = Form(...), #"B-3 Orange",
        location: str = Form(...), #"Indoor Range"
):
    try:
        file_id = str(uuid.uuid4())
        target_path = os.path.join(UPLOAD_DIR, f"{file_id}.jpeg")

        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # result = await detect_bullet_holes_with_openai(target_path)
        #with inputs
        result = await detect_bullet_holes_with_openai(target_path, shooter_name = f"{first_name} {last_name}", shooter_handedness = handedness, shooter_dominant_eye = dominant_eye, shooter_training_goals = training_goals, shooter_distance = distance, shooter_firearm_make = firearm_make, shooter_firearm_model = firearm_model, shooter_caliber = firearm_caliber, shooter_target_type = target_type, shooter_range_location = location)
        return result
    except Exception as e:
        logging.exception("Error occurred while processing the image")
        raise HTTPException(status_code=500, detail="An error occurred while processing the image.")

#with inputs
async def detect_bullet_holes_with_openai(image_path: str, shooter_name: str, shooter_handedness: str, shooter_dominant_eye: str, shooter_training_goals: str, shooter_distance: str, shooter_firearm_make: str, shooter_firearm_model: str, shooter_caliber: str, shooter_target_type: str, shooter_range_location: str) -> ScoreResult:
    try:
        with open(image_path, "rb") as img_file:
            b64_img = base64.b64encode(img_file.read()).decode("utf-8")

        prompt = (
            "You are an expert firearms instructor and target analysis AI. "
            "You are given an image of a paper shooting target from uploaded image with information about the shooter's handedness, dominant eye, distance from target, firearm make, firearm model, firearm caliber, target type, and whether the shooting range is indoor or outdoor. "
            f"The shooter's information is as follows: Shooter's name is {shooter_name}, Handedness is {shooter_handedness}, Dominant eye is {shooter_dominant_eye}, "
            f"Training goals is {shooter_training_goals}, Distance from target is {shooter_distance} yards, "
            f"Firearm make is {shooter_firearm_make}, Firearm model is {shooter_firearm_model}, "
            f"Ammunition is {shooter_caliber}, Target type is {shooter_target_type}, Target Shooting Range Location is {shooter_range_location}. "
            #test
            #"The shooter's information is as follows: Shooter's name is Mauricio Patino, Handedness is left, Dominant eye is left, Training goals is self-defense,  Distance from target is 7 yards, Firearm make  Glock, Firearm model is 34 Gen4, Firearm ammunition is 9mm Luger, Target type is B-3 Orange, Location is Indoor Range. "
            #end test
            "Provide shot group pattern, shot vertical pattern, shot distribution overview, "
            "coaching analysis, corrective drills, analysis, recommendations, suggestions, and areas of improvement. "
            "Respond ONLY in compact JSON format like: "
            "{\"shot_group_pattern\": text, \"shot_vertical_pattern\": text,\"shot_distribution_overview\":  text, "
            "\"coaching_analysis\": [\"tip1\"], \"areas_of_improvement\": [\"tip1\"], \"suggestions\": [\"tip1\"], "
            "\"summary\": text, \"shooter_handedness\": text, \"shooter_distance\": text, \"shooter_caliber\": text, "
            "\"shooter_target_type\": text, \"shooter_name\": text, \"shooter_dominant_eye\": text, "
            "\"shooter_training_goals\": text, \"shooter_firearm_make\": text, \"shooter_firearm_model\": text, "
            "\"shooter_range_location\": text, \"recommendations\": text, \"corrective_drills\": text}"
        )

        response = openai.ChatCompletion.create(
            model="gpt-4.1",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]}
            ],
            #max_tokens=500
        )

        
        #import json
        #data = json.loads(content)
        import json
        from pydantic import ValidationError
        try:
            content = response["choices"][0]["message"]["content"]
            logging.info(f"OpenAI Response: {content}") 
            data = json.loads(content)

            # Validate keys required by ScoreResult (optional)
            #expected_keys = set(ScoreResult.model_fields.keys())
            #missing_keys = expected_keys - data.keys()
            #if missing_keys:
                #logging.warning(f"Missing keys in response: {missing_keys}")
            
            # Optional: Log unexpected or missing keys
            expected_fields = set(ScoreResult.model_fields.keys())
            actual_fields = set(data.keys())
            missing = expected_fields - actual_fields
            extra = actual_fields - expected_fields

            if missing:
                logging.warning(f"Missing expected fields: {missing}")
            if extra:
                logging.warning(f"Unexpected fields returned by OpenAI: {extra}")

            #return ScoreResult(**data)
            result = ScoreResult(**data)
            return result

            #data = json.loads(content).get("html_response", "")
            #return ScoreResult(html_response=data)
        except json.JSONDecodeError as json_err:
            logging.error(f"JSON parsing failed MAPV281: {json_err}")
            logging.error(f"Raw content: {content}")
            logging.error(f"Raw response: {response}")
            raise HTTPException(status_code=500, detail="OpenAI returned invalid JSON Format MAPV281_2.")
        
        except TypeError as type_err:
            logging.error(f"Type mismatch in JSON -> ScoreResult: {type_err}")
            logging.error(f"Raw content: {content}")
            logging.error(f"Raw response: {response}")
            raise HTTPException(status_code=500, detail="Data type mismatch in OpenAI response MAPV281_3.")

    except ValidationError as ve:
        logging.error(f"Pydantic validation error: {ve}")
        raise HTTPException(status_code=500, detail=f"OpenAI response failed schema validation: {ve}")
        
    except Exception as e:
        logging.error(f"OpenAI Vision processing failed: {str(e)}")
        logging.error(f"Raw content: {content}")
        logging.error(f"Raw response: {response}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            logging.error(f"OpenAI API response: {e.response.text}")
        raise HTTPException(status_code=500, detail=f"OpenAI Vision processing failed: {str(e)}")
