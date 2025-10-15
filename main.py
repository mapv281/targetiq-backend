# 📦 FastAPI backend - Bullet Hole Detection using OpenAI Vision (Improved with Error Logging)

from fastapi import FastAPI, Form, UploadFile, File, HTTPException
#from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
import shutil
import uuid
import os
import openai
import base64
import logging
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
from typing import Optional

#heatmap rendering helpers
def _normalized_to_pixels(shots: list[dict], w: int, h: int) -> list[tuple[int,int,float]]:
    pts = []
    for s in shots:
        x = max(0.0, min(1.0, float(s.get("x", 0.5))))
        y = max(0.0, min(1.0, float(s.get("y", 0.5))))
        conf = float(s.get("confidence", 1.0))
        pts.append((int(round(x * w)), int(round(y * h)), conf))
    return pts

def _encode_png_b64(image_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def _render_heatmap_overlay_b64(image_path: str, shots: list[dict], alpha: float = 0.45, max_width: int = 1600):
    """
    Returns (heatmap_png_b64, overlay_png_b64).
    - Builds density from normalized shot coords.
    - Colorizes with JET, blends onto original using alpha.
    - Optionally downsizes very large images for payload control.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to read image for heatmap rendering")

    # Optional downsize to control payload size
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    pts = _normalized_to_pixels(shots, w, h)

    # Density map
    density = np.zeros((h, w), dtype=np.float32)
    base_sigma = max(8, int(min(w, h) * 0.015))

    for (px, py, conf) in pts:
        if 0 <= py < h and 0 <= px < w:
            delta = np.zeros_like(density)
            delta[py, px] = 255.0 * max(0.1, min(1.0, conf))
            sigma = int(base_sigma)
            delta = cv2.GaussianBlur(delta, (0,0), sigmaX=sigma, sigmaY=sigma)
            delta = cv2.GaussianBlur(delta, (0,0), sigmaX=sigma*0.5, sigmaY=sigma*0.5)
            density += delta

    # Normalize and colorize
    if np.max(density) > 0:
        density = (density / np.max(density) * 255.0).astype(np.uint8)
    else:
        density = density.astype(np.uint8)

    heatmap_color = cv2.applyColorMap(density, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1.0, heatmap_color, alpha, 0)

    return _encode_png_b64(heatmap_color), _encode_png_b64(overlay)
    #end of heatmap rendering helpers

app = FastAPI()

origins = [
    #"http://localhost:3000",  # Your local frontend development server
    #"https://targetiq-frontend-f2p3axuf1-mauricios-projects-1565b5ab.vercel.app",
    #"https://targetiq-frontend.vercel.app",
    "https://preview--target-coach-ai.lovable.app/*", #accept all pages
    "https://www.vantagetarget.com/*",
    "https://*.uptimerobot.com/*"
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
STATIC_DIR = "static"
OVERLAY_DIR = os.path.join(STATIC_DIR, "overlays")
os.makedirs(OVERLAY_DIR, exist_ok=True)

# Serve static files (heatmaps)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

openai.api_key = os.getenv("OPENAI_API_KEY")  # Set this in your Render environment variables

class Shot(BaseModel):
    # normalized coordinates in [0,1], (0,0) top-left of the image
    x: float
    y: float
    confidence: Optional[float] = None

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
    # NEW: vision outputs
    shots: list[Shot] = []                # normalized shot list
    heatmap_image_b64: Optional[str] = None   # PNG, base64 (no prefix)
    overlay_image_b64: Optional[str] = None   # PNG, base64 (no prefix)   
    

@app.post("/upload", response_model=ScoreResult)
#async def upload_target(file: UploadFile = File(...)):
async def upload_target(
        file: UploadFile = File(...),
        first_name: str = Form(...), #"Mauricio",
        last_name: str = Form(...), #"Patino",
        handedness: str = Form(...), #"Left-handed",
        dominant_eye: str = Form(...), #"Left Eye",
        distance: str = Form(...), #str(25),
        location: str = Form(...), #"Indoor Range",
        training_goals: str = Form(...), #"Self-Defense",
        target_type: str = Form(...), #"B-3 Orange",
        firearm_make: str = Form(...), #"Glock",
        firearm_model: str = Form(...), #"34 Gen4",
        firearm_caliber: str = Form(...), #"9mm Luger"
):
    try:
        file_id = str(uuid.uuid4())
        target_path = os.path.join(UPLOAD_DIR, f"{file_id}.jpeg")

        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # result = await detect_bullet_holes_with_openai(target_path)
        #with inputs
        result = await detect_bullet_holes_with_openai(target_path, shooter_name = f"{first_name} {last_name}", shooter_handedness = handedness, shooter_dominant_eye = dominant_eye, shooter_training_goals = training_goals, shooter_distance = f"{distance} yards", shooter_firearm_make = firearm_make, shooter_firearm_model = firearm_model, shooter_caliber = firearm_caliber, shooter_target_type = target_type, shooter_range_location = location)
        #result = await detect_bullet_holes_with_openai(target_path, shooter_name = f"{first_name} {last_name}", shooter_handedness = handedness, shooter_dominant_eye = dominant_eye, shooter_training_goals = training_goals, shooter_firearm_make = firearm_make, shooter_firearm_model = firearm_model, shooter_caliber = firearm_caliber, shooter_target_type = target_type, shooter_range_location = location)
        return result
    except Exception as e:
        logging.exception("Error occurred while processing the image")
        raise HTTPException(status_code=500, detail="An error occurred while processing the image.")

#with inputs
async def detect_bullet_holes_with_openai(image_path: str, shooter_name: str, shooter_handedness: str, shooter_dominant_eye: str, shooter_training_goals: str, shooter_distance: str, shooter_firearm_make: str, shooter_firearm_model: str, shooter_caliber: str, shooter_target_type: str, shooter_range_location: str) -> ScoreResult:
#async def detect_bullet_holes_with_openai(image_path: str, shooter_name: str, shooter_handedness: str, shooter_dominant_eye: str, shooter_training_goals: str, shooter_firearm_make: str, shooter_firearm_model: str, shooter_caliber: str, shooter_target_type: str, shooter_range_location: str) -> ScoreResult:    
    try:
        with open(image_path, "rb") as img_file:
            b64_img = base64.b64encode(img_file.read()).decode("utf-8")

        prompt = (
    "You are an expert firearms instructor who provides NRA, USPSA, IPSC, IDPA style coaching in precision, tactical, self-defense, and personalized AI target analysis depending on the inputs provided. Provide personalized coaching style depending on the shooter's training goals input."
    "You are given an image of a paper shooting (USPSA, IPSC, IDPA, NRA style targets) or steel targets plus shooter context, do your best to detect target type from uploaded image. If the chosen Target Type is Silhouette, the look for vital zones like Head, Chest, or Center Mass and provide analysis on those specific target areas. Otherwise, look for Target Types of Bullseye or Precision Target and provide analysis accordingly."
    
"INTEGRATED ADJUSTMENT SUMMARY (apply these emphases in analysis & coaching):"
"- Grip Pressure & Finger Placement:"
  "* Avoid gripping too hard with the strong hand; let the support hand provide consistent, stabilizing pressure."
  "* Maintain repeatable trigger finger placement on the pad, ensuring straight rearward movement."
"- Trigger Control / Trigger Path Discipline:"
  "* Smooth, steady trigger press is critical."
  "* Prevent sideways pulling (causing low-left for righties, low-right for lefties)."
  "* Focus on pressing straight to the rear with uniform cadence."
"- Wrist & Forearm Alignment:"
  "* Keep wrists locked in line with the bore to absorb recoil directly back."
  "* Prevent wrist collapse or bending that dips the muzzle downward."
  "* Small angle corrections can reduce muzzle flip and lateral drift."
"- Follow-Through & Sight Maintenance:"
  "* Maintain grip and sight alignment momentarily after each shot—don’t relax or flinch."
  "* Hold through recoil to reveal errors and increase consistency."
"- Recoil Anticipation / Flinch Correction:"
  "* Diagnose and eliminate flinch or “pre-ignition push” that drives shots low."
  "* Train with Ball & Dummy drills, Dry Fire, and Wall Drill to reinforce steady input."
"- Mindset & Simplification:"
  "* Do not overcomplicate technique—correct one core element at a time."
  "* Build habits through repetitive drills that reinforce correct grip, trigger, and follow-through."
"- Corrective Drills:"
  "* Ball & Dummy Drill → fixes recoil anticipation and flinching."
  "* Wall Drill → isolates trigger press discipline."
  "* One-Hole / Dot Drill → builds precision and consistency in grip & trigger."
  "* Bill Drill, El Presidente, Mozambique → integrate speed with accuracy once fundamentals are corrected."
  "* Support-hand grip pressure tests → identify imbalance and correct low-left / low-right errors."
    
    
    "Apply analysis and coaching to these specific shooter details: "
    f"Shooter's name: {shooter_name}. Handedness: {shooter_handedness}. Dominant eye: {shooter_dominant_eye}. "
    f"Training goals: {shooter_training_goals}. Distance: {shooter_distance}. "
    f"Firearm: {shooter_firearm_make} {shooter_firearm_model}. Ammunition: {shooter_caliber}. "
    f"Target Type: {shooter_target_type}. Range: {shooter_range_location}. "
    "First, identify each bullet hole center on the target and return them as normalized coordinates, "
    "with (0,0) at the top-left of the image and (1,1) at the bottom-right. "
    "Apply this analysis and the coaching recommendations relevant to these shooting drills, include ammunition grain tips & recommendations for the caliber ammunition and shooting conditions: Ball & Dummy Drill, Dry Fire Practice, Wall Drill, One-Hole Drill, Dot Drill, Bill Drill, El Presidente, Mozambique Drill, Failure to Stop Drill, Box Drill, Accelerator Drill, Dot Torture Drill, 5x5 Drill, Bill Wilson 5x5 Classifier."
    "Respond with compact JSON ONLY, using exactly these keys and structure:\n"
    "{"
    "\"shot_group_pattern\": text, "
    "\"shot_vertical_pattern\": text, "
    "\"shot_distribution_overview\": text, "
    "\"coaching_analysis\": [\"tip1\"], "
    "\"areas_of_improvement\": [\"tip1\"], "
    "\"suggestions\": [\"tip1\"], "
    "\"summary\": text, "
    "\"shooter_handedness\": text, "
    "\"shooter_distance\": text, "
    "\"shooter_caliber\": text, "
    "\"shooter_target_type\": text, "
    "\"shooter_name\": text, "
    "\"shooter_dominant_eye\": text, "
    "\"shooter_training_goals\": text, "
    "\"shooter_firearm_make\": text, "
    "\"shooter_firearm_model\": text, "
    "\"shooter_range_location\": text, "
    "\"recommendations\": text, "
    "\"corrective_drills\": text, "
    "\"shots\": [{\"x\": number, \"y\": number, \"confidence\": number}]"
    "}\n"
    "Rules: coordinates are floats in [0,1]; do not include any extra fields or markdown."
)                

        response = openai.ChatCompletion.create(
            #model="gpt-4.1",
            #model="gpt-4.1-nano", #best for low latency, most cost-effective
            #model="gpt-4.1-mini", #Balanced for intelligence, speed, and cost
            model="gpt-5-mini",
            #model = "o4-mini", #lighter, uses less tokens, faster
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]}
            ]
            #text_format="json_schema"
            #max_tokens=500
            #text_format=ScoreResult
        )

        
        #import json
        #data = json.loads(content)
        errorResult: str = "Init Error Result"

        import json
        
        try:
            errorResult = prompt
            content = response["choices"][0]["message"]["content"]
            logging.info(f"OpenAI Response: {content}") 
            data = json.loads(content)

            #heatmap
            shots_list = data.get("shots", [])
            if not isinstance(shots_list, list):
                shots_list = []
            
            try:
                heatmap_b64, overlay_b64 = _render_heatmap_overlay_b64(
                    image_path=image_path,
                    shots=shots_list,
                    alpha=0.45,         # tweak if you want more/less heat dominance
                    max_width=1600      # cap width to keep payloads reasonable
                )
                data["heatmap_image_b64"] = heatmap_b64
                data["overlay_image_b64"] = overlay_b64
            except Exception as heat_err:
                logging.exception(f"Heatmap rendering failed: {heat_err}")
                # Don't fail the whole request just because of an overlay issue:
                data["heatmap_image_b64"] = None
                data["overlay_image_b64"] = None           
            #end of heatmap
            
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
            logging.error(f"Raw result: {errorResult}")
            raise HTTPException(status_code=500, detail="OpenAI returned invalid JSON Format MAPV281_2.")
        
        except TypeError as type_err:
            logging.error(f"Type mismatch in JSON -> ScoreResult: {type_err}")
            logging.error(f"Raw content: {content}")
            logging.error(f"Raw response: {response}")
            logging.error(f"Raw result: {errorResult}")
            raise HTTPException(status_code=500, detail="Data type mismatch in OpenAI response MAPV281_3.")

    except ValidationError as ve:
        logging.error(f"Pydantic validation error Scott Mosher: {ve}")
        logging.error(repr(ve.errors()[0]['type']))
        #logging.error(f"Raw content: {content}")
        #logging.error(f"Raw response: {response}")
        logging.error(f"Raw result: {errorResult}")
        raise HTTPException(status_code=500, detail=f"OpenAI response failed schema validation: {ve}")

    except Exception as e:
        logging.error(f"OpenAI Vision processing failed: {str(e)}")
        logging.error(f"Raw content: {content}")
        logging.error(f"Raw response: {response}")
        logging.error(f"Raw result: {errorResult}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            logging.error(f"OpenAI API response: {e.response.text}")
        raise HTTPException(status_code=500, detail=f"OpenAI Vision processing failed: {str(e)}")