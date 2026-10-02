# 📦 FastAPI backend - Bullet Hole Detection using OpenAI Vision (Improved with Error Logging)

from fastapi import FastAPI, Form, UploadFile, File, HTTPException
#from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError, Field
import shutil
import uuid
import os
from openai import AsyncOpenAI
import base64
import logging
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
from typing import Optional
import json

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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logging.warning("OPENAI_API_KEY is not configured")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Model configuration can be overridden in Render environment variables
VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-5.6-terra")
COMPARE_MODEL = os.getenv("OPENAI_COMPARE_MODEL", "gpt-5.6-luna")

class ComparisonRequest(BaseModel):
    current_report: dict
    previous_report: dict
    shooter_context: Optional[dict] = None

class ComparisonResponse(BaseModel):
    improved_overall: bool | None = None
    accuracy_change_mm: float | None = None
    grouping_change_mm: float | None = None
    shot_group_pattern_change: str | None = None
    vertical_pattern_change: str | None = None
    distribution_change: str | None = None
    key_differences: list[str] | None = None
    coaching_recommendations: list[str] | None = None
    drill_suggestions: list[str] | None = None
    summary: str | None = None

class Shot(BaseModel):
    # normalized coordinates in [0,1], (0,0) top-left of the image
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

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
    shots: list[Shot] = Field(default_factory=list)  # normalized shot list
    heatmap_image_b64: Optional[str] = None   # PNG, base64 (no prefix)
    overlay_image_b64: Optional[str] = None   # PNG, base64 (no prefix)   

@app.get("/ping")
def ping():
    return {"status": "ok"} 

@app.post("/upload", response_model=ScoreResult)
#async def upload_target(file: UploadFile = File()):
async def upload_target(
        file: UploadFile = File(),
        first_name: str = Form(), #"Mauricio",
        last_name: str = Form(), #"Patino",
        handedness: str = Form(), #"Left-handed",
        dominant_eye: str = Form(), #"Left Eye",
        distance: str = Form(), #str(25),
        location: str = Form(), #"Indoor Range",
        training_goals: str = Form(), #"Self-Defense",
        target_type: str = Form(), #"B-3 Orange",
        firearm_make: str = Form(), #"Glock",
        firearm_model: str = Form(), #"34 Gen4",
        firearm_caliber: str = Form(), #"9mm Luger"
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

        prompt = f"""
You are VantageTarget AI Coach, an expert firearms marksmanship instructor and
computer-vision target-analysis assistant.

Your role is to analyze an uploaded shooting-target image together with the
provided shooter context and return personalized, evidence-based coaching.

Your coaching knowledge may incorporate principles commonly used in NRA-style
precision shooting and USPSA, IPSC, and IDPA sport-shooting practice.

Adapt the coaching to the shooter's stated training goals. Focus on lawful,
safe target practice, sport shooting, marksmanship fundamentals, accuracy,
precision, consistency, and measurable improvement.

======================================================================
SHOOTER CONTEXT
======================================================================

Shooter's name: {shooter_name}
Handedness: {shooter_handedness}
Dominant eye: {shooter_dominant_eye}
Training goals: {shooter_training_goals}
Distance: {shooter_distance}
Firearm: {shooter_firearm_make} {shooter_firearm_model}
Ammunition / caliber: {shooter_caliber}
Selected target type: {shooter_target_type}
Range / location: {shooter_range_location}

Use this context when interpreting the target and generating coaching.

Do not ignore handedness. Directional interpretations must be mirrored or
adjusted appropriately for left-handed versus right-handed shooters.

======================================================================
1. TARGET IDENTIFICATION
======================================================================

Inspect the uploaded image and determine the most likely target type.

Possible target categories include:
- Bullseye / precision target
- Silhouette target
- USPSA-style target
- IPSC-style target
- IDPA-style target
- Steel target
- Other / unknown

Use the shooter's selected Target Type as useful context, but also evaluate
what is actually visible in the image.

If the selected target type and visible target appear inconsistent, base the
visual analysis primarily on what can actually be observed.

Do not invent scoring zones or target features that are not visible.

For SILHOUETTE targets:
Analyze shot placement relative to visible target regions such as head,
upper torso, chest, or center-mass regions when those areas can actually
be identified.

For BULLSEYE / PRECISION targets:
Analyze:
- relationship to the visible aiming/reference point
- group center
- group size
- horizontal dispersion
- vertical dispersion
- directional bias
- outliers
- consistency
- overall group shape

For USPSA / IPSC / IDPA-style targets:
Use visible scoring zones when they can be reliably identified.
Do not invent scoring-zone hits when boundaries cannot be determined.

For STEEL targets:
Analyze visible impact evidence only when identifiable.
Do not automatically classify the absence of a visible impact as a miss.

======================================================================
2. BULLET-HOLE DETECTION
======================================================================

Identify each bullet-hole center that can be reasonably detected in the
uploaded image.

Return every detected shot using normalized image coordinates:

- (0, 0) = top-left corner
- (1, 0) = top-right corner
- (0, 1) = bottom-left corner
- (1, 1) = bottom-right corner

For every detected shot return:

{
  "x": normalized horizontal coordinate,
  "y": normalized vertical coordinate,
  "confidence": confidence from 0.0 to 1.0
}

Only return bullet holes that have reasonable visual evidence.

Do NOT invent bullet holes to satisfy an expected shot count.

Do NOT confuse printed target markings, previous patches, tears, shadows,
staples, target graphics, scoring-zone markings, or background objects with
bullet holes.

When uncertain, lower the confidence value rather than pretending certainty.

======================================================================
3. OBSERVE BEFORE DIAGNOSING
======================================================================

Analyze the detected shots objectively BEFORE diagnosing shooter technique.

Evaluate:

- group center
- group shape
- group tightness
- horizontal dispersion
- vertical dispersion
- diagonal dispersion
- directional bias
- clustering
- outliers
- multiple apparent clusters
- consistency
- relationship to the visible aiming/reference point

Classify the dominant shot pattern when possible:

- tight centered group
- tight but displaced group
- low group
- high group
- left-biased group
- right-biased group
- vertical stringing
- horizontal stringing
- diagonal stringing
- broad/random dispersion
- central cluster with outliers
- multiple clusters
- insufficient evidence

Distinguish between ACCURACY and PRECISION.

Accuracy = how close the group is to the intended aiming/reference point.

Precision = how tightly the shots group together.

A shooter may demonstrate:
- good precision but poor accuracy
- good accuracy but poor precision
- both
- neither

======================================================================
4. DIAGNOSTIC REASONING
======================================================================

Separate OBSERVATIONS from POSSIBLE CAUSES.

A target pattern alone usually cannot prove a specific technique error.

Use cautious diagnostic language such as:

- "This pattern is consistent with..."
- "One possible contributor is..."
- "This could indicate..."
- "If this pattern repeats across multiple groups..."
- "Another possibility is..."

Never claim that one isolated shot proves a technique problem.

Confidence in a diagnosis should increase when the same pattern appears
repeatedly across several shots or groups.

Consider multiple plausible contributors before recommending a correction.

======================================================================
5. GRIP PRESSURE AND FINGER PLACEMENT
======================================================================

Evaluate whether the shot pattern could be consistent with grip-pressure
imbalance or inconsistent grip.

Coaching principles:

- Favor a consistent and repeatable grip.
- Avoid unnecessary excessive tension in the strong hand.
- The support hand should provide consistent stabilizing pressure.
- Maintain repeatable trigger-finger placement.
- Trigger movement should disturb the firearm as little as possible.

Do not automatically attribute every lateral error to grip.

======================================================================
6. TRIGGER CONTROL / TRIGGER PATH
======================================================================

Evaluate whether lateral or diagonal dispersion could be consistent with
trigger-path disturbance.

Encourage:

- smooth and repeatable trigger movement
- straight rearward trigger movement
- minimal disturbance to sight alignment
- consistent trigger-finger placement
- repeatable cadence

Do NOT automatically diagnose:

"low-left = trigger problem"

For a repeated directional pattern, consider several possibilities including:

- trigger press direction
- trigger-finger placement
- grip-pressure imbalance
- recoil anticipation
- wrist movement
- sight alignment
- aiming consistency
- follow-through
- fatigue
- cadence

Account for shooter handedness when discussing directional tendencies.

======================================================================
7. WRIST AND FOREARM ALIGNMENT
======================================================================

Consider whether inconsistent wrist or forearm alignment could contribute
to vertical or lateral dispersion.

Encourage:

- repeatable wrist alignment
- stable firearm alignment
- consistency between shots
- avoiding unnecessary wrist collapse or movement

Do not diagnose wrist movement unless the target pattern reasonably
supports that hypothesis.

======================================================================
8. RECOIL ANTICIPATION
======================================================================

Repeated low displacement may be consistent with anticipation or
pre-ignition movement.

However:

DO NOT diagnose flinching from one shot alone.

Look for repeated patterns before suggesting recoil anticipation as the
primary contributor.

When appropriate, Ball & Dummy practice may be recommended as a diagnostic
exercise to help determine whether involuntary pre-ignition movement is
occurring.

======================================================================
9. FOLLOW-THROUGH AND SIGHT MAINTENANCE
======================================================================

Encourage the shooter to maintain:

- consistent grip
- sight alignment
- visual focus
- firearm stability
- follow-through through the shot cycle

Consider inconsistent follow-through when dispersion increases during
strings of fire.

Also consider sight alignment, sight picture, optic/sight zero, aiming
reference, and visual consistency before attributing displacement entirely
to shooter technique.

======================================================================
10. PRIORITIZE THE MOST IMPORTANT CORRECTION
======================================================================

Do NOT overwhelm the shooter with every possible correction.

Determine:

1. Primary observed pattern
2. Most plausible contributor or contributors
3. Highest-priority correction
4. One secondary correction when useful
5. Best corrective drill to test the hypothesis

Prefer correcting ONE major variable at a time.

The coaching should help the shooter test whether the proposed correction
actually changes the next group.

======================================================================
11. CORRECTIVE DRILLS
======================================================================

Recommend drills only when they are relevant to the observed pattern and
the shooter's training goals.

Available drills include:

Ball & Dummy Drill
Purpose: identify anticipation or involuntary movement associated with the
trigger press.

Dry Fire Practice
Purpose: develop repeatable trigger control, grip, sight alignment, and
movement-free trigger operation.

Wall Drill
Purpose: isolate trigger movement and sight disturbance.

One-Hole Drill
Purpose: develop precision and repeatability.

Dot Drill
Purpose: develop aiming consistency, trigger control, and precision.

Support-Hand Grip Pressure Test
Purpose: experiment with grip-pressure balance and determine whether group
location or dispersion changes.

Dot Torture Drill
Purpose: evaluate fundamentals, precision, transitions, and consistency.

5x5 Drill / Bill Wilson 5x5 Classifier
Purpose: evaluate repeatable accuracy and performance across multiple
fundamental shooting tasks.

Speed-oriented sport drills such as:
- Bill Drill
- El Presidente
- Box Drill
- Accelerator Drill

should only be recommended when appropriate to the shooter's stated
sport-shooting goals and demonstrated fundamentals.

Do not automatically recommend every available drill.

Select the smallest number of drills that directly address the observed
issue.

======================================================================
12. AMMUNITION CONSIDERATIONS
======================================================================

When useful, provide general ammunition-weight considerations appropriate
to the provided caliber and target-shooting context.

Consider:

- common bullet-weight ranges for the caliber
- recoil characteristics
- precision versus practice considerations
- firearm compatibility
- distance
- target type
- shooting conditions

Do not invent ammunition specifications.

Do not assume a particular firearm supports ammunition outside its normal
manufacturer specifications.

If exact ammunition information is unavailable, clearly characterize the
recommendation as a general consideration rather than a firearm-specific
requirement.

======================================================================
13. PERSONALIZED COACHING
======================================================================

Make coaching:

- concise
- personalized
- technically grounded
- encouraging
- actionable
- easy to understand
- appropriate to the shooter's stated goals

Avoid vague coaching such as:

"Practice more."
"Improve your grip."
"Work on accuracy."

Instead explain:

WHAT was observed.
WHY it may be happening.
WHAT the shooter should change.
HOW the shooter can test the correction.
WHAT should improve on the next target if the hypothesis is correct.

Do not overcomplicate the coaching.

Correct one core element at a time whenever possible.

======================================================================
14. ANALYSIS LIMITATIONS
======================================================================

Never claim certainty when the image does not provide enough evidence.

Target analysis can identify shot patterns and suggest plausible causes,
but the target image alone cannot directly observe:

- actual grip pressure
- trigger-finger movement
- stance
- wrist movement
- recoil anticipation
- sight behavior during the shot
- shooter fatigue

Do not invent shooter behavior.

Do not invent:
- bullet holes
- distances
- target zones
- scoring results
- firearm characteristics
- ammunition characteristics
- shooter actions

When evidence is insufficient, say so within the appropriate JSON field.

======================================================================
15. RESPONSE FORMAT
======================================================================

Return compact, syntactically valid JSON ONLY.

Use EXACTLY the following keys and structure:

{
  "shot_group_pattern": "text",
  "shot_vertical_pattern": "text",
  "shot_distribution_overview": "text",
  "coaching_analysis": ["tip1"],
  "areas_of_improvement": ["tip1"],
  "suggestions": ["tip1"],
  "summary": "text",
  "shooter_handedness": "text",
  "shooter_distance": "text",
  "shooter_caliber": "text",
  "shooter_target_type": "text",
  "shooter_name": "text",
  "shooter_dominant_eye": "text",
  "shooter_training_goals": "text",
  "shooter_firearm_make": "text",
  "shooter_firearm_model": "text",
  "shooter_range_location": "text",
  "recommendations": "text",
  "corrective_drills": "text",
  "shots": [
    {
      "x": 0.0,
      "y": 0.0,
      "confidence": 0.0
    }
  ]
}

STRICT OUTPUT RULES:

- Output JSON only.
- Do not output Markdown.
- Do not use ```json code fences.
- Do not add commentary before or after the JSON.
- Do not add additional fields.
- Do not remove required fields.
- Coordinates must be floating-point numbers from 0.0 through 1.0.
- Confidence must be a floating-point number from 0.0 through 1.0.
- "shots" must be a JSON array.
- If no bullet holes can be reliably detected, return "shots": [].
- coaching_analysis must be a JSON array of strings.
- areas_of_improvement must be a JSON array of strings.
- suggestions must be a JSON array of strings.
- recommendations must be a string.
- corrective_drills must be a string.
- All shooter context fields must reflect the supplied input.
- Never use NaN, Infinity, undefined, Python None, tuples, or comments.
- Ensure the final response can be parsed directly by Python json.loads().
"""               

        response = await client.responses.create(
            model=VISION_MODEL,
            reasoning={"effort": "medium"},
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{b64_img}",
                            "detail": "high",
                        },
                    ],
                }
            ],
            text={"format": {"type": "json_object"}},
        )


        #import json
        #data = json.loads(content)
        errorResult: str = "Init Error Result"

        import json

        try:
            errorResult = prompt
            content = response.output_text
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


@app.post("/compare")
async def compare_reports(payload: ComparisonRequest):
    """
    Lightweight comparison using OpenAI (GPT-5 mini).
    Trims heavy fields and caps output tokens to avoid rate-limit / TPM overages.
    """
    try:
        cr = payload.current_report or {}
        pr = payload.previous_report or {}
        ctx = payload.shooter_context or {}

        # Helper to keep only small, relevant fields
        def _truncate_text(s: str, n: int = 500) -> str:
            if not isinstance(s, str):
                return s
            return s[:n]

        def _slim_ai_analysis(ai: dict) -> dict:
            if not isinstance(ai, dict):
                return {}
            # Keep only small text/numeric fields if present
            keep = {
                "summary": ai.get("summary"),
                "shotPattern": ai.get("shotPattern"),
                "areasOfImprovement": ai.get("areasOfImprovement"),
                "recommendations": ai.get("recommendations"),
                "accuracy_mm": ai.get("accuracy_mm"),
                "grouping_mm": ai.get("grouping_mm"),
                "vertical_bias": ai.get("vertical_bias"),
                "horizontal_bias": ai.get("horizontal_bias"),
            }
            # Truncate long text fields defensively
            for k in ("summary", "shotPattern"):
                if keep.get(k):
                    keep[k] = _truncate_text(keep[k], 700)
            # Ensure lists are short
            for k in ("areasOfImprovement", "recommendations"):
                v = keep.get(k) or []
                if isinstance(v, list):
                    keep[k] = [ _truncate_text(str(x), 180) for x in v[:6] ]
                else:
                    keep[k] = []
            return keep

        def slim_report(r: dict) -> dict:
            # Prefer ai_analysis if available, otherwise look for equivalent keys
            ai = r.get("ai_analysis") if isinstance(r, dict) else None
            slim_ai = _slim_ai_analysis(ai or {})
            # include tiny numeric summary if present at root
            return {
                "ai": slim_ai,
                "accuracy_mm": r.get("accuracy_mm"),
                "grouping_mm": r.get("grouping_mm"),
                "target_type": r.get("target_type"),
                "target_distance": r.get("target_distance"),
            }

        cr_slim = slim_report(cr)
        pr_slim = slim_report(pr)

        # Extremely compact instructions
        system_prompt = "You are an expert firearms instructor. Compare two handgun shooting reports for coaching."
        user_prompt = (
            "Compare ONLY these trimmed summaries (current vs previous). "
            "Assess accuracy, grouping, and pattern trends. "
            "Return JSON ONLY with keys: improved_overall (bool), accuracy_change_mm (number), "
            "grouping_change_mm (number), shot_group_pattern_change (string), key_differences (array of strings), "
            "coaching_recommendations (array of strings), summary (string).\n\n"
            f"CONTEXT: {json.dumps(ctx, ensure_ascii=False)}\n\n"
            f"CURRENT_TRIM: {json.dumps(cr_slim, ensure_ascii=False)}\n\n"
            f"PREVIOUS_TRIM: {json.dumps(pr_slim, ensure_ascii=False)}"
        )

        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY environment variable")

        response = await client.responses.create(
            model=COMPARE_MODEL,
            reasoning={"effort": "low"},
            instructions=system_prompt,
            input=user_prompt,
            max_output_tokens=600,
            text={"format": {"type": "json_object"}},
        )

        content = response.output_text
        try:
            return json.loads(content)
        except Exception:
            # If the model slipped out of JSON, wrap minimally
            return {
                "improved_overall": None,
                "accuracy_change_mm": None,
                "grouping_change_mm": None,
                "shot_group_pattern_change": None,
                "key_differences": [],
                "coaching_recommendations": [],
                "summary": content.strip(),
            }

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("OpenAI comparison failed")
        raise HTTPException(status_code=500, detail=f"OpenAI comparison failed: {str(e)}")

# -----------------------------------------------------------------------------
# Shooter Performance Profile
# Added for VantageTarget longitudinal progress tracking.
# Supabase remains the system of record; the frontend sends the user's recent
# analyzed sessions to this endpoint so the API never needs the Supabase service key.
# -----------------------------------------------------------------------------
from datetime import datetime
from statistics import mean

PROFILE_MODEL = os.getenv("OPENAI_PROFILE_MODEL", COMPARE_MODEL)

class PerformanceSession(BaseModel):
    id: Optional[str] = None
    analyzed_at: Optional[str] = None
    accuracy_score: Optional[float] = None
    grouping_mm: Optional[float] = None
    accuracy_mm: Optional[float] = None
    shot_count: Optional[int] = None
    horizontal_bias: Optional[float] = None
    vertical_bias: Optional[float] = None
    shot_group_pattern: Optional[str] = None
    target_distance: Optional[float] = None
    target_type: Optional[str] = None
    firearm_label: Optional[str] = None

class PerformanceProfileRequest(BaseModel):
    shooter_name: Optional[str] = None
    training_goal: Optional[str] = None
    sessions: list[PerformanceSession] = Field(default_factory=list)

class PerformanceProfileResponse(BaseModel):
    sessions_analyzed: int
    current_streak: int
    accuracy_current: Optional[float] = None
    accuracy_change_pct: Optional[float] = None
    grouping_current_mm: Optional[float] = None
    grouping_change_pct: Optional[float] = None
    best_grouping_mm: Optional[float] = None
    best_accuracy_score: Optional[float] = None
    consistency_score: Optional[float] = None
    dominant_pattern: Optional[str] = None
    progress_status: str
    next_goal: str
    highlights: list[str] = Field(default_factory=list)
    ai_insight: str


def _pct_change(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    if current is None or previous in (None, 0):
        return None
    return round(((current - previous) / abs(previous)) * 100.0, 1)


def _avg(values: list[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    return round(mean(clean), 2) if clean else None


def _calculate_streak(sessions: list[PerformanceSession]) -> int:
    """Count consecutive calendar days represented by the newest sessions."""
    dates = []
    for s in sessions:
        if not s.analyzed_at:
            continue
        try:
            dates.append(datetime.fromisoformat(s.analyzed_at.replace("Z", "+00:00")).date())
        except ValueError:
            continue
    unique_dates = sorted(set(dates), reverse=True)
    if not unique_dates:
        return 0
    streak = 1
    for i in range(1, len(unique_dates)):
        if (unique_dates[i - 1] - unique_dates[i]).days == 1:
            streak += 1
        else:
            break
    return streak


def _consistency_score(groupings: list[float]) -> Optional[float]:
    """0-100 score based on coefficient of variation; higher = more consistent."""
    if len(groupings) < 2:
        return None
    avg = float(np.mean(groupings))
    if avg <= 0:
        return None
    cv = float(np.std(groupings)) / avg
    return round(max(0.0, min(100.0, 100.0 * (1.0 - cv))), 1)


@app.post("/performance-profile", response_model=PerformanceProfileResponse)
async def build_performance_profile(payload: PerformanceProfileRequest):
    """
    Build a longitudinal Shooter Performance Profile from Supabase session history.
    Send newest-first or oldest-first; the endpoint sorts ISO timestamps when present.
    Recommended input: the most recent 25-50 completed analyses.
    """
    try:
        sessions = list(payload.sessions or [])
        if not sessions:
            return PerformanceProfileResponse(
                sessions_analyzed=0,
                current_streak=0,
                progress_status="Getting started",
                next_goal="Analyze your first target to establish a baseline.",
                highlights=[],
                ai_insight="Your performance profile will become more useful as you analyze more targets."
            )

        # Sort oldest -> newest where valid timestamps exist; preserve supplied order otherwise.
        if all(s.analyzed_at for s in sessions):
            try:
                sessions.sort(key=lambda s: datetime.fromisoformat(s.analyzed_at.replace("Z", "+00:00")))
            except ValueError:
                pass

        recent = sessions[-5:]
        prior = sessions[-10:-5]
        current = sessions[-1]

        recent_accuracy = _avg([s.accuracy_score for s in recent])
        prior_accuracy = _avg([s.accuracy_score for s in prior])
        accuracy_change = _pct_change(recent_accuracy, prior_accuracy)

        recent_grouping = _avg([s.grouping_mm for s in recent])
        prior_grouping = _avg([s.grouping_mm for s in prior])
        # For grouping, smaller is better. Report positive percentage when group size improved.
        raw_grouping_change = _pct_change(recent_grouping, prior_grouping)
        grouping_improvement = round(-raw_grouping_change, 1) if raw_grouping_change is not None else None

        grouping_values = [float(s.grouping_mm) for s in sessions if s.grouping_mm is not None and s.grouping_mm > 0]
        accuracy_values = [float(s.accuracy_score) for s in sessions if s.accuracy_score is not None]
        patterns = [s.shot_group_pattern.strip() for s in sessions if s.shot_group_pattern and s.shot_group_pattern.strip()]
        dominant_pattern = max(set(patterns), key=patterns.count) if patterns else None

        best_group = round(min(grouping_values), 2) if grouping_values else None
        best_accuracy = round(max(accuracy_values), 2) if accuracy_values else None
        consistency = _consistency_score(grouping_values[-10:])

        positive_signals = sum([
            accuracy_change is not None and accuracy_change > 2,
            grouping_improvement is not None and grouping_improvement > 2,
        ])
        negative_signals = sum([
            accuracy_change is not None and accuracy_change < -2,
            grouping_improvement is not None and grouping_improvement < -2,
        ])
        if len(sessions) < 3:
            progress_status = "Building baseline"
        elif positive_signals > negative_signals:
            progress_status = "Trending up"
        elif negative_signals > positive_signals:
            progress_status = "Needs focus"
        else:
            progress_status = "Holding steady"

        highlights = []
        if current.grouping_mm is not None and best_group is not None and abs(current.grouping_mm - best_group) < 0.001:
            highlights.append("New personal best group")
        if current.accuracy_score is not None and best_accuracy is not None and abs(current.accuracy_score - best_accuracy) < 0.001:
            highlights.append("New personal best accuracy")
        if grouping_improvement is not None and grouping_improvement > 0:
            highlights.append(f"Average group improved {grouping_improvement}% vs. the previous 5 sessions")
        if accuracy_change is not None and accuracy_change > 0:
            highlights.append(f"Average accuracy improved {accuracy_change}% vs. the previous 5 sessions")

        if best_group is not None:
            next_goal_value = round(max(1.0, best_group * 0.95), 1)
            next_goal = f"Try to set a new personal best below {next_goal_value} mm."
        elif best_accuracy is not None:
            next_goal = f"Try to beat your personal-best accuracy score of {best_accuracy}."
        else:
            next_goal = "Complete a few more analyzed sessions to unlock a personalized goal."

        # Compact AI narrative. All statistics are calculated server-side so the model
        # interprets trends rather than inventing measurements.
        stats = {
            "sessions_analyzed": len(sessions),
            "recent_accuracy_avg": recent_accuracy,
            "accuracy_change_pct": accuracy_change,
            "recent_grouping_avg_mm": recent_grouping,
            "grouping_improvement_pct": grouping_improvement,
            "best_grouping_mm": best_group,
            "best_accuracy_score": best_accuracy,
            "consistency_score": consistency,
            "dominant_pattern": dominant_pattern,
            "progress_status": progress_status,
            "training_goal": payload.training_goal,
        }

        ai_insight = "Keep logging sessions to reveal stronger performance trends."
        if OPENAI_API_KEY and len(sessions) >= 2:
            try:
                profile_prompt = (
                    "You are VantageTarget's encouraging performance coach. Interpret the supplied shooting "
                    "performance statistics without inventing numbers. Write 2 concise sentences in plain English. "
                    "Sentence 1 should celebrate or neutrally describe the most meaningful trend. Sentence 2 should "
                    "give one simple, safety-conscious practice focus. Do not repeat every metric and do not use markdown.\n"
                    f"Shooter: {payload.shooter_name or 'Shooter'}\n"
                    f"Statistics: {json.dumps(stats, ensure_ascii=False)}"
                )
                ai_response = await client.responses.create(
                    model=PROFILE_MODEL,
                    reasoning={"effort": "low"},
                    input=profile_prompt,
                    max_output_tokens=180,
                )
                if ai_response.output_text:
                    ai_insight = ai_response.output_text.strip()
            except Exception as ai_err:
                logging.warning(f"Performance profile AI insight failed; using fallback: {ai_err}")

        return PerformanceProfileResponse(
            sessions_analyzed=len(sessions),
            current_streak=_calculate_streak(sessions),
            accuracy_current=recent_accuracy,
            accuracy_change_pct=accuracy_change,
            grouping_current_mm=recent_grouping,
            grouping_change_pct=grouping_improvement,
            best_grouping_mm=best_group,
            best_accuracy_score=best_accuracy,
            consistency_score=consistency,
            dominant_pattern=dominant_pattern,
            progress_status=progress_status,
            next_goal=next_goal,
            highlights=highlights[:4],
            ai_insight=ai_insight,
        )
    except Exception as e:
        logging.exception("Performance profile generation failed")
        raise HTTPException(status_code=500, detail=f"Performance profile generation failed: {str(e)}")
