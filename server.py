"""
Department of Aquaculture and Fisheries
Fish Nutrigenomics and AI Lab | Dr. Yathish Ramena, Director
Advanced Species Detection with Length and Weight Estimation

Now with Roboflow Integration for Largemouth Bass Detection!
Version 3.1 - Fixed: Bass shows species + confidence only (no estimated measurements)
"""

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
from ultralytics import YOLO
import numpy as np
import cv2
import base64
import tempfile
import os
import io
import requests
from datetime import datetime

# Matplotlib setup - MUST be before pyplot import
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════════
# ROBOFLOW CONFIGURATION FOR BASS DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

ROBOFLOW_API_KEY = "tya4HWqSPsfoQAmR03ES"  # Your Roboflow Private API Key
ROBOFLOW_MODEL_ENDPOINT = "https://serverless.roboflow.com/bass-fish-detection-06gec/1"

# ═══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD WEIGHTS FROM GOOGLE DRIVE (for shrimp model)
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PATH = "weights.pt"
GOOGLE_DRIVE_FILE_ID = "14VSgbeQJyBizH-wTAq36WTph1miZoG5b"

def download_weights_from_gdrive(file_id: str, destination: str):
    """Download file from Google Drive using gdown"""
    if os.path.exists(destination):
        file_size = os.path.getsize(destination)
        if file_size > 1000000:  # > 1MB means it's likely valid
            print(f"✓ Weights file already exists: {destination} ({file_size} bytes)")
            return True
        else:
            print(f"⚠ Existing file too small ({file_size} bytes), re-downloading...")
            os.remove(destination)
    
    print(f"⬇ Downloading weights from Google Drive...")
    print(f"  File ID: {file_id}")
    
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"  URL: {url}")
        output = gdown.download(url, destination, quiet=False, fuzzy=True)
        
        if output and os.path.exists(destination):
            file_size = os.path.getsize(destination)
            print(f"✓ Downloaded: {destination} ({file_size} bytes)")
            if file_size > 1000000:
                return True
            else:
                print(f"✗ File too small, might be an error page")
                return False
        else:
            print(f"✗ gdown returned: {output}")
            return False
            
    except Exception as e:
        print(f"✗ gdown failed: {e}")
        
    # Fallback: try with confirm parameter
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}&confirm=t"
        print(f"  Trying fallback URL: {url}")
        output = gdown.download(url, destination, quiet=False)
        
        if output and os.path.exists(destination) and os.path.getsize(destination) > 1000000:
            print(f"✓ Fallback download successful")
            return True
    except Exception as e:
        print(f"✗ Fallback also failed: {e}")
    
    return False

# Download weights if not present
print("=" * 60)
print("INITIALIZING AQUACULTURE VISION API")
print("=" * 60)

if not download_weights_from_gdrive(GOOGLE_DRIVE_FILE_ID, MODEL_PATH):
    print("=" * 60)
    print("ERROR: Could not download weights.pt from Google Drive!")
    print("Please check:")
    print("  1. File ID is correct: " + GOOGLE_DRIVE_FILE_ID)
    print("  2. File is shared as 'Anyone with the link'")
    print("  3. Google Drive link is accessible")
    print("=" * 60)
    raise FileNotFoundError(f"Could not download {MODEL_PATH} from Google Drive")

print(f"Loading YOLO model from {MODEL_PATH}...")
print(f"Roboflow API configured for Bass detection")

app = FastAPI(
    title="Aquaculture Vision API",
    description="AI-powered detection and biomass estimation for aquaculture species",
    version="3.1 - Bass detection shows species + confidence only"
)

# Mount static files for logo and other assets
app.mount("/static", StaticFiles(directory="."), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PATH = "weights.pt"
PIXELS_PER_MM = 6.5
CONF_THRESHOLD = 0.40
MASK_ALPHA = 0.4

# Load YOLO model for shrimp
model = YOLO(MODEL_PATH)

# ═══════════════════════════════════════════════════════════════════════════════
# SPECIES CONFIG - Length-Weight Relationships: W = a × L^b
# ═══════════════════════════════════════════════════════════════════════════════

SPECIES_CONFIG = {
    "vannamei": {
        "display_name": "Pacific White Shrimp",
        "scientific_name": "Litopenaeus vannamei",
        "weight_a": 8.54e-6,
        "weight_b": 2.997,
        "color": (0, 255, 127),
        "min_harvest_mm": 100,
        "optimal_harvest_mm": 130,
        "use_roboflow": False,
    },
    "monodon": {
        "display_name": "Tiger Shrimp",
        "scientific_name": "Penaeus monodon",
        "weight_a": 7.2e-6,
        "weight_b": 3.05,
        "color": (255, 165, 0),
        "min_harvest_mm": 120,
        "optimal_harvest_mm": 150,
        "use_roboflow": False,
    },
    "bass": {
        "display_name": "Largemouth Bass",
        "scientific_name": "Micropterus salmoides",
        "weight_a": 7.0e-6,
        "weight_b": 3.19,
        "color": (100, 149, 237),  # Cornflower blue
        "min_harvest_mm": 250,
        "optimal_harvest_mm": 350,
        "use_roboflow": True,  # Use Roboflow API for bass!
    },
    "prawn": {
        "display_name": "Giant River Prawn",
        "scientific_name": "Macrobrachium rosenbergii",
        "weight_a": 6.8e-6,
        "weight_b": 3.08,
        "color": (147, 112, 219),
        "min_harvest_mm": 150,
        "optimal_harvest_mm": 200,
        "use_roboflow": False,
    },
    "default": {
        "display_name": "Unknown Species",
        "scientific_name": "N/A",
        "weight_a": 8.54e-6,
        "weight_b": 3.0,
        "color": (0, 255, 127),
        "min_harvest_mm": 100,
        "optimal_harvest_mm": 150,
        "use_roboflow": False,
    }
}


def get_species_config(species_key: str) -> dict:
    return SPECIES_CONFIG.get(species_key, SPECIES_CONFIG["default"])


def is_target_class(class_name: str) -> bool:
    """Check if detected class is a target species"""
    if class_name == "shrimp - v1 2025-10-24 5-22pm":
        return True
    lower = class_name.lower()
    return any(kw in lower for kw in ["shrimp", "fish", "prawn", "bass"])


def max_pairwise_distance(points_xy: np.ndarray) -> float:
    if points_xy.shape[0] < 2:
        return 0.0
    diff = points_xy[:, None, :] - points_xy[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    return float(dist.max())


def estimate_weight(length_mm: float, species_config: dict) -> float:
    """W = a × L^b"""
    a = species_config["weight_a"]
    b = species_config["weight_b"]
    if length_mm <= 0:
        return 0.0
    return a * (length_mm ** b)


def get_size_category(length_mm: float, species_config: dict) -> str:
    min_harvest = species_config["min_harvest_mm"]
    optimal_harvest = species_config["optimal_harvest_mm"]
    if length_mm < min_harvest * 0.7:
        return "juvenile"
    elif length_mm < min_harvest:
        return "sub-harvest"
    elif length_mm < optimal_harvest:
        return "harvestable"
    return "optimal"


# ═══════════════════════════════════════════════════════════════════════════════
# ROBOFLOW BASS DETECTION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_bass_with_roboflow(image_path: str, confidence: float = 0.40):
    """
    Detect bass fish using Roboflow API
    Returns list of detections with bounding boxes
    """
    try:
        # Read image and encode to base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Call Roboflow API
        response = requests.post(
            ROBOFLOW_MODEL_ENDPOINT,
            params={
                "api_key": ROBOFLOW_API_KEY,
                "confidence": 40,  # Roboflow uses 0-100
            },
            data=image_data,
            headers={
                "Content-Type": "application/x-www-form-urlencoded"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Roboflow API response: {len(result.get('predictions', []))} detections")
            return result
        else:
            print(f"✗ Roboflow API error: {response.status_code} - {response.text}")
            return {"predictions": [], "error": response.text}
            
    except Exception as e:
        print(f"✗ Roboflow API exception: {str(e)}")
        return {"predictions": [], "error": str(e)}


def process_bass_image(image_path: str, species_config: dict, calibration: float):
    """
    Process a single image for bass detection using Roboflow
    Returns detection results with annotations
    NOW: Shows only species + confidence (no fake length/weight)
    """
    # Get Roboflow detections
    roboflow_result = detect_bass_with_roboflow(image_path, CONF_THRESHOLD)
    predictions = roboflow_result.get("predictions", [])
    
    # Read image for annotation
    bgr = cv2.imread(image_path)
    if bgr is None:
        return None, 0, [], [], "Could not read image"
    
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()
    
    detection_count = 0
    species_detected = []
    confidences = []
    
    for pred in predictions:
        # Get bounding box
        x = pred.get("x", 0)
        y = pred.get("y", 0)
        width = pred.get("width", 0)
        height = pred.get("height", 0)
        confidence = pred.get("confidence", 0)
        class_name = pred.get("class", "bass")
        
        detection_count += 1
        species_detected.append(class_name)
        confidences.append(confidence)
        
        # Calculate bounding box corners
        x1 = int(x - width / 2)
        y1 = int(y - height / 2)
        x2 = int(x + width / 2)
        y2 = int(y + height / 2)
        
        # Draw bounding box
        color = species_config["color"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)  # Filled for overlay effect
    
    # Blend overlay
    annotated = cv2.addWeighted(rgb, 1 - MASK_ALPHA, overlay, MASK_ALPHA, 0)
    
    # Draw bounding boxes and labels (species + confidence ONLY)
    for pred in predictions:
        x = pred.get("x", 0)
        y = pred.get("y", 0)
        width = pred.get("width", 0)
        height = pred.get("height", 0)
        confidence = pred.get("confidence", 0)
        class_name = pred.get("class", "bass")
        
        x1 = int(x - width / 2)
        y1 = int(y - height / 2)
        x2 = int(x + width / 2)
        y2 = int(y + height / 2)
        
        # Draw box outline
        cv2.rectangle(annotated, (x1, y1), (x2, y2), species_config["color"], 3)
        
        # Draw label: SPECIES + CONFIDENCE only (no length/weight)
        label = f"{class_name}"
        conf_label = f"{confidence*100:.0f}%"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Get text sizes
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        (cw, ch), _ = cv2.getTextSize(conf_label, font, 0.5, 1)
        
        # Position text above bounding box
        text_x = max(0, x1)
        text_y = max(th + 10, y1 - 10)
        
        # Background rectangle for species name
        cv2.rectangle(annotated, (text_x - 2, text_y - th - 8), (text_x + tw + 4, text_y + 4), (0, 0, 0), -1)
        # Species name
        cv2.putText(annotated, label, (text_x, text_y - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        # Confidence badge (green)
        cv2.rectangle(annotated, (text_x - 2, text_y + 6), (text_x + cw + 4, text_y + ch + 12), (0, 200, 100), -1)
        cv2.putText(annotated, conf_label, (text_x, text_y + ch + 8), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return annotated, detection_count, species_detected, confidences, None


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "ok": True,
        "model_loaded": model is not None,
        "roboflow_configured": bool(ROBOFLOW_API_KEY),
        "version": "3.1",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/species")
def list_species():
    """List all supported species configurations"""
    return {k: {
        "display_name": v["display_name"], 
        "scientific_name": v["scientific_name"],
        "detection_method": "Roboflow API" if v.get("use_roboflow") else "Local YOLO"
    } for k, v in SPECIES_CONFIG.items() if k != "default"}


@app.post("/detect")
async def detect(
    files: List[UploadFile] = File(...),
    pixels_per_mm: Optional[float] = Query(default=None),
    species: Optional[str] = Query(default="vannamei")
):
    calibration = pixels_per_mm if pixels_per_mm else PIXELS_PER_MM
    species_config = get_species_config(species)
    use_roboflow = species_config.get("use_roboflow", False)
    
    per_image = []
    all_lengths: List[float] = []
    all_weights: List[float] = []
    overall_total = 0
    
    for up in files:
        suffix = os.path.splitext(up.filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await up.read())
            image_path = tmp.name
        
        try:
            # ═══════════════════════════════════════════════════════════════════
            # ROUTE: Use Roboflow for Bass, Local YOLO for Shrimp/Prawn
            # ═══════════════════════════════════════════════════════════════════
            
            if use_roboflow:
                # BASS DETECTION via Roboflow API
                # Shows: COUNT + SPECIES + CONFIDENCE only (no length/weight)
                print(f"🐟 Using Roboflow API for bass detection: {up.filename}")
                annotated, detection_count, species_list, confidences, error = process_bass_image(
                    image_path, species_config, calibration
                )
                
                if error:
                    per_image.append({
                        "filename": up.filename,
                        "error": error,
                        "shrimp_count": 0,
                        "specimen_count": 0,
                        "average_length_mm": 0.0,
                        "lengths_mm": [],
                        "annotated_image_png_base64": ""
                    })
                    continue
                
                overall_total += detection_count
                
                # Encode annotated image
                annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                ok, png = cv2.imencode(".png", annotated_bgr)
                b64 = base64.b64encode(png.tobytes()).decode("utf-8") if ok else ""
                
                # For bass: NO length/weight data (just count and species)
                per_image.append({
                    "filename": up.filename,
                    "shrimp_count": detection_count,
                    "specimen_count": detection_count,
                    "species_detected": species_list,
                    "confidences": [round(c * 100, 1) for c in confidences],
                    "average_length_mm": 0,  # Not available for bass
                    "lengths_mm": [],  # Not available for bass
                    "weights_g": [],  # Not available for bass
                    "summary": {
                        "note": "Length/weight requires calibrated camera setup",
                        "detection_count": detection_count,
                        "species_found": list(set(species_list)),
                        "avg_confidence": round(sum(confidences) / len(confidences) * 100, 1) if confidences else 0
                    },
                    "annotated_image_png_base64": b64,
                    "detection_method": "Roboflow API"
                })
                
            else:
                # SHRIMP/PRAWN DETECTION via Local YOLO model
                # Shows: COUNT + LENGTH + WEIGHT (calibrated for shrimp tank)
                print(f"🦐 Using local YOLO model for detection: {up.filename}")
                results = model(image_path, verbose=False, conf=CONF_THRESHOLD)
                r = results[0]
                
                bgr = cv2.imread(image_path)
                if bgr is None:
                    per_image.append({
                        "filename": up.filename,
                        "error": "Could not read image.",
                        "shrimp_count": 0,
                        "average_length_mm": 0.0,
                        "lengths_mm": [],
                        "annotated_image_png_base64": ""
                    })
                    continue
                
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                overlay = rgb.copy()
                
                lengths_mm = []
                weights_g = []
                text_labels = []
                
                if r.masks is not None and r.boxes is not None:
                    for mask, box in zip(r.masks, r.boxes):
                        class_id = int(box.cls[0])
                        class_name = model.names.get(class_id, str(class_id))
                        conf = float(box.conf[0])
                        
                        if not is_target_class(class_name) or conf < CONF_THRESHOLD:
                            continue
                        if mask.xy is None or len(mask.xy) == 0:
                            continue
                        
                        pts = np.array(mask.xy[0], dtype=np.float32)
                        if pts.shape[0] < 2:
                            continue
                        
                        max_px = max_pairwise_distance(pts)
                        length_mm = max_px / calibration
                        weight_g = estimate_weight(length_mm, species_config)
                        
                        lengths_mm.append(float(length_mm))
                        weights_g.append(float(weight_g))
                        
                        # Draw mask
                        pts_int = pts.astype(np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(overlay, [pts_int], color=species_config["color"])
                        
                        # Label
                        x1 = int(box.xyxy[0][0])
                        y1 = int(box.xyxy[0][1])
                        text_labels.append((x1, y1, f"{length_mm:.1f}mm | {weight_g:.2f}g"))
                
                annotated = cv2.addWeighted(rgb, 1 - MASK_ALPHA, overlay, MASK_ALPHA, 0)
                
                # Draw text
                for (x, y, text) in text_labels:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.55
                    thickness = 2
                    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
                    x = max(0, x)
                    y = max(th + 8, y - 8)
                    cv2.rectangle(annotated, (x - 2, y - th - 8), (x + tw + 4, y + 4), (0, 0, 0), -1)
                    cv2.putText(annotated, text, (x, y - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                
                specimen_count = len(lengths_mm)
                avg_len = float(np.mean(lengths_mm)) if specimen_count > 0 else 0.0
                avg_weight = float(np.mean(weights_g)) if specimen_count > 0 else 0.0
                total_biomass = sum(weights_g)
                
                overall_total += specimen_count
                all_lengths.extend(lengths_mm)
                all_weights.extend(weights_g)
                
                # Encode image
                annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                ok, png = cv2.imencode(".png", annotated_bgr)
                b64 = base64.b64encode(png.tobytes()).decode("utf-8") if ok else ""
                
                per_image.append({
                    "filename": up.filename,
                    "shrimp_count": specimen_count,
                    "specimen_count": specimen_count,
                    "average_length_mm": round(avg_len, 2),
                    "lengths_mm": [round(x, 2) for x in lengths_mm],
                    "weights_g": [round(x, 3) for x in weights_g],
                    "summary": {
                        "average_length_mm": round(avg_len, 2),
                        "average_weight_g": round(avg_weight, 3),
                        "total_biomass_g": round(total_biomass, 3),
                    },
                    "annotated_image_png_base64": b64,
                    "detection_method": "Local YOLO"
                })
        
        except Exception as e:
            per_image.append({
                "filename": up.filename,
                "error": f"Processing failed: {str(e)}",
                "shrimp_count": 0,
                "average_length_mm": 0.0,
                "lengths_mm": [],
                "annotated_image_png_base64": ""
            })
        finally:
            try:
                os.remove(image_path)
            except:
                pass
    
    # Overall stats
    overall_avg_length = float(np.mean(all_lengths)) if all_lengths else 0.0
    overall_avg_weight = float(np.mean(all_weights)) if all_weights else 0.0
    total_biomass_g = sum(all_weights)
    
    # Size distribution (only for shrimp with actual measurements)
    size_dist = {"juvenile": 0, "sub-harvest": 0, "harvestable": 0, "optimal": 0}
    for length in all_lengths:
        cat = get_size_category(length, species_config)
        size_dist[cat] += 1
    
    total = len(all_lengths) if all_lengths else 1
    size_pct = {k: round(v / total * 100, 1) for k, v in size_dist.items()}
    
    # Generate histograms (only for shrimp with actual measurements)
    histograms = {}
    
    if all_lengths and not use_roboflow:
        try:
            # Length histogram - compact size
            fig1, ax1 = plt.subplots(figsize=(6, 3), facecolor='#F8FAFC')
            ax1.set_facecolor('#F8FAFC')
            ax1.hist(all_lengths, bins=20, color='#0066FF', edgecolor='white', alpha=0.85)
            ax1.axvline(x=overall_avg_length, color='#00C48C', linestyle='--', linewidth=2, label=f'Mean: {overall_avg_length:.1f}mm')
            ax1.set_xlabel('Length (mm)', fontsize=10, color='#0F172A')
            ax1.set_ylabel('Frequency', fontsize=10, color='#0F172A')
            ax1.set_title(f'{species_config["display_name"]} Length Distribution', fontsize=11, fontweight='bold', color='#0F172A')
            ax1.legend(loc='upper right', fontsize=8)
            ax1.tick_params(colors='#0F172A', labelsize=8)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['bottom'].set_color('#CBD5E1')
            ax1.spines['left'].set_color('#CBD5E1')
            plt.tight_layout()
            buf1 = io.BytesIO()
            fig1.savefig(buf1, format="png", dpi=120, facecolor='#F8FAFC', bbox_inches='tight')
            plt.close(fig1)
            histograms["length_histogram_base64"] = base64.b64encode(buf1.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"Length histogram error: {e}")
        
        try:
            # Weight histogram - compact size
            fig2, ax2 = plt.subplots(figsize=(6, 3), facecolor='#F8FAFC')
            ax2.set_facecolor('#F8FAFC')
            ax2.hist(all_weights, bins=20, color='#7B61FF', edgecolor='white', alpha=0.85)
            ax2.axvline(x=overall_avg_weight, color='#00C48C', linestyle='--', linewidth=2, label=f'Mean: {overall_avg_weight:.1f}g')
            ax2.set_xlabel('Weight (g)', fontsize=10, color='#0F172A')
            ax2.set_ylabel('Frequency', fontsize=10, color='#0F172A')
            ax2.set_title(f'{species_config["display_name"]} Weight Distribution', fontsize=11, fontweight='bold', color='#0F172A')
            ax2.legend(loc='upper right', fontsize=8)
            ax2.tick_params(colors='#0F172A', labelsize=8)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['bottom'].set_color('#CBD5E1')
            ax2.spines['left'].set_color('#CBD5E1')
            plt.tight_layout()
            buf2 = io.BytesIO()
            fig2.savefig(buf2, format="png", dpi=120, facecolor='#F8FAFC', bbox_inches='tight')
            plt.close(fig2)
            histograms["weight_histogram_base64"] = base64.b64encode(buf2.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"Weight histogram error: {e}")
    
    # Build response based on detection type
    if use_roboflow:
        # BASS response - no length/weight data
        return JSONResponse({
            "timestamp": datetime.now().isoformat(),
            "species": species,
            "species_info": {
                "display_name": species_config["display_name"],
                "scientific_name": species_config["scientific_name"]
            },
            "detection_method": "Roboflow API",
            "calibration_pixels_per_mm": calibration,
            "overall_summary": {
                "total_specimens": overall_total,
                "note": "Length/weight measurement requires calibrated camera setup. Currently showing detection count only.",
                "average_length_mm": 0,
                "average_weight_g": 0,
                "total_biomass_g": 0,
            },
            "histograms": {},  # No histograms for bass
            "per_image": per_image,
            # Legacy fields
            "overall_total_shrimp": overall_total,
            "overall_average_length_mm": 0,
            "histogram_png_base64": ""
        })
    else:
        # SHRIMP response - full length/weight data
        return JSONResponse({
            "timestamp": datetime.now().isoformat(),
            "species": species,
            "species_info": {
                "display_name": species_config["display_name"],
                "scientific_name": species_config["scientific_name"]
            },
            "detection_method": "Local YOLO",
            "calibration_pixels_per_mm": calibration,
            "overall_summary": {
                "total_specimens": overall_total,
                "average_length_mm": round(overall_avg_length, 2),
                "average_weight_g": round(overall_avg_weight, 3),
                "total_biomass_g": round(total_biomass_g, 3),
                "total_biomass_kg": round(total_biomass_g / 1000, 6),
                "size_distribution": {
                    "counts": size_dist,
                    "percentages": size_pct
                },
                "length_stats": {
                    "min": round(min(all_lengths), 2) if all_lengths else 0,
                    "max": round(max(all_lengths), 2) if all_lengths else 0,
                    "std": round(float(np.std(all_lengths)), 2) if all_lengths else 0
                },
                "weight_stats": {
                    "min": round(min(all_weights), 3) if all_weights else 0,
                    "max": round(max(all_weights), 3) if all_weights else 0,
                    "std": round(float(np.std(all_weights)), 3) if all_weights else 0
                }
            },
            "histograms": histograms,
            "per_image": per_image,
            # Legacy fields
            "overall_total_shrimp": overall_total,
            "overall_average_length_mm": round(overall_avg_length, 2),
            "histogram_png_base64": histograms.get("length_histogram_base64", "")
        })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=True)
