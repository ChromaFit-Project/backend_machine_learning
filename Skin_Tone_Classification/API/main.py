# Importing Modules
import cv2
import os
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # <-- Added for CORS
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from typing import List, Dict

# Initialize the FastAPI app
app = FastAPI(
    title="Fashion Recommendation API",
    description="An API that detects faces, analyzes skin tone, and provides clothing color recommendations.",
    version="2.0.0"
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # Allow any origin (public use)
        "https://dvaltor.com", 
        "https://app.dvaltor.com"
    ],
    allow_credentials=False,  # Must be False when allow_origins includes "*"
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- End CORS ---

# --- DNN Model Setup ---

prototxt_path = "./Models/deploy.prototxt.txt"
model_path = "./Models/res10_300x300_ssd_iter_140000.caffemodel"

if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
    raise RuntimeError("DNN model files not found. Please download 'deploy.prototxt.txt' and 'res10_300x300_ssd_iter_140000.caffemodel'.")

# Load the pre-trained Caffe model for face detection
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
print("Successfully loaded DNN face detection model.")
# --- End of DNN Model Setup ---

# --- Skin Tone Dataset and Helper Functions ---
try:
    df_recommendations = pd.read_csv("./Dataset/skin_tone_recommendations.csv")
except FileNotFoundError:
    raise RuntimeError("CSV file not found at './Dataset/skin_tone_recommendations.csv'. Please ensure the file and directory exist.")

def hex_to_rgb(hex_code: str) -> List[int]:
    hex_code = hex_code.lstrip('#')
    return [int(hex_code[i:i+2], 16) for i in (0, 2, 4)]

df_recommendations['RGB'] = df_recommendations['Hex Code'].apply(hex_to_rgb)
SKIN_TONE_RGB_LIST = np.array(df_recommendations['RGB'].tolist())

def get_average_skin_color_from_roi(face_roi: np.ndarray) -> List[int] | None:
    """Calculates the average skin color from a cropped face image (ROI)."""
    try:
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 48, 80], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        skin = cv2.bitwise_and(face_roi, face_roi, mask=mask)
        skin_rgb = cv2.cvtColor(skin, cv2.COLOR_BGR2RGB)
        
        pixels = skin_rgb.reshape((-1, 3))
        pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]
        
        if len(pixels) == 0:
            return None
            
        avg_color = np.mean(pixels, axis=0).astype(int)
        return avg_color.tolist()
    except Exception as e:
        print(f"Error during skin color analysis: {e}")
        return None

def get_recommendation_json(avg_rgb_color: List[int]) -> Dict:
    """Finds the closest skin tone and returns fashion recommendations."""
    distances = euclidean_distances([avg_rgb_color], SKIN_TONE_RGB_LIST)
    closest_index = np.argmin(distances)
    closest_match_data = df_recommendations.iloc[closest_index]

    recommendations = []
    for i in range(1, 21):  # Get up to 20 colors
        rec_name = closest_match_data.get(f'Rec_Color_{i}_Name')
        rec_hex = closest_match_data.get(f'Rec_Color_{i}_Hex')
        if pd.notna(rec_name) and pd.notna(rec_hex):
            recommendations.append({"name": rec_name, "hex": rec_hex})

    return {
        "detected_average_rgb": avg_rgb_color,
        "closest_skin_tone": {
            "description": closest_match_data['Skin Shade Description'],
            "hex_code": closest_match_data['Hex Code']
        },
        "recommended_colors": recommendations
    }
# --- End of Skin Tone Helpers ---

# Initialize the FastAPI app
app = FastAPI(
    title="Fashion Recommendation API",
    description="An API that detects faces, analyzes skin tone, and provides clothing color recommendations.",
    version="2.0.0"
)

@app.post("/analyze-fashion/")
async def detect_and_recommend(file: UploadFile = File(...)):
    """
    Accepts an image, detects faces, and for each face, provides fashion recommendations.
    """
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    original_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if original_image is None:
        raise HTTPException(status_code=400, detail="Invalid image file. Could not decode image.")

    (h, w) = original_image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(original_image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    analysis_results = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            crop_startX, crop_startY = max(0, startX), max(0, startY)
            crop_endX, crop_endY = min(w, endX), min(h, endY)

            face_roi = original_image[crop_startY:crop_endY, crop_startX:crop_endX]

            if face_roi.size > 0:
                avg_color = get_average_skin_color_from_roi(face_roi)
                
                recommendation_data = None
                if avg_color:
                    recommendation_data = get_recommendation_json(avg_color)
                else:
                    recommendation_data = {"error": "Could not determine skin tone from this face."}

                analysis_results.append({
                    "face_details": {
                        "box_coordinates": [int(startX), int(startY), int(endX), int(endY)],
                        "confidence": float(confidence)
                    },
                    "fashion_recommendation": recommendation_data
                })

    if not analysis_results:
        return JSONResponse(
            content={"error": "Could not detect any faces in the uploaded image."},
            status_code=404
        )

    return JSONResponse(content={"image_analysis_results": analysis_results})

@app.get("/")
def read_root():
    """Root endpoint with instructions."""
    return {"message": "Unlock your personal style with Dvaltor Fashion AI. Upload an image to discover the colors that make you shine."}