from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import shutil
import os
from typing import List, Dict
from PIL import Image

app = FastAPI(
    title="Skin Tone and Fashion API",
    description="Upload an image to detect skin tone and get clothing color recommendations.",
    version="1.0.0"
)

# --- Load Skin Tone Dataset ---
try:
    df_recommendations = pd.read_csv("./Dataset/skin_tone_recommendations.csv")
except FileNotFoundError:
    raise RuntimeError("CSV file not found at './Dataset/skin_tone_recommendations.csv'. Please check the path.")

def hex_to_rgb(hex_code: str) -> List[int]:
    hex_code = hex_code.lstrip('#')
    return [int(hex_code[i:i+2], 16) for i in (0, 2, 4)]

df_recommendations['RGB'] = df_recommendations['Hex Code'].apply(hex_to_rgb)
SKIN_TONE_RGB_LIST = np.array(df_recommendations['RGB'].tolist())

# --- Convert Any Image to JPG ---
def convert_image_to_jpg(original_path: str, converted_path: str):
    try:
        with Image.open(original_path) as img:
            rgb_img = img.convert('RGB')  # Ensure it's in RGB mode
            rgb_img.save(converted_path, format='JPEG')
    except Exception as e:
        print(f"Image conversion failed: {e}")
        raise

# --- Get Average Skin Color from Image ---
def get_average_skin_color(image_path: str) -> List[int] | None:
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 48, 80], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin = cv2.bitwise_and(image, image, mask=mask)
        skin_rgb = cv2.cvtColor(skin, cv2.COLOR_BGR2RGB)
        pixels = skin_rgb.reshape((-1, 3))
        pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]
        if len(pixels) == 0:
            return None
        avg_color = np.mean(pixels, axis=0).astype(int)
        return avg_color.tolist()
    except Exception as e:
        print(f"Error during image processing: {e}")
        return None

# --- Unified Recommendation Function ---
def get_recommendation_json(avg_rgb_color: List[int]) -> Dict:
    distances = euclidean_distances([avg_rgb_color], SKIN_TONE_RGB_LIST)
    closest_index = np.argmin(distances)
    closest_match_data = df_recommendations.iloc[closest_index]

    recommendations = []
    for i in range(1, 21):  # Get up to 20 colors
        rec_name = closest_match_data.get(f'Rec_Color_{i}_Name')
        rec_hex = closest_match_data.get(f'Rec_Color_{i}_Hex')
        if pd.notna(rec_name) and pd.notna(rec_hex):
            recommendations.append({
                "name": rec_name,
                "hex": rec_hex
            })

    return {
        "detected_average_rgb": avg_rgb_color,
        "closest_skin_tone": {
            "description": closest_match_data['Skin Shade Description'],
            "hex_code": closest_match_data['Hex Code']
        },
        "recommended_colors": recommendations
    }

# --- API Endpoints ---

@app.post("/predict/", response_class=JSONResponse)
async def upload_and_predict(file: UploadFile = File(...)):
    temp_orig_path = f"temp_orig_{file.filename}"
    temp_jpg_path = f"temp_converted.jpg"

    try:
        with open(temp_orig_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Convert uploaded image to JPEG format
        convert_image_to_jpg(temp_orig_path, temp_jpg_path)

        avg_color = get_average_skin_color(temp_jpg_path)
        if avg_color is None:
            return JSONResponse(
                content={"error": "Could not detect skin in the uploaded image. Try another photo."},
                status_code=400
            )

        result = get_recommendation_json(avg_color)
        return JSONResponse(content=result)

    finally:
        for path in [temp_orig_path, temp_jpg_path]:
            if os.path.exists(path):
                os.remove(path)

@app.get("/", response_class=JSONResponse)
def root():
    return {"message": "Welcome to the Skin Tone Prediction and Fashion API!"}
