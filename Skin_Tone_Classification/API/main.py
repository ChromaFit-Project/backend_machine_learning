from fastapi import FastAPI, File, UploadFile
import joblib
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import shutil
import os

app = FastAPI()

# Load the trained model and label encoder
clf = joblib.load("./Models/skin_tone_model.pkl")
label_encoder = joblib.load("./Models/label_encoder.pkl")

# Load skin tone dataset
df_skin_tones = pd.read_csv("./Dataset/skin_shades_india.csv")

# Normalize column names (remove leading/trailing whitespace and uppercase them)
df_skin_tones.columns = [col.strip().upper() for col in df_skin_tones.columns]

# Rename necessary column to standard name
if "HEX CODE" in df_skin_tones.columns:
    df_skin_tones = df_skin_tones.rename(columns={"HEX CODE": "HEX Code"})
else:
    raise ValueError("CSV file must contain a 'HEX Code' column.")

# Convert HEX to RGB if 'RGB' column is missing
def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')  # Remove '#' if present
    return [int(hex_code[i:i+2], 16) for i in (0, 2, 4)]

if 'RGB' not in df_skin_tones.columns:
    df_skin_tones['RGB'] = df_skin_tones['HEX Code'].apply(hex_to_rgb)

# Function to extract skin region from image
def extract_skin_region(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin = cv2.bitwise_and(image, image, mask=mask)
    return skin

# Function to get average skin color
def get_average_skin_color(image):
    skin = extract_skin_region(image)
    skin_rgb = cv2.cvtColor(skin, cv2.COLOR_BGR2RGB)
    pixels = skin_rgb.reshape((-1, 3))
    pixels = pixels[np.all(pixels != [0, 0, 0], axis=1)]
    
    if len(pixels) == 0:
        return None  # No skin detected
    
    avg_color = np.mean(pixels, axis=0).astype(int)
    return avg_color.tolist()

# Predict skin tone
def predict_skin_tone(image_path):
    img = cv2.imread(image_path)
    avg_color = get_average_skin_color(img)
    
    if avg_color is None:
        return {"error": "No skin detected in the image"}
    
    predicted_label = clf.predict([avg_color])[0]
    predicted_skin_tone = label_encoder.inverse_transform([predicted_label])[0]
    
    distances = euclidean_distances([avg_color], np.array(df_skin_tones['RGB'].tolist()))
    closest_index = np.argmin(distances)
    predicted_hex_code = df_skin_tones.iloc[closest_index]['HEX Code']
    
    return {
        "Predicted Skin Tone": predicted_skin_tone,
        "Predicted HEX Code": predicted_hex_code,
        "Average RGB Color": avg_color
    }

# API endpoint
@app.post("/predict/")
async def upload_and_predict(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Predict skin tone
    result = predict_skin_tone(temp_file_path)

    # Delete the temp file
    os.remove(temp_file_path)

    return result

# Root endpoint
@app.get("/")
def root():
    return {"message": "Skin Tone Prediction API is running!"}
