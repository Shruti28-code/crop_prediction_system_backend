from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize app
app = FastAPI()

# Allow React frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5173"] for specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input schema
class CropInput(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# Load trained model
model = joblib.load("crop_prediction_model.pkl")

# Define route
@app.post("/predict")
def predict_crop(data: CropInput):
    # Convert data to numpy array
    features = np.array([[data.nitrogen, data.phosphorus, data.potassium,
                          data.temperature, data.humidity, data.ph, data.rainfall]])
    # Predict
    prediction = model.predict(features)[0]
    return {"predicted_crop": prediction}
