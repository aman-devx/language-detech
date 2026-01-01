from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "lang_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

app = FastAPI(title="Language Detection API")

# CORS (IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "API running"}

@app.post("/predict")
def predict_language(data: TextInput):
    vector = vectorizer.transform([data.text])
    prediction = model.predict(vector)
    proba = model.predict_proba(vector)

    confidence = round(float(proba.max()) * 100, 2)

    return {
        "text": data.text,
        "language": prediction[0],
        "confidence": confidence
    }
