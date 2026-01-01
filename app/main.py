import os
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lang_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

# ---------------- Load Model ----------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    cv = pickle.load(f)

# ---------------- FastAPI ----------------
app = FastAPI(title="Language Detection API")

class TextRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Language Detection API is running 🚀"}

@app.post("/predict")
def predict_language(request: TextRequest):
    vector = cv.transform([request.text]).toarray()
    prediction = model.predict(vector)

    return {
        "input_text": request.text,
        "predicted_language": prediction[0]
    }
