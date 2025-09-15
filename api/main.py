from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Crear app FastAPI
app = FastAPI(title="Modelo ML API", version="1.0")

# Ruta del modelo guardado (ajústala según tu proyecto)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "final_model.pkl")
model = joblib.load(MODEL_PATH)

# Definir el esquema de entrada con Pydantic
class InputData(BaseModel):
    # pon aquí los features que espera tu modelo
    Product: str
    State: str
    Company: str
    days_to_company: int
    Timely_response: int

@app.get("/")
def read_root():
    return {"msg": "API de predicción lista 🚀"}

@app.post("/predict")
def predict(data: InputData):
    # Convertir entrada a DataFrame
    df = pd.DataFrame([data.dict()])

    # Inferir predicción (según tu modelo)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0, 1]
        pred = int(proba >= 0.5)
    else:
        pred = int(model.predict(df)[0])
        proba = None

    return {"prediction": pred, "probability": proba}
