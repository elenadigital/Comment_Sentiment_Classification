"""
Comment Sentiment Classification API

Endpoints:
1. POST /predict_model — predict sentiment of texts
2. GET /stats — API usage statistics
3. GET /health — health check
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

# Load TF-IDF and classifier
with open("best_model_tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("best_model_clf.pkl", "rb") as f:
    clf = pickle.load(f)

# Request counter
request_count = 0

# Input model
class PredictionInput(BaseModel):
    text: list[str]  # list of texts to predict

@app.get("/stats")
def stats():
    return {"request_count": request_count}

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/predict_model")
def predict_model(input_data: PredictionInput):
    global request_count
    request_count += 1

    # Create DataFrame from input texts
    new_data = pd.DataFrame({'text': input_data.text})

    # Vectorize and predict
    X_vect = tfidf.transform(new_data['text'])
    predictions = clf.predict(X_vect)

    # Return list of predictions
    return {"predictions": predictions.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)