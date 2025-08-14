from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
import joblib
import pandas as pd
import urllib.parse

# Create FastAPI app and router

load = APIRouter()

# Load model and scaler at startup (edit file paths as needed)
model = joblib.load("phishing_model.pkl")
scaler = joblib.load("scaler.pkl")

def extract_features(url: str):
    parsed_url = urllib.parse.urlparse(url)
    return {
        'url_length': len(url),
        'is_https': 1 if parsed_url.scheme == 'https' else 0,
        'contains_verify': 1 if 'verify' in url.lower() else 0,
        'contains_login': 1 if 'login' in url.lower() else 0,
        'num_subdomains': len(parsed_url.hostname.split('.')) - 2 if parsed_url.hostname else 0,
        'contains_fake': 1 if 'fake' in url.lower() else 0,
        'contains_bank': 1 if 'bank' in url.lower() else 0
    }

class URLRequest(BaseModel):
    url: str

@load.post("/check-url")
async def check_url(request: URLRequest):
    url = request.url.strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    try:
        features = extract_features(url)
        df = pd.DataFrame([features])
        df_scaled = scaler.transform(df)
        prediction = model.predict(df_scaled)[0]
        print("features:", features)
        print("prediction output:", prediction)
        if prediction == 1:
            return {
                "url": url,
                "status": "Phishing",
                "warning": "This URL is not safe"
            }
        else:
            return {
                "url": url,
                "status": "Legitimate",
                "warning": "This URL is safe"
            }
    except Exception as e:
        print("Prediction ERROR:", str(e))
        return {
            "url": url,
            "status": None,
            "warning": f"Unable to determine. Error: {str(e)}"
        }


