import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv

# Load environment variables (API Keys)
load_dotenv(dotenv_path="../.env")


app = FastAPI(title="SmartHire AI Backend")

# 1. Load the ML Brain (XGBoost)
# Make sure your .pkl files are in the 'models/' folder
try:
    model = joblib.load('../models/xgboost_model.pkl')
    tfidf = joblib.load('../models/tfidf_vectorizer.pkl')
except Exception as e:
    print(f"Error loading models: {e}")

# 2. Setup GenAI Client
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_KEY)

# Define what the input data should look like
class ResumeRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"status": "API is running"}

@app.post("/analyze")
async def analyze_resume(request: ResumeRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")

    try:
        # --- PART 1: Machine Learning (XGBoost) ---
        # Transform the input text into numbers the model understands
        features = tfidf.transform([request.text])
        prediction = model.predict(features)[0]
        # Ensure the score is a clean integer between 0 and 100
        score = int(min(max(prediction, 0), 100))

        # --- PART 2: Generative AI (Gemini) ---
        prompt = f"""
        You are an expert technical recruiter. Analyze the following resume segment and 
        provide 3 specific, actionable bullet points to improve its impact. 
        Focus on quantification (numbers) and action verbs.
        
        Resume Text: {request.text}
        """
        
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=prompt
        )

        # --- PART 3: Return the combined result ---
        return {
            "match_score": score,
            "ai_suggestions": response.text,
            "model_used": "XGBoost + Gemini 2.0 Flash"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))