import os
import joblib
from fastapi import FastAPI
from google import genai
from pydantic import BaseModel

app = FastAPI()

# 1. Load the ML Brain (XGBoost)
model = joblib.load('models/xgboost_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')

# 2. Setup GenAI Client
# We will set the "GEMINI_API_KEY" later in our hosting environment
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_KEY)

class ResumeData(BaseModel):
    text: str

@app.post("/analyze")
async def analyze_resume(data: ResumeData):
    # --- PART 1: Machine Learning Scoring ---
    features = tfidf.transform([data.text])
    score = model.predict(features)[0]
    
    # --- PART 2: GenAI Enhancement ---
    prompt = f"""
    You are a professional career coach. Review this resume segment and rewrite it 
    to be more impactful using the STAR method (Situation, Task, Action, Result). 
    Resume: {data.text}
    """
    
    response = client.models.generate_content(
        model="gemini-2.0-flash", 
        contents=prompt
    )
    
    return {
        "match_score": round(float(score), 2),
        "suggested_rewrite": response.text
    }