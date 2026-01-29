import streamlit as st
import joblib
import os
from google import genai
from dotenv import load_dotenv

# 1. Setup & Config
st.set_page_config(page_title="Agent", page_icon="🚀")
load_dotenv() # This looks for your API key in a .env file

# 2. Load the "Brain" (The models you uploaded to the /models folder)
@st.cache_resource # This keeps the model in memory so it's fast
def load_models():
    model = joblib.load('models/xgboost_model.pkl')
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    return model, tfidf

model, tfidf = load_models()

# 3. Setup Gemini AI
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_KEY)

# --- UI LAYOUT ---
st.title("🚀 AI: Resume Analyzer")
st.markdown("Developed with **XGBoost** and **Google Gemini 2.0**")

resume_input = st.text_area("Paste your Resume Text here:", height=250, placeholder="Example: Python developer with experience in...")

if st.button("Analyze Resume"):
    if resume_input:
        with st.spinner("Analyzing with ML & GenAI..."):
            try:
                # --- STEP 1: ML Scoring (XGBoost) ---
                features = tfidf.transform([resume_input])
                score = model.predict(features)[0]
                # Keep score within 0-100 range
                final_score = min(max(int(score), 0), 100)

                # --- STEP 2: GenAI Improvement (Gemini) ---
                prompt = f"""
                You are a technical recruiter. Provide 2 concise, high-impact suggestions 
                to improve this resume text based on industry standards:
                {resume_input}
                """
                response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)

                # --- STEP 3: Display Results ---
                st.success("Analysis Complete!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Match Score", value=f"{final_score}%")
                
                st.subheader("🤖 AI Suggestions")
                st.info(response.text)

            except Exception as e:
                st.error(f"Something went wrong: {e}")
    else:
        st.warning("Please paste some text first!")