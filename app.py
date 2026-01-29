import streamlit as st
import requests
import json

# Page config
st.set_page_config(page_title="SmartHire AI Agent", page_icon="🚀", layout="centered")

st.title("🚀 SmartHire AI: Resume Analyzer")
st.subheader("Level up your resume with ML & GenAI")

# Sidebar info
st.sidebar.header("About the Project")
st.sidebar.info("""
This project uses:
1. **XGBoost** for Match Scoring.
2. **FastAPI** for the Backend.
3. **Gemini 2.0** for Resume Enhancement.
4. **Streamlit** for the UI.
""")

# User Input
resume_text = st.text_area("Paste your Resume or Job Description details below:", height=250)

if st.button("Run AI Analysis"):
    if resume_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("🤖 ML model is calculating score and GenAI is writing suggestions..."):
            try:
                # IMPORTANT: This URL is for your local Codespace testing.
                # When you deploy to Hugging Face, this logic might change 
                # to call the function directly.
                backend_url = "http://localhost:8000/analyze"
                payload = {"text": resume_text}
                
                response = requests.post(backend_url, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display Match Score
                    st.success("Analysis Complete!")
                    score = result["match_score"]
                    st.metric(label="Resume Match Score", value=f"{score}%")
                    st.progress(score / 100)
                    
                    # Display AI Suggestions
                    st.markdown("### 💡 AI-Powered Suggestions")
                    st.write(result["ai_suggestions"])
                    
                else:
                    st.error(f"Backend Error: {response.status_code}")
            
            except Exception as e:
                st.error(f"Could not connect to FastAPI backend. Make sure it's running! Error: {e}")

st.divider()
st.caption("Developed for AI Engineer Portfolio | Host on GitHub")