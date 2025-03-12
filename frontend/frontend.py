import streamlit as st
import requests
import json
import base64
import os
from dotenv import load_dotenv

load_dotenv()

# API credentials - for production, use environment variables
API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "password")
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Set up basic auth
def get_auth_header():
    credentials = f"{API_USERNAME}:{API_PASSWORD}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    return {"Authorization": f"Basic {encoded_credentials}"}

st.set_page_config(page_title="NER App", layout="centered")
st.title("📝 Named Entity Recognition (NER) App")
st.markdown("Select a model and enter text to extract named entities.")

model_choice = st.selectbox("Choose an NER Model", ["BERT", "SpaCy"], index=0)

user_input = st.text_area("Enter text for NER prediction:", "Apple Inc. announced the iPhone 15 in California on September 12, 2023.")

if st.button("🔍 Analyze Text"):
    if not user_input.strip():
        st.warning("Please enter some text before submitting.")
    else:
        model_endpoint = "/predict/bert" if model_choice == "BERT" else "/predict/spacy"
        
        try:
            response = requests.post(
                f"{API_URL}{model_endpoint}", 
                json={"text": user_input},
                headers=get_auth_header()
            )
            
            if response.status_code == 200:
                data = response.json()
                st.subheader("🔹 Annotated Text:")
                st.markdown(f"<p style='font-size:18px; color:#333; background-color:#f0f0f0; padding:10px; border-radius:5px;'>{data['annotated_text']}</p>", unsafe_allow_html=True)
                
                st.subheader("🔹 Predictions:")
                for word, label in data["predictions"]:
                    if label != "O":
                        st.markdown(f"<span style='background-color: #FFD700; color: black; padding: 4px; border-radius: 5px;'>{word} ({label})</span> ", unsafe_allow_html=True)
                    else:
                        st.write(word, end=" ")
            elif response.status_code == 401:
                st.error("Authentication failed. Please check your API credentials.")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the API. Ensure the FastAPI backend is running.")