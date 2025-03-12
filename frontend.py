import streamlit as st
import requests
import json

API_URL = "http://localhost:8000/predict"

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
        response = requests.post(f"http://localhost:8000{model_endpoint}", json={"text": user_input})
        
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
        else:
            st.error("Failed to fetch predictions. Ensure the FastAPI backend is running.")
